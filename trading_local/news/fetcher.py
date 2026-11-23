"""Fetching and parsing of RSS/Atom feeds (httpx + feedparser), ported from the news probe's ``newsfetch.py``.

* ``fetch_feed``  one polite conditional GET (ETag / Last-Modified) with redirects, a total deadline and a
                  2 MB size cap; skips parsing when the body is byte-identical to the last one; parses off the
                  event loop. Never raises: every failure ends up in the returned ``FeedState`` (exponential
                  backoff), so one broken feed never affects the others.
* ``parse_feed``  bytes -> ``NewsItem`` objects with stable ``uid`` and ``title_key``, plain-text summaries and
                  robust publication dates.

Time semantics: ``fetched_ts`` (first seen) is the as-of time of an item; ``published_ts`` is what the feed
claims and is display data only. Items regularly appear 10 minutes to 4 hours after their own pubDate.
"""

import asyncio
import calendar
import contextlib
import functools
import hashlib
import html
import io
import logging
import math
import re
import time
import unicodedata
from collections import Counter
from collections.abc import AsyncIterator, Mapping
from concurrent.futures import Executor
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any, Final
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

import feedparser
import httpx

from trading_local.db.repo_news import FeedState
from trading_local.models import NewsItem
from trading_local.news.feeds import Feed

log = logging.getLogger(__name__)

# A declared reader User-Agent: a browser-like UA gets HTTP 403 from the Akamai-protected hosts (finanzen.net,
# GlobeNewswire ...). Never impersonate a browser.
USER_AGENT: Final = "Mozilla/5.0 (compatible; RSS-Reader/1.0; private feed reader)"
REQUEST_HEADERS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "User-Agent": USER_AGENT,
        "Accept": "application/rss+xml, application/atom+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.5",
        "Accept-Language": "en-US,en;q=0.8,de;q=0.6",
    }
)
HTTP_TIMEOUT: Final = httpx.Timeout(12.0, connect=5.0)  # per phase: a dead route must not cost 12 s
REQUEST_DEADLINE_S: Final = 25.0  # whole request including the body (httpx has no total timeout)
MAX_BODY_BYTES: Final = 2 * 1024 * 1024  # the largest useful feed in the probe had 575 kB
MAX_ENTRIES: Final = 200  # entries considered per document
TITLE_MAX_CHARS: Final = 400
SUMMARY_MAX_CHARS: Final = 600

# Backoff after failures: (first delay, cap) in seconds by kind of failure.
_BACKOFF_GONE: Final = (6 * 3600, 24 * 3600)  # 404 / 410: moved or gone, check a few times per day
_BACKOFF_BLOCKED: Final = (30 * 60, 6 * 3600)  # 401 / 403 / 451: do not knock every minute
_BACKOFF_RATE_LIMITED: Final = (5 * 60, 2 * 3600)  # 429
_BACKOFF_OTHER_CAP: Final = 30 * 60  # 5xx, timeouts, network and parse errors: first max(60 s, interval)
_RETRY_AFTER_CAP_S: Final = 6 * 3600


def make_client(transport: httpx.AsyncBaseTransport | None = None) -> httpx.AsyncClient:
    """The shared HTTP client of the news service (reader UA, timeouts, up to 5 redirects).

    Keep-alive outlives the one-minute poll interval so that the next round re-uses the TLS connections;
    the transport retries a failed connect once (never a sent request). ``transport`` replaces the network
    transport (tests pass an ``httpx.MockTransport``).
    """
    if transport is None:
        limits = httpx.Limits(max_connections=20, max_keepalive_connections=20, keepalive_expiry=100)
        transport = httpx.AsyncHTTPTransport(retries=1, limits=limits)  # limits only work on the transport
    return httpx.AsyncClient(
        headers=dict(REQUEST_HEADERS),
        timeout=HTTP_TIMEOUT,
        follow_redirects=True,
        max_redirects=5,
        transport=transport,
    )


# ------------------------------------------------------------------------------------------------
# Results
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FeedHints:
    """Exact identifiers the feed attaches to an item (finanznachrichten fn:isin, GlobeNewswire categories,
    Seeking Alpha symbol categories, StockTitan links, Nasdaq Trader halt symbols)."""

    tickers: tuple[str, ...] = ()
    isins: tuple[str, ...] = ()


@dataclass(slots=True)
class FetchResult:
    """Outcome of one ``fetch_feed`` call.

    ``items``         parsed items (empty unless the body changed)
    ``state``         the feed's new polling state (validators, failure count, next poll time)
    ``not_modified``  True for HTTP 304 and for a body identical to the previous one (nothing to parse)
    ``body_hash``     sha1 of the last successfully parsed body (pass it to the next call)
    ``hints``         feed-delivered identifiers by item uid
    ``error``         German error text ('' on success)
    ``unreachable``   the host did not answer (timeout, connect error): skip its other feeds this round
    """

    items: list[NewsItem]
    state: FeedState
    not_modified: bool
    body_hash: str | None = None
    hints: dict[str, FeedHints] = field(default_factory=dict)
    status: int | None = None
    error: str = ""
    unreachable: bool = False


class FeedParseError(ValueError):
    """The body is not an RSS/Atom document (HTML error page, consent wall, truncated XML ...)."""


class _BodyTooLarge(Exception):
    pass


# ------------------------------------------------------------------------------------------------
# Fetching
# ------------------------------------------------------------------------------------------------
class HostLimiter:
    """One request at a time per host, with a minimum pause between two requests to the same host.

    The pause uses the monotonic clock for pacing only; scheduling decisions use the injected time.
    """

    def __init__(self, min_gap_s: float = 1.0) -> None:
        self._min_gap_s = min_gap_s
        self._locks: dict[str, asyncio.Lock] = {}
        self._last: dict[str, float] = {}

    @contextlib.asynccontextmanager
    async def slot(self, host: str) -> AsyncIterator[None]:
        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:
            wait = self._min_gap_s - (time.monotonic() - self._last.get(host, -math.inf))
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                yield
            finally:
                self._last[host] = time.monotonic()


async def fetch_feed(
    client: httpx.AsyncClient,
    feed: Feed,
    state: FeedState,
    now_ts: int,
    *,
    body_hash: str | None = None,
    parse_executor: Executor | None = None,
) -> FetchResult:
    """Fetch and parse one feed. Never raises (except ``asyncio.CancelledError``); does not modify ``state``.

    Sends the stored validators as If-None-Match / If-Modified-Since. ``body_hash`` is the sha1 of the last
    parsed body: an identical body is not parsed again (many servers rotate validators without new content).
    Validators and the hash are only taken over after a successful parse, so a broken document is fetched
    again next time. On success the next poll is due ``feed.poll_seconds`` after ``now_ts``; failures back off
    exponentially (see ``backoff_delay``).

    Parsing is pure-Python CPU work and runs off the event loop: in ``parse_executor`` if given (the news
    service passes its single worker thread, so that parallel fetches never mean parallel CPU-bound threads),
    otherwise in a thread of the loop's default executor (like ``asyncio.to_thread``).
    """
    headers: dict[str, str] = {}
    if state.etag:
        headers["If-None-Match"] = state.etag
    if state.last_modified:
        headers["If-Modified-Since"] = state.last_modified
    try:
        async with asyncio.timeout(REQUEST_DEADLINE_S):
            async with client.stream("GET", feed.url, headers=headers) as response:
                status = response.status_code
                if status == 304:
                    validators = _validators(response, state)
                    return FetchResult([], _succeeded(state, feed, now_ts, *validators), True, body_hash, status=304)
                if status != 200:
                    retry_after = _retry_after_s(response.headers.get("retry-after"), now_ts)
                    return _failed(feed, state, now_ts, f"HTTP {status}", status=status, retry_after_s=retry_after)
                body = await _read_body(response)
                content_type = response.headers.get("content-type", "")
                final_url = str(response.url)
                etag, last_modified = _validators(response, state)
    except TimeoutError:
        return _failed(feed, state, now_ts, "Zeitüberschreitung", unreachable=True)
    except _BodyTooLarge:
        return _failed(feed, state, now_ts, f"Antwort größer als {MAX_BODY_BYTES // (1024 * 1024)} MB", status=200)
    except httpx.TimeoutException:
        return _failed(feed, state, now_ts, "Zeitüberschreitung", unreachable=True)
    except httpx.ConnectError as exc:
        return _failed(feed, state, now_ts, f"Verbindungsfehler ({type(exc).__name__})", unreachable=True)
    except httpx.HTTPError as exc:
        return _failed(feed, state, now_ts, f"Übertragungsfehler ({type(exc).__name__})")
    except Exception as exc:  # noqa: BLE001 - the module boundary: one feed must never stop the poll round
        log.exception("feed %s: unexpected error while fetching", feed.id)
        return _failed(feed, state, now_ts, f"Unerwarteter Fehler ({type(exc).__name__})")

    digest = hashlib.sha1(body, usedforsecurity=False).hexdigest()
    if digest == body_hash:
        return FetchResult([], _succeeded(state, feed, now_ts, etag, last_modified), True, digest, status=200)
    parse = functools.partial(parse_feed, body, feed, now_ts, content_type=content_type, final_url=final_url)
    try:
        items, hints = await asyncio.get_running_loop().run_in_executor(parse_executor, parse)
    except FeedParseError:
        return _failed(feed, state, now_ts, "Kein gültiger RSS/Atom-Feed", status=200)
    except Exception as exc:  # noqa: BLE001 - feedparser on hostile input must not stop the round
        log.exception("feed %s: unexpected error while parsing", feed.id)
        return _failed(feed, state, now_ts, f"Unerwarteter Fehler beim Lesen ({type(exc).__name__})", status=200)
    return FetchResult(items, _succeeded(state, feed, now_ts, etag, last_modified), False, digest, hints, status=200)


async def _read_body(response: httpx.Response) -> bytes:
    """The decoded body, aborting as soon as it exceeds ``MAX_BODY_BYTES``."""
    declared = response.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_BODY_BYTES and not response.headers.get("content-encoding"):
        raise _BodyTooLarge
    chunks: list[bytes] = []
    size = 0
    async for chunk in response.aiter_bytes():
        size += len(chunk)
        if size > MAX_BODY_BYTES:
            raise _BodyTooLarge
        chunks.append(chunk)
    return b"".join(chunks)


def _validators(response: httpx.Response, state: FeedState) -> tuple[str | None, str | None]:
    """ETag and Last-Modified of a response; a 304 without them keeps the stored ones."""
    etag = response.headers.get("etag")
    last_modified = response.headers.get("last-modified")
    if response.status_code == 304:
        return etag or state.etag, last_modified or state.last_modified
    return etag, last_modified


def _succeeded(state: FeedState, feed: Feed, now_ts: int, etag: str | None, last_modified: str | None) -> FeedState:
    return replace(
        state,
        etag=etag,
        last_modified=last_modified,
        last_ok_ts=now_ts,
        last_error="",
        fail_count=0,
        next_poll_ts=now_ts + feed.poll_seconds,
    )


def _failed(
    feed: Feed,
    state: FeedState,
    now_ts: int,
    error: str,
    *,
    status: int | None = None,
    retry_after_s: int | None = None,
    unreachable: bool = False,
) -> FetchResult:
    fail_count = state.fail_count + 1
    delay = backoff_delay(None if status == 200 else status, fail_count, feed.poll_seconds, retry_after_s)
    new_state = replace(state, last_error=error, fail_count=fail_count, next_poll_ts=now_ts + delay)
    log.info("feed %s failed (%s), attempt %d, next try in %d s", feed.id, error, fail_count, delay)
    return FetchResult([], new_state, False, None, status=status, error=error, unreachable=unreachable)


def backoff_delay(status: int | None, fail_count: int, poll_seconds: int, retry_after_s: int | None = None) -> int:
    """Seconds until the next attempt after the ``fail_count``-th failure in a row (deterministic).

    The delay doubles per failure (at most 8 doublings considered) up to a cap that depends on the failure:
    404/410 from 6 h up to 24 h, 401/403/451 from 30 min up to 6 h, 429 from 5 min up to 2 h, everything else
    (``status`` None: timeouts, network and parse errors; 5xx) from max(60 s, poll interval) up to 30 min.
    A Retry-After header is never undercut (but honoured for at most 6 h).
    """
    if status in (404, 410):
        first, cap = _BACKOFF_GONE
    elif status in (401, 403, 451):
        first, cap = _BACKOFF_BLOCKED
    elif status == 429:
        first, cap = _BACKOFF_RATE_LIMITED
    else:
        first, cap = max(60, poll_seconds), _BACKOFF_OTHER_CAP
    delay = min(cap, first * 2 ** (min(max(fail_count, 1), 8) - 1))
    if retry_after_s:
        delay = max(delay, min(retry_after_s, _RETRY_AFTER_CAP_S))
    return int(delay)


def _retry_after_s(value: str | None, now_ts: int) -> int | None:
    """Retry-After as seconds from ``now_ts`` (delta-seconds or HTTP date); None if absent or unreadable."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return int(value)
    when = parse_published(value)
    return None if when is None else max(0, int(when.timestamp()) - now_ts)


# ------------------------------------------------------------------------------------------------
# Parsing
# ------------------------------------------------------------------------------------------------
_CHARSET: Final = re.compile(r"charset\s*=\s*[\"']?([\w.:-]+)", re.I)
_STOCKTITAN_LINK: Final = re.compile(r"stocktitan\.net/news/([A-Z0-9.]{1,8})/")
_ISIN: Final = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}\d$")


def parse_feed(
    body: bytes, feed: Feed, fetched_ts: int, *, content_type: str = "", final_url: str = ""
) -> tuple[list[NewsItem], dict[str, FeedHints]]:
    """Parse an RSS/Atom document into news items (oldest first) and the feed hints by item uid.

    Raises ``FeedParseError`` when the body is no feed at all; malformed but recoverable XML is parsed
    leniently. Per entry: the title is cleaned and the feed's ``title_strip`` / ``drop_title_re`` applied
    (entries without a title are skipped), only http/https links are kept (relative links are resolved),
    the summary (or the content) becomes plain text of at most 600 characters, the publication date is
    converted to UTC epoch seconds (None if missing or unreadable, clamped to ``fetched_ts`` if in the
    future), ``lang`` is the feed's language. Entries with the same uid are reported once.
    """
    # BytesIO: feedparser would open a str/bytes payload that looks like a URL or a file name. The content type
    # is normalised: 'text/xml' without charset means us-ascii to feedparser, no content type iso-8859-1.
    charset = _CHARSET.search(content_type or "")
    parsed = feedparser.parse(
        io.BytesIO(body),
        response_headers={
            "content-type": "application/xml" + (f"; charset={charset.group(1)}" if charset else ""),
            "content-location": final_url or feed.url,
        },
        sanitize_html=False,  # all markup is stripped by clean_text
        resolve_relative_uris=False,
    )
    entries = parsed.get("entries") or []
    if not parsed.get("version") and not entries:
        raise FeedParseError("not an RSS/Atom document")
    base_url = final_url or feed.url
    host = urlsplit(base_url).netloc.lower()
    link_counts = Counter((entry.get("link") or "").strip() for entry in entries)
    strip, drop = _regex(feed.title_strip), _regex(feed.drop_title_re)
    items: dict[str, NewsItem] = {}
    hints: dict[str, FeedHints] = {}
    for entry in entries[:MAX_ENTRIES]:
        title = clean_text(entry.get("title"), TITLE_MAX_CHARS)
        if strip is not None:
            title = strip.sub("", title).strip()
        if not title or (drop is not None and drop.search(title)):
            continue
        link = (entry.get("link") or "").strip()
        absolute = urljoin(base_url, link) if link else ""
        uid = item_uid(host, dict.get(entry, "id") or "", absolute, title, link_unique=link_counts[link] <= 1)
        if uid in items:
            continue
        summary = clean_text(entry.get("summary"), SUMMARY_MAX_CHARS) or _content_text(entry)
        items[uid] = NewsItem(
            uid=uid,
            title_key=title_key(title),
            source=feed.id,
            url=absolute if _is_http(absolute) else "",
            title=title,
            fetched_ts=fetched_ts,
            summary="" if summary == title else summary,
            lang=feed.language,
            published_ts=_published_ts(entry, feed.default_tz, fetched_ts),
        )
        found = _symbol_hints(entry, absolute)
        if found.tickers or found.isins:
            hints[uid] = found
    ordered = sorted(items.values(), key=lambda item: item.published_ts or fetched_ts)  # stable: ties keep order
    return ordered, hints


@functools.lru_cache(maxsize=64)
def _regex(pattern: str | None) -> re.Pattern[str] | None:
    return re.compile(pattern) if pattern else None


def _is_http(url: str) -> bool:
    parts = urlsplit(url)
    return parts.scheme in ("http", "https") and bool(parts.netloc)


def _content_text(entry: Any) -> str:
    """Plain text of the first content block (Atom <content>, RSS content:encoded) or ''."""
    content = entry.get("content") or []
    if not content or not isinstance(content[0], Mapping):
        return ""
    return clean_text(content[0].get("value"), SUMMARY_MAX_CHARS)


def _published_ts(entry: Any, default_tz: str, fetched_ts: int) -> int | None:
    # dict.get: FeedParserDict['updated'] silently falls back to 'published' (with a DeprecationWarning).
    raw = dict.get(entry, "published") or dict.get(entry, "updated") or ""
    published = parse_published(str(raw), default_tz)
    return None if published is None else min(int(published.timestamp()), fetched_ts)


def _symbol_hints(entry: Any, link: str) -> FeedHints:
    """Tickers / ISINs that the feed itself attaches to an entry (field names verified by the probe)."""
    tickers: list[str] = []
    isins: list[str] = []
    value = str(dict.get(entry, "fn_isin") or "").strip()  # finanznachrichten.de <fn:isin>
    if _ISIN.match(value):
        isins.append(value)
    value = str(dict.get(entry, "nasdaq_tickers") or "")  # nasdaq.com <nasdaq:tickers>A,B,C
    tickers += [part.strip().upper() for part in value.split(",") if part.strip()]
    for key in ("sa_symbol", "ndaq_issuesymbol"):  # Seeking Alpha <sa:symbol>, Nasdaq Trader halts
        value = str(dict.get(entry, key) or "").strip()
        if value:
            tickers.append(value.upper())
    for tag in entry.get("tags") or []:
        term, scheme = str(tag.get("term") or "").strip(), str(tag.get("scheme") or "")
        if "seekingalpha.com/symbol/" in scheme:  # <category domain=".../symbol/DAL">dal</category>
            tickers.append(scheme.rsplit("/", 1)[-1].upper())
        elif scheme.endswith("/rss/stock") and ":" in term:  # GlobeNewswire 'Nasdaq:XYZ'
            tickers.append(term.split(":", 1)[1].strip().upper())
        elif scheme.endswith("/rss/ISIN") and _ISIN.match(term):
            isins.append(term)
    found = _STOCKTITAN_LINK.search(link)
    if found:
        tickers.append(found.group(1))
    return FeedHints(tuple(dict.fromkeys(t for t in tickers if t))[:40], tuple(dict.fromkeys(isins))[:10])


# ------------------------------------------------------------------------------------------------
# Identity: uid and title_key
# ------------------------------------------------------------------------------------------------
_TRACKING_PREFIXES: Final = ("utm_", "at_", "pk_", "mtm_", "wt_", "ns_", "feed_")
_TRACKING_PARAMS: Final = frozenset(
    {
        ".tsrc", "oc", "mod", "ftcamp", "fbclid", "gclid", "cmpid", "cmp", "ito", "xtor", "ref", "source", "src",
        "rss", "from", "xy", "partner", "ncid", "smid", "taid", "sref", "time", "syndication", "guccounter",
        "soc_src", "soc_trk", "tlnk", "campaign_id", "emc", "rsskey",
    }
)


def canonical_url(url: str) -> str:
    """https, lower-case host without 'www.', no fragment, no trailing slash, tracking parameters removed.

    Text that is not an absolute URL is returned stripped but otherwise unchanged.
    """
    url = (url or "").strip()
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if not parts.scheme or not parts.netloc:
        return url
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in _TRACKING_PARAMS and not key.lower().startswith(_TRACKING_PREFIXES)
    ]
    host = parts.netloc.lower().removeprefix("www.")
    scheme = "https" if parts.scheme in ("http", "https") else parts.scheme
    return urlunsplit((scheme, host, parts.path.rstrip("/") or "/", urlencode(query), ""))


def item_uid(host: str, guid: str, link: str, title: str, *, link_unique: bool = True) -> str:
    """Stable dedupe key (sha1 hex) of a feed entry.

    Priority: the canonical article link if it is unique inside the document (the same article keeps its URL
    across several feeds of one site, while guids are sometimes feed-specific); else the guid, namespaced by
    the host unless it is a URL (CNBC, wallstreet-online and FinancialJuice use bare numbers; Seeking Alpha
    ticker feeds and Eurostat reuse one link for all items); else link + title; else host + title.
    """
    guid = guid.strip()
    link = canonical_url(link)
    if link and link_unique:
        basis = "u|" + link
    elif guid:
        basis = "g||" + canonical_url(guid) if guid.startswith("http") else f"g|{host}|{guid}"
    elif link:
        basis = f"ut|{link}|{_normalised_title(title)}"
    else:
        basis = f"t|{host}|{_normalised_title(title)}"
    return hashlib.sha1(basis.encode("utf-8"), usedforsecurity=False).hexdigest()


# A trailing ' - Source' / ' | Source' that looks like a publisher name: one to four capitalised words (or
# '&', 'of', 'the', 'de' ...) without digits. 'Ziel 275 Euro' or 'flights grounded' are content, not sources.
# The word quantifier is possessive: a backtracking one could split a long capitalised run ('AbAbAb...') at
# every capital letter, which made a hostile 400-character title cost minutes.
_SOURCE_SUFFIX: Final = re.compile(
    r"\s+[-–—|]\s+(?:(?:[A-Z][\w.'&-]*+|&|of|the|de|des|der|la|le)\s?){1,4}$"
)
_NON_WORD: Final = re.compile(r"[^\w\s]+")
_WHITESPACE: Final = re.compile(r"\s+")


def _normalised_title(title: str) -> str:
    """NFKC, case-folded, punctuation removed, whitespace collapsed."""
    text = unicodedata.normalize("NFKC", title or "").casefold()
    return _WHITESPACE.sub(" ", _NON_WORD.sub(" ", text)).strip()


def title_key(title: str) -> str:
    """sha1 hex of the normalised title: lower case, punctuation and a trailing ' - Source' suffix removed.

    Equal keys mark the same headline from different sources (``repo_news.insert_news`` flags them as
    duplicates); headlines that merely start the same way keep different keys.
    """
    title = unicodedata.normalize("NFKC", title or "").strip()
    stripped = _SOURCE_SUFFIX.sub("", title)
    if any(ch.isdigit() for ch in title[len(stripped) :]) or not stripped:
        stripped = title
    return hashlib.sha1(_normalised_title(stripped).encode("utf-8"), usedforsecurity=False).hexdigest()


# ------------------------------------------------------------------------------------------------
# Text cleaning
# ------------------------------------------------------------------------------------------------
_BLOCK_TAGS: Final = ("script", "style", "table", "figure", "iframe")  # removed together with their content
_BLOCK_OPEN: Final = re.compile(r"<(" + "|".join(_BLOCK_TAGS) + r")\b", re.I)
_BLOCK_CLOSE: Final[Mapping[str, re.Pattern[str]]] = MappingProxyType(
    {name: re.compile(rf"</{name}\s*>", re.I) for name in _BLOCK_TAGS}
)
# Raw text beyond this is cut before cleaning: the result keeps at most a few hundred characters, and every
# step below must stay cheap on hostile or full-article input (content:encoded can be hundreds of kB).
RAW_TEXT_MAX_CHARS: Final = 64_000
_HTML_BREAK: Final = re.compile(r"(?i)<\s*(?:br\s*/?|/p|/div|/li|/h[1-6]|/tr)\s*>")
_HTML_TAG: Final = re.compile(r"<[^<>]{0,2000}>")
_ENTITY: Final = re.compile(r"&(?:[a-z]{2,8}|#\d{2,6}|#x[0-9a-f]{2,5});", re.I)
_CONTROL: Final = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​]")
_TEASER_TAIL: Final = re.compile(
    r"(?is)\s*(?:the post .{0,200}? (?:appeared first on|first appeared on) .*|read more\W*|continue reading\W*|"
    r"weiterlesen\W*|mehr lesen\W*|\[(?:…|\.\.\.|&#8230;)\]\W*|click here to read .*)$"
)
_SPACE_BEFORE_PUNCT: Final = re.compile(r"\s+([,.;:!?])")


def clean_text(text: str | None, max_len: int = SUMMARY_MAX_CHARS) -> str:
    """HTML fragment -> single-line plain text of at most ``max_len`` characters.

    Removes script/style/table/figure/iframe blocks and tags, unescapes entities twice when a feed
    double-escapes its HTML, normalises to NFC, collapses whitespace, strips teaser tails ('Read more',
    'Weiterlesen', 'The post ... appeared first on') and truncates at a word boundary with '…'. Only the first
    ``RAW_TEXT_MAX_CHARS`` characters of the input are read (a tag cut in half at that point is dropped), so
    the cost is bounded and linear for any input.
    """
    if not text:
        return ""
    value = str(text)
    if len(value) > RAW_TEXT_MAX_CHARS:
        value = value[:RAW_TEXT_MAX_CHARS]
        cut_tag = value.rfind("<")
        if cut_tag > value.rfind(">"):
            value = value[:cut_tag]
    if "<" in value:
        value = _HTML_TAG.sub(" ", _HTML_BREAK.sub(" ", _strip_blocks(value)))
    if "&" in value:
        value = html.unescape(value)
        if "&" in value and _ENTITY.search(value):
            value = html.unescape(value)  # double-escaped feeds (&amp;nbsp; / &amp;#8217;)
        if "<" in value and ">" in value:
            value = _HTML_TAG.sub(" ", value)  # tags that were entity-escaped inside the description
    value = _CONTROL.sub("", unicodedata.normalize("NFC", value).replace(" ", " "))
    value = _TEASER_TAIL.sub("", _WHITESPACE.sub(" ", value).strip()).strip()
    value = _SPACE_BEFORE_PUNCT.sub(r"\1", value)
    if max_len and len(value) > max_len:
        limit = max_len - 1  # room for the ellipsis
        cut = value.rfind(" ", 0, limit)
        value = value[: cut if cut > limit * 0.6 else limit].rstrip(" ,;:-") + "…"
    return value


def _strip_blocks(value: str) -> str:
    """``value`` without script/style/table/figure/iframe blocks (opening tag up to the first matching closing
    tag), in linear time.

    A block without a closing tag stays (its tags are removed later, its text is kept). Once one opener of a
    tag name found no closing tag, no later opener of that name can find one, so it is not searched again -
    a lazy regex over the whole text would rescan the rest of the text for every unclosed opener.
    """
    parts: list[str] = []
    pos = 0
    unclosed: set[str] = set()
    for opening in _BLOCK_OPEN.finditer(value):
        name = opening.group(1).lower()
        if opening.start() < pos or name in unclosed:
            continue
        closing = _BLOCK_CLOSE[name].search(value, opening.end())
        if closing is None:
            unclosed.add(name)
            continue
        parts += (value[pos : opening.start()], " ")
        pos = closing.end()
    if not parts:
        return value
    parts.append(value[pos:])
    return "".join(parts)


# ------------------------------------------------------------------------------------------------
# Dates
# ------------------------------------------------------------------------------------------------
# Abbreviations map to IANA zones applied to the wall-clock time, deliberately not to fixed offsets: EIA writes
# 'EST' all year, a fixed -05:00 would be one hour off in summer. feedparser treats unknown names as UTC.
_ZONE_ABBREVIATIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "UTC": "UTC", "UT": "UTC", "GMT": "UTC", "Z": "UTC",
        "EST": "America/New_York", "EDT": "America/New_York", "ET": "America/New_York",
        "CST": "America/Chicago", "CDT": "America/Chicago", "CT": "America/Chicago",
        "MST": "America/Denver", "MDT": "America/Denver", "MT": "America/Denver",
        "PST": "America/Los_Angeles", "PDT": "America/Los_Angeles", "PT": "America/Los_Angeles",
        "CET": "Europe/Berlin", "CEST": "Europe/Berlin", "MEZ": "Europe/Berlin", "MESZ": "Europe/Berlin",
        "MET": "Europe/Berlin", "MEST": "Europe/Berlin",
        "BST": "Europe/London", "WET": "Europe/Lisbon", "WEST": "Europe/Lisbon",
        "EET": "Europe/Helsinki", "EEST": "Europe/Helsinki", "MSK": "Europe/Moscow",
        "IST": "Asia/Kolkata", "JST": "Asia/Tokyo", "KST": "Asia/Seoul", "HKT": "Asia/Hong_Kong",
        "SGT": "Asia/Singapore", "AEST": "Australia/Sydney", "AEDT": "Australia/Sydney",
    }
)
_MONTHS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "jan": 1, "feb": 2, "mar": 3, "mär": 3, "mrz": 3, "apr": 4, "may": 5, "mai": 5, "jun": 6, "jul": 7,
        "aug": 8, "sep": 9, "oct": 10, "okt": 10, "nov": 11, "dec": 12, "dez": 12,
    }
)
_DATETIME_ATTRIBUTE: Final = re.compile(r'datetime\s*=\s*["\']([^"\']+)["\']', re.I)  # ifo: <time datetime=..>
_ISO_DATE: Final = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})(?:[T\s]+(\d{1,2}):(\d{2})(?::(\d{2})(?:[.,]\d+)?)?)?")
_RFC_DATE: Final = re.compile(
    r"(?:[A-Za-zäÄ]{2,9}\.?,?\s+)?(\d{1,2})\.?\s+([A-Za-zäÄ]{3})[A-Za-zäÄ]*\.?,?\s+(\d{4}|\d{2})(?![\d:])"
    r"(?:,?\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?"
)
_US_DATE: Final = re.compile(
    r"(?:[A-Za-z]{3,9},?\s+)?([A-Za-z]{3})[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})"
    r"(?:,?\s+(?:at\s+)?(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([AaPp])\.?[Mm]\.?)?"
)
_UTC_OFFSET: Final = re.compile(r"^(?:GMT|UTC)?\s*([+-])(\d{1,2}):?(\d{2})?$", re.I)
_DATE_MAX_CHARS: Final = 500


@functools.lru_cache(maxsize=64)
def _zone(name: str) -> ZoneInfo:
    return ZoneInfo(name)


def parse_published(raw: str | None, default_tz: str = "UTC") -> datetime | None:
    """Parse a feed date string to an aware UTC datetime; None if missing or unreadable.

    Understands ISO 8601 (with or without zone, fractions, missing seconds), RFC 822 (two-digit years,
    German month names, numeric offsets with or without colon, zone abbreviations), the US long form
    ('November 20, 2026 4:15 PM ET') and a ``datetime='...'`` attribute inside the value. A string without
    zone is read in ``default_tz``. Years outside 1995-2100 are rejected. Only the first
    ``_DATE_MAX_CHARS`` characters are read: a real date value (even wrapped in a ``<time>`` element) is short.
    """
    if not raw or not raw.strip():
        return None
    text = raw.strip()[:_DATE_MAX_CHARS]
    attribute = _DATETIME_ATTRIBUTE.search(text)
    if attribute:
        text = attribute.group(1).strip()
    text = _WHITESPACE.sub(" ", _HTML_TAG.sub(" ", text)).strip()
    fields = _date_fields(text)
    if fields is None:
        return _feedparser_date(text)
    (year, month, day, hour, minute, second), rest = fields
    if not 1995 <= year <= 2100:
        return None
    try:
        naive = datetime(year, month, day, hour, minute, min(second, 59))
    except ValueError:
        return None
    rest = rest.strip(" ()")
    if rest in ("Z", "z"):
        return naive.replace(tzinfo=UTC)
    offset = _UTC_OFFSET.match(rest) if rest else None
    if offset:
        sign = -1 if offset.group(1) == "-" else 1
        delta = timedelta(hours=int(offset.group(2)), minutes=int(offset.group(3) or 0)) * sign
        return (naive - delta).replace(tzinfo=UTC)
    zone = _ZONE_ABBREVIATIONS.get(rest.upper(), default_tz)
    try:
        return naive.replace(tzinfo=_zone(zone)).astimezone(UTC)
    except (ValueError, KeyError):  # unknown default zone
        return None


def _date_fields(text: str) -> tuple[tuple[int, int, int, int, int, int], str] | None:
    """(year, month, day, hour, minute, second) and the text after the date, or None if no date was found."""
    found = _ISO_DATE.search(text)
    if found:
        year, month, day = int(found.group(1)), int(found.group(2)), int(found.group(3))
        clock = (int(found.group(4)), int(found.group(5)), int(found.group(6) or 0)) if found.group(4) else (0, 0, 0)
        return (year, month, day, *clock), text[found.end() :]
    found = _RFC_DATE.search(text)
    if found and found.group(2).lower() in _MONTHS:
        year = int(found.group(3))
        if year < 100:
            year += 2000 if year < 70 else 1900
        clock = (int(found.group(4)), int(found.group(5)), int(found.group(6) or 0)) if found.group(4) else (0, 0, 0)
        return (year, _MONTHS[found.group(2).lower()], int(found.group(1)), *clock), text[found.end() :]
    found = _US_DATE.search(text)
    if found and found.group(1).lower() in _MONTHS:
        hour, minute, second = 0, 0, 0
        if found.group(4):
            hour, minute, second = int(found.group(4)), int(found.group(5)), int(found.group(6) or 0)
            if found.group(7):
                hour = hour % 12 + (12 if found.group(7).lower() == "p" else 0)
        month = _MONTHS[found.group(1).lower()]
        return (int(found.group(3)), month, int(found.group(2)), hour, minute, second), text[found.end() :]
    return None


def _feedparser_date(text: str) -> datetime | None:
    """Last resort: feedparser's own date handlers (asctime and a few national formats)."""
    handler = getattr(getattr(feedparser, "datetimes", None), "_parse_date", None)
    if handler is None:
        return None
    try:
        parsed = handler(text)
    except Exception:  # noqa: BLE001 - third-party parser on untrusted text
        return None
    return datetime.fromtimestamp(calendar.timegm(parsed), UTC) if parsed else None
