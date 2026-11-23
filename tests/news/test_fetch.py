"""fetch_feed against httpx.MockTransport: conditional GET, body hash, backoff, recovery, size cap, bad documents."""

import asyncio
from collections.abc import AsyncIterator, Callable

import httpx
import pytest

from trading_local.db.repo_news import FeedState
from trading_local.news import fetcher
from trading_local.news.fetcher import MAX_BODY_BYTES, USER_AGENT, FetchResult, backoff_delay, fetch_feed, make_client

from tests.news.helpers import NOW_TS, fixture_bytes, make_feed

FEED = make_feed("t_fetch", host="feeds.example.test", poll_seconds=120)
BODY = fixture_bytes("investing_news.xml")


async def fetch(
    handler: Callable[[httpx.Request], httpx.Response], state: FeedState | None = None, *, now_ts: int = NOW_TS,
    body_hash: str | None = None,
) -> FetchResult:
    async with make_client(httpx.MockTransport(handler)) as client:
        return await fetch_feed(client, FEED, state or FeedState(FEED.id), now_ts, body_hash=body_hash)


async def test_success_parses_items_and_remembers_validators() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        headers = {"ETag": '"v1"', "Last-Modified": "Sun, 22 Nov 2026 18:34:30 GMT"}
        return httpx.Response(200, content=BODY, headers=headers)

    result = await fetch(handler)
    assert len(result.items) == 5 and result.error == "" and not result.not_modified
    assert result.status == 200
    assert result.body_hash is not None and len(result.body_hash) == 40
    state = result.state
    assert (state.etag, state.last_modified) == ('"v1"', "Sun, 22 Nov 2026 18:34:30 GMT")
    assert (state.last_ok_ts, state.fail_count, state.last_error) == (NOW_TS, 0, "")
    assert state.next_poll_ts == NOW_TS + FEED.poll_seconds
    request = seen[0]
    assert request.headers["User-Agent"] == USER_AGENT
    assert "If-None-Match" not in request.headers and "If-Modified-Since" not in request.headers


async def test_conditional_get_304() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("If-None-Match") == '"v1"' and request.headers.get("If-Modified-Since") == "LM":
            return httpx.Response(304)
        return httpx.Response(200, content=BODY, headers={"ETag": '"v1"', "Last-Modified": "LM"})

    first = await fetch(handler)
    second = await fetch(handler, first.state, now_ts=NOW_TS + 120, body_hash=first.body_hash)
    assert second.not_modified and second.items == [] and second.status == 304
    assert (second.state.etag, second.state.last_modified) == ('"v1"', "LM")  # kept although the 304 had none
    assert second.state.last_ok_ts == NOW_TS + 120
    assert second.state.next_poll_ts == NOW_TS + 240
    assert second.body_hash == first.body_hash


async def test_identical_body_without_validators_is_not_parsed_again(monkeypatch: pytest.MonkeyPatch) -> None:
    first = await fetch(lambda request: httpx.Response(200, content=BODY))
    monkeypatch.setattr(fetcher, "parse_feed", lambda *args, **kwargs: pytest.fail("must not parse"))
    second = await fetch(lambda request: httpx.Response(200, content=BODY), first.state, body_hash=first.body_hash)
    assert second.not_modified and second.items == [] and second.error == ""


async def test_changed_body_is_parsed() -> None:
    first = await fetch(lambda request: httpx.Response(200, content=BODY))
    changed = BODY.replace(b"</channel>", b"<!-- regenerated --></channel>")
    second = await fetch(lambda request: httpx.Response(200, content=changed), first.state, body_hash=first.body_hash)
    assert not second.not_modified and len(second.items) == 5 and second.body_hash != first.body_hash


async def test_backoff_after_errors_and_recovery() -> None:
    state = FeedState(FEED.id, etag='"old"', last_ok_ts=NOW_TS - 600)
    first = await fetch(lambda request: httpx.Response(500), state)
    assert first.error == "HTTP 500" and first.status == 500 and first.items == []
    assert first.state.fail_count == 1
    assert first.state.next_poll_ts == NOW_TS + 120  # max(60 s, poll interval)
    assert first.state.etag == '"old"' and first.state.last_ok_ts == NOW_TS - 600  # untouched
    second = await fetch(lambda request: httpx.Response(503), first.state, now_ts=NOW_TS + 120)
    assert second.state.fail_count == 2
    assert second.state.next_poll_ts == NOW_TS + 120 + 240  # doubled
    recovered = await fetch(lambda request: httpx.Response(200, content=BODY), second.state, now_ts=NOW_TS + 400)
    assert recovered.error == "" and len(recovered.items) == 5
    assert (recovered.state.fail_count, recovered.state.last_error) == (0, "")
    assert recovered.state.next_poll_ts == NOW_TS + 400 + 120


@pytest.mark.parametrize(
    ("status", "headers", "delay"),
    [
        (404, {}, 6 * 3600),
        (410, {}, 6 * 3600),
        (403, {}, 30 * 60),
        (429, {}, 5 * 60),
        (429, {"Retry-After": "900"}, 900),
        (429, {"Retry-After": "Sun, 22 Nov 2026 20:00:00 GMT"}, 3600),
        (502, {}, 120),
    ],
)
async def test_backoff_depends_on_the_failure(status: int, headers: dict[str, str], delay: int) -> None:
    result = await fetch(lambda request: httpx.Response(status, headers=headers))
    assert result.error == f"HTTP {status}"
    assert result.state.next_poll_ts - NOW_TS == delay


def test_backoff_delay_grows_and_is_capped() -> None:
    assert [backoff_delay(None, n, 60) for n in (1, 2, 3, 4, 5, 6)] == [60, 120, 240, 480, 960, 1800]
    assert backoff_delay(None, 20, 60) == 1800
    assert backoff_delay(404, 3, 60) == 24 * 3600
    assert backoff_delay(403, 10, 60) == 6 * 3600
    assert backoff_delay(429, 1, 60, retry_after_s=10 * 3600) == 6 * 3600  # Retry-After honoured up to 6 h
    assert backoff_delay(None, 1, 900) == 900


async def test_timeouts_and_connection_errors_mark_the_host_unreachable() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    def reset(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("peer closed", request=request)

    for handler, error, unreachable in (
        (timeout, "Zeitüberschreitung", True),
        (refused, "Verbindungsfehler (ConnectError)", True),
        (reset, "Übertragungsfehler (RemoteProtocolError)", False),
    ):
        result = await fetch(handler)
        assert (result.error, result.unreachable, result.state.fail_count) == (error, unreachable, 1)
        assert result.state.next_poll_ts == NOW_TS + 120


async def test_total_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetcher, "REQUEST_DEADLINE_S", 0.05)

    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.2)
        return httpx.Response(200, content=BODY)

    async with make_client(httpx.MockTransport(slow)) as client:
        result = await fetch_feed(client, FEED, FeedState(FEED.id), NOW_TS)
    assert result.error == "Zeitüberschreitung" and result.unreachable


async def test_size_cap_with_and_without_content_length() -> None:
    big = b"<rss>" + b" " * MAX_BODY_BYTES + b"</rss>"
    declared = await fetch(lambda request: httpx.Response(200, content=big))
    assert declared.error == "Antwort größer als 2 MB" and declared.state.fail_count == 1

    async def chunks() -> AsyncIterator[bytes]:
        for _ in range(3):
            yield b" " * (MAX_BODY_BYTES // 2 + 1)

    streamed = await fetch(lambda request: httpx.Response(200, content=chunks()))
    assert streamed.error == "Antwort größer als 2 MB"


async def test_html_page_and_bad_xml_are_failures_without_new_validators() -> None:
    state = FeedState(FEED.id, etag='"old"')
    html = await fetch(
        lambda request: httpx.Response(200, content=b"<html><body>Bitte Cookies akzeptieren</body></html>",
                                       headers={"ETag": '"new"', "content-type": "text/html"}),
        state,
    )
    assert html.error == "Kein gültiger RSS/Atom-Feed" and html.state.fail_count == 1
    assert html.state.etag == '"old"' and html.body_hash is None
    garbage = await fetch(lambda request: httpx.Response(200, content=b"\x00\xff\xfe garbage"))
    assert garbage.error == "Kein gültiger RSS/Atom-Feed"


async def test_redirects_are_followed_and_relative_links_resolved() -> None:
    body = (b'<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>'
            b"<item><title>Moved story</title><link>/story/1</link></item></channel></rss>")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/t_fetch.xml":
            return httpx.Response(301, headers={"Location": "https://mirror.example.test/rss/feed.xml"})
        return httpx.Response(200, content=body)

    result = await fetch(handler)
    assert [item.url for item in result.items] == ["https://mirror.example.test/story/1"]


async def test_too_many_redirects_is_a_failure() -> None:
    result = await fetch(lambda request: httpx.Response(302, headers={"Location": str(request.url)}))
    assert result.error == "Übertragungsfehler (TooManyRedirects)" and result.state.fail_count == 1


async def test_unexpected_errors_never_escape(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("parser bug")

    monkeypatch.setattr(fetcher, "parse_feed", explode)
    result = await fetch(lambda request: httpx.Response(200, content=BODY))
    assert result.error == "Unerwarteter Fehler beim Lesen (RuntimeError)" and result.state.fail_count == 1
