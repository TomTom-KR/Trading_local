"""News repository: feed items, LLM analyses, per-symbol impact rows and per-feed polling state.

Time semantics (no look-ahead)
* ``fetched_ts``    when this platform first saw an item: nothing can have known it earlier.
* ``analyzed_ts``   when the LLM analysis became known. Decision code reads news only through
                    ``symbol_news``, which filters on this column.
* ``published_ts``  what the feed claims. It may be missing or wrong and is never used as a filter here.

All functions are safe to call from several threads; every multi-statement change is one transaction.
"""

import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from trading_local.db.database import Database, dumps, loads
from trading_local.models import NewsAnalysis, NewsItem, NewsStatus

_NEWS_COLUMNS = (
    "id",
    "uid",
    "title_key",
    "source",
    "url",
    "title",
    "summary",
    "lang",
    "published_ts",
    "fetched_ts",
    "hinted_symbols",
    "status",
    "status_reason",
    "duplicate_of",
)
_ANALYSIS_COLUMNS = (
    "analyzed_ts",
    "model",
    "relevance",
    "sentiment",
    "confidence",
    "category",
    "horizon",
    "summary_de",
    "latency_ms",
)
_FEED_STATE_COLUMNS = (
    "feed_id",
    "etag",
    "last_modified",
    "last_ok_ts",
    "last_error",
    "fail_count",
    "next_poll_ts",
    "items_total",
)

_NEWS_FIELDS = ", ".join(f"n.{column}" for column in _NEWS_COLUMNS)
# Analysis columns are aliased with an 'a_' prefix so they cannot collide with the news columns.
_ANALYSIS_FIELDS = ", ".join(f"a.{column} AS a_{column}" for column in _ANALYSIS_COLUMNS)
_SELECT_FEED_STATE = f"SELECT {', '.join(_FEED_STATE_COLUMNS)} FROM feed_state"

_INSERT_NEWS = """
INSERT INTO news (uid, title_key, source, url, title, summary, lang, published_ts, fetched_ts,
                  hinted_symbols, status, status_reason, duplicate_of)
VALUES (:uid, :title_key, :source, :url, :title, :summary, :lang, :published_ts, :fetched_ts,
        :hinted_symbols, :status, :status_reason, :duplicate_of)
"""

# The oldest stored item with the same title inside the window decides. COALESCE resolves it to its own
# original, so ``duplicate_of`` names the first sighting of the headline and not another duplicate (unless
# retention has deleted that first sighting: then the oldest surviving item of the chain stands in for it).
_FIND_ORIGINAL = """
SELECT COALESCE(duplicate_of, id)
FROM news
WHERE title_key = ? AND fetched_ts BETWEEN ? AND ?
ORDER BY fetched_ts, id
LIMIT 1
"""

_UPSERT_ANALYSIS = """
INSERT INTO news_analysis (news_id, analyzed_ts, model, relevance, sentiment, confidence, category, horizon,
                           summary_de, latency_ms)
VALUES (:news_id, :analyzed_ts, :model, :relevance, :sentiment, :confidence, :category, :horizon,
        :summary_de, :latency_ms)
ON CONFLICT (news_id) DO UPDATE SET
    analyzed_ts = excluded.analyzed_ts,
    model = excluded.model,
    relevance = excluded.relevance,
    sentiment = excluded.sentiment,
    confidence = excluded.confidence,
    category = excluded.category,
    horizon = excluded.horizon,
    summary_de = excluded.summary_de,
    latency_ms = excluded.latency_ms
"""

# One statement per page (no N+1, one consistent snapshot): the CTE picks the page of news ids, the outer
# query adds the analysis and one row per impact symbol. ``{conditions}`` only ever receives SQL fragments
# written in this module; all values are bound parameters.
_NEWS_PAGE = f"""
WITH page AS (
    SELECT n.id
    FROM news n
    LEFT JOIN news_analysis a ON a.news_id = n.id
    WHERE {{conditions}}
    ORDER BY n.id DESC
    LIMIT ?
)
SELECT {_NEWS_FIELDS}, {_ANALYSIS_FIELDS}, s.symbol AS impact_symbol, s.sentiment AS impact_sentiment
FROM page
JOIN news n ON n.id = page.id
LEFT JOIN news_analysis a ON a.news_id = n.id
LEFT JOIN news_symbols s ON s.news_id = a.news_id
ORDER BY n.id DESC, s.symbol
"""

# Lower end of the id walk for a ``since_ts`` filter (parameters: probe, since_ts, probe); it only repeats
# what ``fetched_ts >= ?`` says and changes no result. The inner query takes up to ``probe`` matching entries
# straight from news_fetched_idx. Fewer than ``probe`` means that these are all there are: their smallest id
# is the exact end of the walk, so "nothing new since ..." returns at once instead of reading every stored
# item. Otherwise there is no bound (0 is below every id) and the walk is what it would be without this term.
_SINCE_FLOOR = """(
    SELECT CASE WHEN COUNT(*) < ? THEN MIN(id) ELSE 0 END
    FROM (SELECT id FROM news WHERE fetched_ts >= ? LIMIT ?)
)"""
# Smallest probe. The probe is always one entry larger than the page, so a page that the matches cannot fill
# always gets its bound, and at least this large: that also bounds ranges of up to this many matches when a
# second, rarely matching filter is what keeps the page from filling.
_SINCE_PROBE = 10_000

_SYMBOL_NEWS = """
SELECT s.news_id, s.analyzed_ts, s.relevance, s.sentiment, n.title, a.summary_de, a.horizon, a.category,
       n.fetched_ts
FROM news_symbols s
JOIN news n ON n.id = s.news_id
JOIN news_analysis a ON a.news_id = s.news_id
WHERE s.symbol = ? AND s.analyzed_ts > ? AND s.analyzed_ts <= ? AND s.relevance >= ?
ORDER BY s.analyzed_ts DESC, s.news_id DESC
"""

_NEWS_COUNTS = """
SELECT (SELECT COUNT(*) FROM news WHERE fetched_ts >= :since_ts) AS items,
       (SELECT COUNT(*) FROM news_analysis WHERE analyzed_ts >= :since_ts) AS analyzed,
       (SELECT COUNT(*) FROM news WHERE status IN ('new', 'queued')) AS backlog
"""

_UPSERT_FEED_STATE = """
INSERT INTO feed_state (feed_id, etag, last_modified, last_ok_ts, last_error, fail_count, next_poll_ts, items_total)
VALUES (:feed_id, :etag, :last_modified, :last_ok_ts, :last_error, :fail_count, :next_poll_ts, :items_total)
ON CONFLICT (feed_id) DO UPDATE SET
    etag = excluded.etag,
    last_modified = excluded.last_modified,
    last_ok_ts = excluded.last_ok_ts,
    last_error = excluded.last_error,
    fail_count = excluded.fail_count,
    next_poll_ts = excluded.next_poll_ts,
    items_total = excluded.items_total
"""


@dataclass(frozen=True, slots=True)
class SymbolNews:
    """One analysed news item as it affects one symbol (see ``symbol_news``)."""

    news_id: int
    analyzed_ts: int
    relevance: int  # 1..5
    sentiment: float  # -1..1 for this symbol (not the item's overall sentiment)
    title: str
    summary_de: str
    horizon: str
    category: str
    fetched_ts: int | None = None  # when the platform first saw the item (age reference; never after analyzed_ts)


@dataclass(slots=True)
class FeedState:
    """Polling state of one feed (table ``feed_state``)."""

    feed_id: str
    etag: str | None = None
    last_modified: str | None = None
    last_ok_ts: int | None = None
    last_error: str = ""
    fail_count: int = 0
    next_poll_ts: int = 0
    items_total: int = 0


# ------------------------------------------------------------------------------------------------
# Items
# ------------------------------------------------------------------------------------------------
def insert_news(db: Database, items: Iterable[NewsItem], *, duplicate_window_s: int = 172_800) -> list[NewsItem]:
    """Store the items whose ``uid`` is not known yet and return them in input order.

    Items with a known uid - stored earlier or repeated inside the batch - are skipped silently.

    Duplicate detection: a new item is a duplicate when an item stored before it (in an earlier batch or
    earlier in this batch) has the same ``title_key`` and a ``fetched_ts`` within ``duplicate_window_s``
    of the new item's. The window reaches in both directions, so items that arrive slightly out of
    order are recognised as well. A duplicate is stored with status ``'duplicate'`` and ``duplicate_of``
    set to the original of the oldest such item. Because every match counts, a headline that keeps
    reappearing stays a duplicate until it has been absent for a whole window.

    The returned objects are the given ones, updated with their ``id`` and, for duplicates, with the
    final ``status`` and ``duplicate_of``. Nothing is changed on them if the transaction fails.
    """
    batch = list(items)
    if not batch:
        return []
    stored: list[tuple[NewsItem, int, int | None]] = []
    with db.write():
        for item in batch:
            if db.scalar("SELECT 1 FROM news WHERE uid = ?", (item.uid,)) is not None:
                continue
            original_id: int | None = db.scalar(
                _FIND_ORIGINAL,
                (item.title_key, item.fetched_ts - duplicate_window_s, item.fetched_ts + duplicate_window_s),
            )
            params = _news_params(item)
            if original_id is not None:
                params.update(status="duplicate", duplicate_of=original_id)
            stored.append((item, db.execute(_INSERT_NEWS, params), original_id))
    for item, news_id, original_id in stored:
        item.id = news_id
        if original_id is not None:
            item.status = "duplicate"
            item.duplicate_of = original_id
    return [item for item, _, _ in stored]


def get_news(db: Database, news_id: int) -> tuple[NewsItem, NewsAnalysis | None] | None:
    """One item with its analysis (None when the item has not been analysed)."""
    found = _query_news(db, ["n.id = ?"], [news_id], limit=1)
    return found[0] if found else None


def set_news_status(db: Database, news_id: int, status: NewsStatus, reason: str = "") -> None:
    """Set the processing status of an item; ``reason`` replaces the stored reason."""
    db.execute("UPDATE news SET status = ?, status_reason = ? WHERE id = ?", (status, reason, news_id))


def save_analysis(db: Database, analysis: NewsAnalysis) -> None:
    """Store (or replace) the analysis of a news item and mark the item as analysed.

    One transaction: upsert ``news_analysis``, replace the item's ``news_symbols`` rows by
    ``analysis.symbols`` (each row carries the analysis' relevance and ``analyzed_ts`` for as-of queries)
    and set the item's status to ``'analyzed'``. Raises ``sqlite3.IntegrityError`` if the item does not
    exist; nothing is written in that case.
    """
    params = asdict(analysis)
    impacts = [
        (analysis.news_id, symbol, analysis.analyzed_ts, analysis.relevance, sentiment)
        for symbol, sentiment in analysis.symbols.items()
    ]
    with db.write() as conn:
        conn.execute(_UPSERT_ANALYSIS, params)
        conn.execute("DELETE FROM news_symbols WHERE news_id = ?", (analysis.news_id,))
        conn.executemany(
            "INSERT INTO news_symbols (news_id, symbol, analyzed_ts, relevance, sentiment) VALUES (?, ?, ?, ?, ?)",
            impacts,
        )
        conn.execute("UPDATE news SET status = 'analyzed', status_reason = '' WHERE id = ?", (analysis.news_id,))


def list_news(
    db: Database,
    *,
    limit: int = 50,
    before_id: int | None = None,
    symbol: str | None = None,
    min_relevance: int | None = None,
    analyzed_only: bool = False,
    include_duplicates: bool = False,
    since_ts: int | None = None,
) -> list[tuple[NewsItem, NewsAnalysis | None]]:
    """Items with their analyses, newest first (id descending); page with ``before_id``.

    ``symbol`` matches items whose analysis names the symbol or whose ``hinted_symbols`` contain it.
    ``min_relevance`` implies analysed items. ``since_ts`` refers to ``fetched_ts`` (inclusive).

    Cost: the page is collected by walking the items from the newest one down until ``limit`` of them
    match. That is instant when matches are frequent; a filter that matches fewer than ``limit`` items
    (a rarely mentioned ``symbol``, a high ``min_relevance``) walks the whole table - up to 0.5 ms per
    1000 stored items. A ``since_ts`` of the last few days ends the walk there (see ``_SINCE_FLOOR``).
    """
    conditions: list[str] = []
    params: list[Any] = []
    if before_id is not None:
        conditions.append("n.id < ?")
        params.append(before_id)
    if since_ts is not None:
        probe = max(limit, _SINCE_PROBE) + 1
        conditions.append(f"n.fetched_ts >= ? AND n.id >= {_SINCE_FLOOR}")
        params += [since_ts, probe, since_ts, probe]
    if not include_duplicates:
        conditions.append("n.status != 'duplicate'")
    if analyzed_only:
        conditions.append("a.news_id IS NOT NULL")
    if min_relevance is not None:
        conditions.append("a.relevance >= ?")
        params.append(min_relevance)
    if symbol is not None:
        # hinted_symbols is a compact JSON list of strings, so the quoted symbol can only match a whole element.
        conditions.append(
            "(EXISTS (SELECT 1 FROM news_symbols x WHERE x.news_id = n.id AND x.symbol = ?)"
            " OR instr(n.hinted_symbols, ?) > 0)"
        )
        params += [symbol, dumps(symbol)]
    return _query_news(db, conditions, params, limit=limit)


def pending_news(db: Database, *, now_ts: int, max_age_s: int, limit: int = 200) -> list[NewsItem]:
    """Items still waiting for their analysis (status 'new' or 'queued') that were fetched within
    ``max_age_s``, newest first. Used to re-queue the backlog after a restart."""
    rows = db.query(
        f"SELECT {_NEWS_FIELDS} FROM news n "
        "WHERE n.status IN ('new', 'queued') AND n.fetched_ts >= ? "
        "ORDER BY n.fetched_ts DESC, n.id DESC LIMIT ?",
        (now_ts - max_age_s, limit),
    )
    return [_news_from_row(row) for row in rows]


def skipped_news_for(
    db: Database, symbols: Iterable[str], *, since_ts: int, reasons: Iterable[str], limit: int = 100
) -> list[NewsItem]:
    """Skipped items whose ``hinted_symbols`` contain one of ``symbols``, whose skip reason is one of
    ``reasons`` and that were fetched at or after ``since_ts``, newest first. Used to give items another
    chance when their instrument suddenly moves (news service, ``prioritize``)."""
    wanted = [dumps(symbol) for symbol in dict.fromkeys(symbols)]
    allowed = list(dict.fromkeys(reasons))
    if not wanted or not allowed:
        return []
    # hinted_symbols is a compact JSON list of strings, so a quoted symbol can only match a whole element.
    symbol_match = " OR ".join("instr(n.hinted_symbols, ?) > 0" for _ in wanted)
    rows = db.query(
        f"SELECT {_NEWS_FIELDS} FROM news n "
        f"WHERE n.status = 'skipped' AND n.fetched_ts >= ? AND n.status_reason IN ({', '.join('?' * len(allowed))}) "
        f"AND ({symbol_match}) ORDER BY n.fetched_ts DESC, n.id DESC LIMIT ?",
        (since_ts, *allowed, *wanted, limit),
    )
    return [_news_from_row(row) for row in rows]


def symbol_news(
    db: Database, symbol: str, *, asof_ts: int, window_s: int, min_relevance: int = 1
) -> list[SymbolNews]:
    """Analysed news for ``symbol`` as known at ``asof_ts``, newest first.

    As-of safe accessor for decision code: returns the rows with
    ``asof_ts - window_s < analyzed_ts <= asof_ts``. The filter is the time the analysis became known,
    never the publication time, so an analysis that was finished after ``asof_ts`` is invisible even if
    the item was published or fetched long before.
    """
    rows = db.query(_SYMBOL_NEWS, (symbol, asof_ts - window_s, asof_ts, min_relevance))
    return [SymbolNews(**dict(row)) for row in rows]


def news_counts(db: Database, *, since_ts: int) -> dict[str, int]:
    """Counters for the status page.

    ``items``: items fetched since ``since_ts`` (duplicates included); ``analyzed``: analyses finished
    since ``since_ts``; ``backlog``: items of any age that still wait for their analysis.
    """
    return dict(db.query(_NEWS_COUNTS, {"since_ts": since_ts})[0])  # a SELECT without FROM yields one row


def delete_news_before(db: Database, ts: int) -> int:
    """Delete the items fetched before ``ts`` and return how many items were deleted.

    The schema removes what depends on a deleted item in the same statement: its analysis and impact rows
    are deleted (ON DELETE CASCADE) and younger duplicates lose their ``duplicate_of`` reference
    (ON DELETE SET NULL) but stay stored with status 'duplicate'.

    Cost: for every deleted item SQLite looks up the items whose ``duplicate_of`` names it. Without an
    index on ``news.duplicate_of`` each lookup reads the whole table, which makes a purge quadratic:
    measured 8 ms per deleted item with 40 000 stored items (9 ms for 500 items once the index exists).
    All other writers wait meanwhile, so purge often and in small steps as long as the schema lacks it.
    """
    with db.write() as conn:
        return conn.execute("DELETE FROM news WHERE fetched_ts < ?", (ts,)).rowcount


# ------------------------------------------------------------------------------------------------
# Feed state
# ------------------------------------------------------------------------------------------------
def get_feed_state(db: Database, feed_id: str) -> FeedState | None:
    row = db.query_one(f"{_SELECT_FEED_STATE} WHERE feed_id = ?", (feed_id,))
    return FeedState(**dict(row)) if row is not None else None


def save_feed_state(db: Database, state: FeedState) -> None:
    """Insert or completely replace the state of ``state.feed_id``."""
    db.execute(_UPSERT_FEED_STATE, asdict(state))


def list_feed_states(db: Database) -> dict[str, FeedState]:
    """All stored feed states by feed id."""
    return {row["feed_id"]: FeedState(**dict(row)) for row in db.query(_SELECT_FEED_STATE)}


# ------------------------------------------------------------------------------------------------
# Row mapping
# ------------------------------------------------------------------------------------------------
def _query_news(
    db: Database, conditions: Sequence[str], params: Sequence[Any], *, limit: int
) -> list[tuple[NewsItem, NewsAnalysis | None]]:
    """Run the page query for the AND-ed ``conditions`` and fold its rows into (item, analysis) pairs."""
    sql = _NEWS_PAGE.format(conditions=" AND ".join(conditions) or "TRUE")
    pairs: dict[int, tuple[NewsItem, NewsAnalysis | None]] = {}
    for row in db.query(sql, (*params, limit)):
        pair = pairs.get(row["id"])
        if pair is None:
            pair = pairs[row["id"]] = (_news_from_row(row), _analysis_from_row(row))
        analysis = pair[1]
        if analysis is not None and row["impact_symbol"] is not None:
            analysis.symbols[row["impact_symbol"]] = row["impact_sentiment"]
    return list(pairs.values())


def _news_from_row(row: sqlite3.Row) -> NewsItem:
    values = {column: row[column] for column in _NEWS_COLUMNS}
    values["hinted_symbols"] = loads(values["hinted_symbols"], [])
    return NewsItem(**values)


def _analysis_from_row(row: sqlite3.Row) -> NewsAnalysis | None:
    """The analysis part of a joined row, without impact symbols (None when the item is not analysed)."""
    if row["a_analyzed_ts"] is None:
        return None
    return NewsAnalysis(news_id=row["id"], **{column: row[f"a_{column}"] for column in _ANALYSIS_COLUMNS})


def _news_params(item: NewsItem) -> dict[str, Any]:
    params = asdict(item)
    params["hinted_symbols"] = dumps(params["hinted_symbols"])
    return params
