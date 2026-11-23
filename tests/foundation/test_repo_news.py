import sqlite3
import threading
from typing import Any

import pytest

from trading_local.db import repo_news
from trading_local.db.database import Database
from trading_local.db.repo_news import FeedState, SymbolNews
from trading_local.models import NewsAnalysis, NewsItem

T0 = 1_800_000_000
HOUR = 3600
DAY = 86_400
WINDOW = 2 * DAY  # default duplicate window of insert_news


def make_item(uid: str, *, title_key: str | None = None, fetched_ts: int = T0, **fields: Any) -> NewsItem:
    return NewsItem(
        uid=uid,
        title_key=title_key or f"key-{uid}",
        source="feed-a",
        url=f"https://news.example/{uid}",
        title=f"Headline {uid}",
        fetched_ts=fetched_ts,
        **fields,
    )


def make_analysis(
    news_id: int | None, *, analyzed_ts: int = T0 + 60, symbols: dict[str, float] | None = None, **fields: Any
) -> NewsAnalysis:
    assert news_id is not None
    values: dict[str, Any] = {
        "model": "test-model",
        "relevance": 3,
        "sentiment": 0.25,
        "category": "earnings",
        "horizon": "days",
        "summary_de": "Zusammenfassung",
    }
    values.update(fields)
    return NewsAnalysis(news_id=news_id, analyzed_ts=analyzed_ts, symbols=dict(symbols or {}), **values)


def uids(pairs: list[tuple[NewsItem, NewsAnalysis | None]]) -> list[str]:
    return [item.uid for item, _ in pairs]


def stored_status(db: Database, news_id: int) -> tuple[Any, ...]:
    [row] = db.query("SELECT status, status_reason FROM news WHERE id = ?", (news_id,))
    return tuple(row)


# ------------------------------------------------------------------------------------------------
# insert_news
# ------------------------------------------------------------------------------------------------
def test_insert_returns_the_given_objects_with_ids_in_input_order(db: Database) -> None:
    first, second = make_item("a"), make_item("b")

    stored = repo_news.insert_news(db, iter([first, second]))  # any iterable is accepted

    assert stored[0] is first and stored[1] is second
    assert [item.id for item in stored] == [1, 2]
    assert all(item.status == "new" and item.duplicate_of is None for item in stored)
    assert repo_news.insert_news(db, []) == []


def test_item_round_trip_keeps_every_field(db: Database) -> None:
    item = make_item(
        "full",
        summary="Umsatz über den Erwartungen",
        lang="de",
        published_ts=T0 - 900,
        hinted_symbols=["SAP.DE", "^GDAXI"],
        status="skipped",
        status_reason="kein Bezug zur Watchlist",
    )

    repo_news.insert_news(db, [item])

    assert repo_news.get_news(db, 1) == (item, None)
    assert repo_news.get_news(db, 2) is None


def test_known_uids_are_skipped_silently(db: Database) -> None:
    repo_news.insert_news(db, [make_item("a")])

    stored = repo_news.insert_news(
        db, [make_item("a", fetched_ts=T0 + 60), make_item("b"), make_item("b"), make_item("c")]
    )

    assert [item.uid for item in stored] == ["b", "c"]
    assert db.scalar("SELECT COUNT(*) FROM news") == 3
    assert db.scalar("SELECT fetched_ts FROM news WHERE uid = 'a'") == T0  # the first sighting stays


def test_duplicate_inside_one_batch(db: Database) -> None:
    original, copy, other = make_item("a", title_key="same"), make_item("b", title_key="same"), make_item("c")

    stored = repo_news.insert_news(db, [original, copy, other])

    assert [(item.status, item.duplicate_of) for item in stored] == [
        ("new", None),
        ("duplicate", original.id),
        ("new", None),
    ]
    [row] = db.query("SELECT status, duplicate_of FROM news WHERE uid = 'b'")
    assert tuple(row) == ("duplicate", original.id)


@pytest.mark.parametrize(
    ("age_s", "expected"),
    [(0, "duplicate"), (HOUR, "duplicate"), (WINDOW, "duplicate"), (WINDOW + 1, "new")],
)
def test_duplicate_across_batches_depends_on_the_window(db: Database, age_s: int, expected: str) -> None:
    [original] = repo_news.insert_news(db, [make_item("a", title_key="same")])

    [later] = repo_news.insert_news(db, [make_item("b", title_key="same", fetched_ts=T0 + age_s)])

    assert later.status == expected
    assert later.duplicate_of == (original.id if expected == "duplicate" else None)
    assert repo_news.get_news(db, 2) == (later, None)


def test_duplicate_window_is_configurable(db: Database) -> None:
    repo_news.insert_news(db, [make_item("a", title_key="same")])

    [inside] = repo_news.insert_news(
        db, [make_item("b", title_key="same", fetched_ts=T0 + 600)], duplicate_window_s=600
    )
    [outside] = repo_news.insert_news(
        db, [make_item("c", title_key="same", fetched_ts=T0 + 1300)], duplicate_window_s=600
    )

    assert (inside.status, outside.status) == ("duplicate", "new")


def test_other_titles_are_never_duplicates(db: Database) -> None:
    stored = repo_news.insert_news(db, [make_item("a", title_key="one"), make_item("b", title_key="two")])

    assert [item.status for item in stored] == ["new", "new"]


def test_duplicate_of_names_the_original_and_never_another_duplicate(db: Database) -> None:
    [original] = repo_news.insert_news(db, [make_item("a", title_key="same")])
    [second] = repo_news.insert_news(db, [make_item("b", title_key="same", fetched_ts=T0 + DAY)])

    # 2.5 days after the original only the duplicate of day 1 is inside the window: the chain continues
    [third] = repo_news.insert_news(db, [make_item("c", title_key="same", fetched_ts=T0 + 2 * DAY + 12 * HOUR)])

    assert (second.status, second.duplicate_of) == ("duplicate", original.id)
    assert (third.status, third.duplicate_of) == ("duplicate", original.id)


def test_headline_returning_after_a_quiet_window_starts_a_new_original(db: Database) -> None:
    [first] = repo_news.insert_news(db, [make_item("a", title_key="same")])
    [second] = repo_news.insert_news(db, [make_item("b", title_key="same", fetched_ts=T0 + WINDOW + 1)])
    [third] = repo_news.insert_news(db, [make_item("c", title_key="same", fetched_ts=T0 + WINDOW + 2)])
    # both originals are inside this item's window: the oldest one wins
    [between] = repo_news.insert_news(db, [make_item("d", title_key="same", fetched_ts=T0 + WINDOW)])

    assert (second.status, second.duplicate_of) == ("new", None)
    assert (third.status, third.duplicate_of) == ("duplicate", second.id)
    assert (between.status, between.duplicate_of) == ("duplicate", first.id)


def test_item_arriving_out_of_order_is_still_a_duplicate(db: Database) -> None:
    [stored_first] = repo_news.insert_news(db, [make_item("fast-feed", title_key="same", fetched_ts=T0 + 5)])

    [fetched_earlier] = repo_news.insert_news(db, [make_item("slow-feed", title_key="same", fetched_ts=T0)])
    [long_before] = repo_news.insert_news(db, [make_item("archive", title_key="same", fetched_ts=T0 - WINDOW - 1)])

    assert (fetched_earlier.status, fetched_earlier.duplicate_of) == ("duplicate", stored_first.id)
    assert (long_before.status, long_before.duplicate_of) == ("new", None)  # the window is limited on both sides


def test_caller_status_is_kept_unless_the_item_is_a_duplicate(db: Database) -> None:
    skipped = make_item("a", title_key="same", status="skipped", status_reason="kein Bezug")
    duplicate = make_item("b", title_key="same", status="skipped", status_reason="kein Bezug")

    repo_news.insert_news(db, [skipped, duplicate])

    assert (skipped.status, duplicate.status) == ("skipped", "duplicate")
    assert db.scalar("SELECT status FROM news WHERE uid = 'a'") == "skipped"


def test_failed_batch_stores_nothing_and_leaves_the_objects_untouched(db: Database) -> None:
    good, copy = make_item("a", title_key="same"), make_item("b", title_key="same")
    bad = make_item("c", status="bogus")  # violates the CHECK constraint of the status column

    with pytest.raises(sqlite3.IntegrityError):
        repo_news.insert_news(db, [good, copy, bad])

    assert db.scalar("SELECT COUNT(*) FROM news") == 0
    assert (good.id, copy.id, copy.status, copy.duplicate_of) == (None, None, "new", None)


def test_insert_news_joins_the_callers_transaction(db: Database) -> None:
    with pytest.raises(RuntimeError), db.write():
        repo_news.insert_news(db, [make_item("a"), make_item("b", title_key="key-a")])
        repo_news.save_feed_state(db, FeedState(feed_id="feed-a", items_total=2))
        raise RuntimeError("the poll failed after storing the items")

    assert db.scalar("SELECT COUNT(*) FROM news") == 0
    assert repo_news.list_feed_states(db) == {}


def test_concurrent_inserts_store_every_uid_once(db: Database) -> None:
    errors: list[BaseException] = []
    stored_counts: list[int] = []

    def worker() -> None:
        try:
            for batch in range(10):
                items = [make_item(f"uid-{batch}-{n}", fetched_ts=T0 + batch) for n in range(5)]
                stored_counts.append(len(repo_news.insert_news(db, items)))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert db.scalar("SELECT COUNT(*) FROM news") == 50
    assert sum(stored_counts) == 50


# ------------------------------------------------------------------------------------------------
# Status and analysis
# ------------------------------------------------------------------------------------------------
def test_set_news_status(db: Database) -> None:
    repo_news.insert_news(db, [make_item("a"), make_item("b")])

    repo_news.set_news_status(db, 1, "error", "LLM nicht erreichbar")
    assert stored_status(db, 1) == ("error", "LLM nicht erreichbar")

    repo_news.set_news_status(db, 1, "queued")
    assert stored_status(db, 1) == ("queued", "")
    assert stored_status(db, 2) == ("new", "")


def test_save_analysis_stores_analysis_symbols_and_status(db: Database) -> None:
    [item] = repo_news.insert_news(db, [make_item("a", status="error", status_reason="Zeitüberschreitung")])
    analysis = make_analysis(item.id, symbols={"SAP.DE": 0.8, "NVDA": -0.3}, confidence=0.9, latency_ms=1234)

    repo_news.save_analysis(db, analysis)

    loaded = repo_news.get_news(db, 1)
    assert loaded is not None
    assert (loaded[0].status, loaded[0].status_reason) == ("analyzed", "")
    assert loaded[1] == analysis
    rows = db.query("SELECT symbol, analyzed_ts, relevance, sentiment FROM news_symbols ORDER BY symbol")
    assert [tuple(row) for row in rows] == [("NVDA", T0 + 60, 3, -0.3), ("SAP.DE", T0 + 60, 3, 0.8)]


def test_save_analysis_replaces_an_earlier_analysis(db: Database) -> None:
    [item] = repo_news.insert_news(db, [make_item("a")])
    repo_news.save_analysis(db, make_analysis(item.id, relevance=2, symbols={"SAP.DE": 0.5, "NVDA": 0.1}))
    second = make_analysis(item.id, analyzed_ts=T0 + 600, relevance=5, sentiment=-0.6, symbols={"SAP.DE": -0.7})

    repo_news.save_analysis(db, second)

    loaded = repo_news.get_news(db, 1)
    assert loaded is not None and loaded[1] == second
    assert db.scalar("SELECT COUNT(*) FROM news_analysis") == 1
    rows = db.query("SELECT symbol, analyzed_ts, relevance, sentiment FROM news_symbols")
    assert [tuple(row) for row in rows] == [("SAP.DE", T0 + 600, 5, -0.7)]


def test_save_analysis_without_symbols(db: Database) -> None:
    [item] = repo_news.insert_news(db, [make_item("a")])
    analysis = make_analysis(item.id)

    repo_news.save_analysis(db, analysis)

    loaded = repo_news.get_news(db, 1)
    assert loaded is not None and loaded[1] == analysis and loaded[1].symbols == {}


def test_save_analysis_for_an_unknown_item_writes_nothing(db: Database) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        repo_news.save_analysis(db, make_analysis(999, symbols={"SAP.DE": 0.5}))

    assert db.scalar("SELECT COUNT(*) FROM news_analysis") == 0
    assert db.scalar("SELECT COUNT(*) FROM news_symbols") == 0


# ------------------------------------------------------------------------------------------------
# list_news
# ------------------------------------------------------------------------------------------------
@pytest.fixture
def corpus(db: Database) -> None:
    """a: analysed (SAP.DE, relevance 4) - b: analysed (NVDA, relevance 2) and hinted NVDA -
    c: hinted SAP.DE - d: hinted SAP - e: duplicate of a, hinted SAP.DE. Ids ascend from a to e."""
    items = [
        make_item("a", title_key="same", fetched_ts=T0),
        make_item("b", fetched_ts=T0 + 100, hinted_symbols=["NVDA"]),
        make_item("c", fetched_ts=T0 + 200, hinted_symbols=["^GDAXI", "SAP.DE"]),
        make_item("d", fetched_ts=T0 + 300, hinted_symbols=["SAP"]),
        make_item("e", title_key="same", fetched_ts=T0 + 400, hinted_symbols=["SAP.DE"]),
    ]
    repo_news.insert_news(db, items)
    repo_news.save_analysis(db, make_analysis(1, relevance=4, symbols={"SAP.DE": 0.6, "^GDAXI": 0.2}))
    repo_news.save_analysis(db, make_analysis(2, relevance=2, symbols={"NVDA": -0.4}))


@pytest.mark.usefixtures("corpus")
def test_list_news_is_newest_first_and_hides_duplicates_by_default(db: Database) -> None:
    assert uids(repo_news.list_news(db)) == ["d", "c", "b", "a"]
    assert uids(repo_news.list_news(db, include_duplicates=True)) == ["e", "d", "c", "b", "a"]


@pytest.mark.usefixtures("corpus")
def test_list_news_attaches_analyses_with_their_symbols(db: Database) -> None:
    by_uid = {item.uid: analysis for item, analysis in repo_news.list_news(db)}

    assert by_uid["c"] is None and by_uid["d"] is None
    assert by_uid["a"] is not None and by_uid["a"].symbols == {"SAP.DE": 0.6, "^GDAXI": 0.2}
    assert by_uid["b"] is not None and by_uid["b"].symbols == {"NVDA": -0.4}
    assert (by_uid["a"].news_id, by_uid["a"].relevance, by_uid["a"].model) == (1, 4, "test-model")


@pytest.mark.usefixtures("corpus")
def test_list_news_paginates_with_before_id(db: Database) -> None:
    first_page = repo_news.list_news(db, limit=3, include_duplicates=True)
    second_page = repo_news.list_news(db, limit=3, include_duplicates=True, before_id=first_page[-1][0].id)

    assert uids(first_page) == ["e", "d", "c"]
    assert uids(second_page) == ["b", "a"]
    assert repo_news.list_news(db, before_id=1) == []


@pytest.mark.usefixtures("corpus")
def test_list_news_filters(db: Database) -> None:
    assert uids(repo_news.list_news(db, analyzed_only=True)) == ["b", "a"]
    assert uids(repo_news.list_news(db, min_relevance=2)) == ["b", "a"]
    assert uids(repo_news.list_news(db, min_relevance=3)) == ["a"]
    assert uids(repo_news.list_news(db, since_ts=T0 + 200)) == ["d", "c"]
    assert uids(repo_news.list_news(db, since_ts=T0 + 100, analyzed_only=True, limit=5)) == ["b"]


@pytest.mark.usefixtures("corpus")
def test_list_news_symbol_filter_matches_analysis_or_hint(db: Database) -> None:
    assert uids(repo_news.list_news(db, symbol="SAP.DE")) == ["c", "a"]  # hint / analysis
    assert uids(repo_news.list_news(db, symbol="SAP.DE", include_duplicates=True)) == ["e", "c", "a"]
    assert uids(repo_news.list_news(db, symbol="SAP")) == ["d"]  # no partial match with 'SAP.DE'
    assert uids(repo_news.list_news(db, symbol="NVDA")) == ["b"]  # hint and analysis: listed once
    assert uids(repo_news.list_news(db, symbol="^GDAXI")) == ["c", "a"]
    assert uids(repo_news.list_news(db, symbol="SAP.DE", min_relevance=1)) == ["a"]
    assert repo_news.list_news(db, symbol="TSLA") == []


def test_list_news_runs_one_statement_whatever_the_page_size(db: Database) -> None:
    items = [make_item(f"n{n}", fetched_ts=T0 + n) for n in range(40)]
    repo_news.insert_news(db, items)
    for item in items:
        repo_news.save_analysis(db, make_analysis(item.id, symbols={"SAP.DE": 0.5, "NVDA": -0.5}))
    statements: list[str] = []
    db.conn().set_trace_callback(statements.append)
    try:
        page = repo_news.list_news(db, limit=40)
        single = repo_news.get_news(db, 7)
    finally:
        db.conn().set_trace_callback(None)

    assert len(page) == 40
    assert all(analysis is not None and len(analysis.symbols) == 2 for _, analysis in page)
    assert single is not None and single[1] is not None and len(single[1].symbols) == 2
    assert len(statements) == 2  # one per call, not one per item


# ------------------------------------------------------------------------------------------------
# pending_news, symbol_news, counts, retention
# ------------------------------------------------------------------------------------------------
def test_pending_news_returns_recent_unanalysed_items_newest_first(db: Database) -> None:
    repo_news.insert_news(
        db,
        [
            make_item("too-old", fetched_ts=T0 - HOUR - 1),
            make_item("just-in-time", fetched_ts=T0 - HOUR, status="queued"),
            make_item("new", fetched_ts=T0 - 60),
            make_item("queued", fetched_ts=T0 - 30, status="queued"),
            make_item("analyzed", fetched_ts=T0 - 20),
            make_item("skipped", fetched_ts=T0 - 10, status="skipped"),
            make_item("failed", fetched_ts=T0 - 5, status="error"),
            make_item("duplicate", title_key="key-new", fetched_ts=T0 - 1),
        ],
    )
    repo_news.save_analysis(db, make_analysis(5))

    pending = repo_news.pending_news(db, now_ts=T0, max_age_s=HOUR)

    assert [item.uid for item in pending] == ["queued", "new", "just-in-time"]
    assert pending[0].status == "queued" and pending[0].id == 4
    assert [item.uid for item in repo_news.pending_news(db, now_ts=T0, max_age_s=HOUR, limit=1)] == ["queued"]


def test_symbol_news_only_sees_analyses_known_at_the_asof_time(db: Database) -> None:
    # All three items were published and fetched long before; what counts is when the analysis was finished.
    early, mid, late = repo_news.insert_news(
        db, [make_item(uid, fetched_ts=T0 - DAY, published_ts=T0 - 2 * DAY) for uid in ("early", "mid", "late")]
    )
    repo_news.save_analysis(db, make_analysis(early.id, analyzed_ts=T0 + 100, symbols={"SAP.DE": 0.5}))
    repo_news.save_analysis(db, make_analysis(mid.id, analyzed_ts=T0 + 200, symbols={"SAP.DE": -0.2, "NVDA": 0.9}))
    repo_news.save_analysis(db, make_analysis(late.id, analyzed_ts=T0 + 300, symbols={"SAP.DE": 0.1}))

    def ids(asof_ts: int, window_s: int) -> list[int]:
        return [news.news_id for news in repo_news.symbol_news(db, "SAP.DE", asof_ts=asof_ts, window_s=window_s)]

    assert ids(T0 + 200, HOUR) == [2, 1]  # newest first; the analysis finished at T0 + 300 is not known yet
    assert ids(T0 + 199, HOUR) == [1]  # one second before the second analysis became known
    assert ids(T0 + 99, HOUR) == []
    assert ids(T0 + 300, HOUR) == [3, 2, 1]
    assert ids(T0 + 200, 100) == [2]  # asof - window < analyzed_ts: the lower bound is exclusive
    assert ids(T0 + 200, 101) == [2, 1]
    assert ids(T0 + 5 * HOUR, HOUR) == []  # everything has left the window


def test_symbol_news_ignores_the_claimed_publication_time(db: Database) -> None:
    # A feed with a wrong clock claims a publication time in the future: the item is known all the same.
    [item] = repo_news.insert_news(db, [make_item("a", published_ts=T0 + DAY)])
    repo_news.save_analysis(db, make_analysis(item.id, analyzed_ts=T0 + 100, symbols={"SAP.DE": 0.5}))

    assert len(repo_news.symbol_news(db, "SAP.DE", asof_ts=T0 + 100, window_s=HOUR)) == 1


def test_symbol_news_returns_the_symbol_view_of_an_item(db: Database) -> None:
    first, second = repo_news.insert_news(db, [make_item("a"), make_item("b")])
    repo_news.save_analysis(
        db,
        make_analysis(
            first.id,
            relevance=2,
            sentiment=0.1,
            symbols={"SAP.DE": -0.2, "NVDA": 0.9},
            category="analyst",
            horizon="weeks",
            summary_de="Analyst senkt das Kursziel",
        ),
    )
    repo_news.save_analysis(db, make_analysis(second.id, relevance=5, symbols={"SAP.DE": 0.7}))

    everything = repo_news.symbol_news(db, "SAP.DE", asof_ts=T0 + 60, window_s=HOUR)
    relevant = repo_news.symbol_news(db, "SAP.DE", asof_ts=T0 + 60, window_s=HOUR, min_relevance=3)

    assert [news.news_id for news in everything] == [2, 1]  # same analysis time: the newer item first
    assert everything[1] == SymbolNews(
        news_id=1,
        analyzed_ts=T0 + 60,
        relevance=2,
        sentiment=-0.2,  # the symbol's sentiment, not the item's overall 0.1
        title="Headline a",
        summary_de="Analyst senkt das Kursziel",
        horizon="weeks",
        category="analyst",
        fetched_ts=T0,  # first sight: the age reference of the news context (CAUS-3)
    )
    assert [news.news_id for news in relevant] == [2]
    assert [news.sentiment for news in repo_news.symbol_news(db, "NVDA", asof_ts=T0 + 60, window_s=HOUR)] == [0.9]
    assert repo_news.symbol_news(db, "TSLA", asof_ts=T0 + 60, window_s=HOUR) == []


def test_news_counts(db: Database) -> None:
    repo_news.insert_news(
        db,
        [
            make_item("old-analysed", fetched_ts=T0 - 3 * HOUR),
            make_item("old-waiting", fetched_ts=T0 - 2 * HOUR),
            make_item("queued", fetched_ts=T0 - HOUR, status="queued"),
            make_item("analysed", fetched_ts=T0 - 10),
            make_item("duplicate", title_key="key-analysed", fetched_ts=T0 - 5),
            make_item("skipped", fetched_ts=T0 - 1, status="skipped"),
        ],
    )
    repo_news.save_analysis(db, make_analysis(1, analyzed_ts=T0 - HOUR - 1))
    repo_news.save_analysis(db, make_analysis(4, analyzed_ts=T0))

    assert repo_news.news_counts(db, since_ts=T0 - HOUR) == {"items": 4, "analyzed": 1, "backlog": 2}
    assert repo_news.news_counts(db, since_ts=T0 + 1) == {"items": 0, "analyzed": 0, "backlog": 2}


def test_delete_news_before_removes_old_items_with_their_analyses(db: Database) -> None:
    repo_news.insert_news(
        db,
        [
            make_item("old-original", title_key="same", fetched_ts=T0),
            make_item("old-duplicate", title_key="same", fetched_ts=T0 + 10),
            make_item("young-duplicate", title_key="same", fetched_ts=T0 + DAY),
            make_item("young", fetched_ts=T0 + DAY + 1),
            make_item("on-the-cutoff", fetched_ts=T0 + HOUR),
        ],
    )
    repo_news.save_analysis(db, make_analysis(1, symbols={"SAP.DE": 0.5}))
    repo_news.save_analysis(db, make_analysis(4, symbols={"NVDA": 0.5}))

    deleted = repo_news.delete_news_before(db, T0 + HOUR)

    assert deleted == 2
    remaining = {item.uid: item for item, _ in repo_news.list_news(db, include_duplicates=True)}
    assert set(remaining) == {"young-duplicate", "young", "on-the-cutoff"}
    assert (remaining["young-duplicate"].status, remaining["young-duplicate"].duplicate_of) == ("duplicate", None)
    assert [row["news_id"] for row in db.query("SELECT news_id FROM news_analysis")] == [4]
    assert [row["symbol"] for row in db.query("SELECT symbol FROM news_symbols")] == ["NVDA"]
    assert repo_news.delete_news_before(db, T0 + HOUR) == 0


# ------------------------------------------------------------------------------------------------
# Feed state
# ------------------------------------------------------------------------------------------------
def test_feed_state_round_trip_and_replace(db: Database) -> None:
    assert repo_news.get_feed_state(db, "feed-a") is None
    state = FeedState(
        feed_id="feed-a",
        etag='W/"abc"',
        last_modified="Fri, 20 Nov 2026 15:00:00 GMT",
        last_ok_ts=T0,
        next_poll_ts=T0 + 300,
        items_total=12,
    )

    repo_news.save_feed_state(db, state)
    assert repo_news.get_feed_state(db, "feed-a") == state

    failing = FeedState(feed_id="feed-a", last_error="HTTP 503", fail_count=3, next_poll_ts=T0 + 900, items_total=12)
    repo_news.save_feed_state(db, failing)
    assert repo_news.get_feed_state(db, "feed-a") == failing  # etag, last_modified and last_ok_ts are cleared too
    assert db.scalar("SELECT COUNT(*) FROM feed_state") == 1


def test_list_feed_states(db: Database) -> None:
    assert repo_news.list_feed_states(db) == {}
    fresh, polled = FeedState(feed_id="feed-b"), FeedState(feed_id="feed-a", last_ok_ts=T0, items_total=3)

    repo_news.save_feed_state(db, fresh)
    repo_news.save_feed_state(db, polled)

    assert repo_news.list_feed_states(db) == {"feed-a": polled, "feed-b": fresh}


# ------------------------------------------------------------------------------------------------
# Review additions: boundaries, hostile input, concurrency, query cost
# ------------------------------------------------------------------------------------------------
def query_plan(db: Database, run: Any) -> list[str]:
    """The query plan of the single statement that ``run`` sends to the database."""
    statements: list[str] = []
    db.conn().set_trace_callback(statements.append)
    try:
        run()
    finally:
        db.conn().set_trace_callback(None)
    [statement] = statements
    return [row["detail"] for row in db.query("EXPLAIN QUERY PLAN " + statement)]


def test_duplicate_window_of_zero_only_matches_the_same_second(db: Database) -> None:
    repo_news.insert_news(db, [make_item("a", title_key="same")])

    [same_second] = repo_news.insert_news(db, [make_item("b", title_key="same")], duplicate_window_s=0)
    [next_second] = repo_news.insert_news(
        db, [make_item("c", title_key="same", fetched_ts=T0 + 1)], duplicate_window_s=0
    )

    assert (same_second.status, same_second.duplicate_of) == ("duplicate", 1)
    assert (next_second.status, next_second.duplicate_of) == ("new", None)


def test_concurrent_inserts_of_one_headline_keep_a_single_original(db: Database) -> None:
    # Six feeds deliver the same headline at the same moment: exactly one of them may become the original.
    feeds = 6
    barrier = threading.Barrier(feeds)
    stored: list[NewsItem] = []
    errors: list[BaseException] = []

    def worker(feed: int) -> None:
        try:
            barrier.wait(timeout=10)
            stored.extend(repo_news.insert_news(db, [make_item(f"feed-{feed}", title_key="same")]))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(feed,)) for feed in range(feeds)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors
    originals = [item for item in stored if item.status == "new"]
    duplicates = [item for item in stored if item.status == "duplicate"]
    assert len(originals) == 1 and len(duplicates) == feeds - 1
    assert {item.duplicate_of for item in duplicates} == {originals[0].id}
    assert db.scalar("SELECT COUNT(*) FROM news WHERE status = 'new'") == 1


def test_items_stored_by_one_thread_are_read_by_another(db: Database) -> None:
    # every thread has its own connection: what one thread committed must be visible to the others
    worker = threading.Thread(
        target=lambda: repo_news.save_analysis(
            db, make_analysis(repo_news.insert_news(db, [make_item("a")])[0].id, symbols={"SAP.DE": 0.5})
        )
    )
    worker.start()
    worker.join(timeout=30)

    loaded = repo_news.get_news(db, 1)

    assert loaded is not None and loaded[0].status == "analyzed"
    assert loaded[1] is not None and loaded[1].symbols == {"SAP.DE": 0.5}
    assert [news.news_id for news in repo_news.symbol_news(db, "SAP.DE", asof_ts=T0 + 60, window_s=HOUR)] == [1]


def test_list_news_limit_counts_items_and_not_impact_rows(db: Database) -> None:
    items = [make_item(f"n{n}", fetched_ts=T0 + n) for n in range(5)]
    repo_news.insert_news(db, items)
    for item in items:
        repo_news.save_analysis(db, make_analysis(item.id, symbols={"SAP.DE": 0.5, "NVDA": -0.5, "^GDAXI": 0.1}))

    first = repo_news.list_news(db, limit=2)
    second = repo_news.list_news(db, limit=2, before_id=first[-1][0].id)
    filtered = repo_news.list_news(db, limit=2, symbol="NVDA", min_relevance=3)

    assert (uids(first), uids(second), uids(filtered)) == (["n4", "n3"], ["n2", "n1"], ["n4", "n3"])
    for _, analysis in first + second + filtered:
        assert analysis is not None and analysis.symbols == {"SAP.DE": 0.5, "NVDA": -0.5, "^GDAXI": 0.1}
    assert repo_news.list_news(db, limit=0) == []


@pytest.mark.parametrize("symbol", ['A"B', "x' OR '1'='1", "100%", "A_B", "SAP.DE'; DROP TABLE news; --", "Ä\\B"])
def test_symbol_filters_take_the_symbol_literally(db: Database, symbol: str) -> None:
    repo_news.insert_news(
        db,
        [
            make_item("hinted", hinted_symbols=["FIRST", symbol]),
            make_item("analysed"),
            make_item("other", hinted_symbols=["AxB", "1000", "x", "SAP.DE"]),  # what a wildcard would match
        ],
    )
    repo_news.save_analysis(db, make_analysis(2, symbols={symbol: 0.5}))
    repo_news.save_analysis(db, make_analysis(3, symbols={"AxB": 0.1, "1000": 0.1}))

    assert uids(repo_news.list_news(db, symbol=symbol)) == ["analysed", "hinted"]
    assert [news.news_id for news in repo_news.symbol_news(db, symbol, asof_ts=T0 + 60, window_s=HOUR)] == [2]
    assert db.scalar("SELECT COUNT(*) FROM news") == 3  # and nothing was executed as SQL


@pytest.mark.parametrize("symbol", ["%", "_", "", '"', "SAP", "AP.D", '","'])
def test_symbol_filter_matches_whole_symbols_only(db: Database, symbol: str) -> None:
    repo_news.insert_news(db, [make_item("hinted", hinted_symbols=["SAP.DE", "NVDA"]), make_item("analysed")])
    repo_news.save_analysis(db, make_analysis(2, symbols={"SAP.DE": 0.5}))

    assert repo_news.list_news(db, symbol=symbol) == []
    assert repo_news.symbol_news(db, symbol, asof_ts=T0 + 60, window_s=HOUR) == []


def test_list_news_since_ts_is_exact_for_items_stored_out_of_order(db: Database) -> None:
    # the ids ascend in storage order, the fetch times do not (item n has id n + 1)
    fetched = [T0 + 50, T0 + 10, T0 + 40, T0, T0 + 30, T0 + 20]
    repo_news.insert_news(db, [make_item(f"n{n}", fetched_ts=ts) for n, ts in enumerate(fetched)])

    for since_ts in (T0 - 1, T0, T0 + 1, T0 + 30, T0 + 50, T0 + 51):
        expected = [n for n in reversed(range(len(fetched))) if fetched[n] >= since_ts]  # newest id first
        assert uids(repo_news.list_news(db, since_ts=since_ts)) == [f"n{n}" for n in expected], since_ts
        assert uids(repo_news.list_news(db, since_ts=since_ts, limit=2)) == [f"n{n}" for n in expected[:2]]
        assert uids(repo_news.list_news(db, since_ts=since_ts, before_id=4)) == [f"n{n}" for n in expected if n < 3]


def test_list_news_since_ts_reads_only_the_recent_part_of_the_table(db: Database) -> None:
    plan = query_plan(db, lambda: repo_news.list_news(db, since_ts=T0))

    assert "SCAN n" not in plan  # no walk over all stored items ...
    assert "SEARCH n USING INTEGER PRIMARY KEY (rowid>?)" in plan  # ... but one with a lower end,
    assert any("COVERING INDEX news_fetched_idx" in step for step in plan)  # which the fetch-time index provides


def vm_steps(db: Database, run: Any) -> int:
    """Work that ``run`` causes on this thread's connection, in units of ten SQLite virtual-machine steps."""
    ticks = 0

    def tick() -> int:
        nonlocal ticks
        ticks += 1
        return 0

    db.conn().set_progress_handler(tick, 10)
    try:
        run()
    finally:
        db.conn().set_progress_handler(None, 0)
    return ticks


def test_list_news_since_ts_costs_what_is_new_and_not_what_is_stored(db: Database) -> None:
    stored = 3000
    db.executemany(
        "INSERT INTO news (uid, title_key, source, url, title, fetched_ts) VALUES (?, ?, 'feed', 'https://x', 't', ?)",
        ((f"n{n}", f"k{n}", T0 + n) for n in range(1, stored + 1)),
    )

    whole_table = vm_steps(db, lambda: repo_news.list_news(db, symbol="TSLA"))  # no match: the walk reads everything
    nothing_new = vm_steps(db, lambda: repo_news.list_news(db, since_ts=T0 + stored + 1))
    five_new = vm_steps(db, lambda: repo_news.list_news(db, since_ts=T0 + stored - 4))
    five_new_filtered = vm_steps(db, lambda: repo_news.list_news(db, since_ts=T0 + stored - 4, symbol="TSLA"))

    assert len(repo_news.list_news(db, since_ts=T0 + stored - 4)) == 5
    assert max(nothing_new, five_new, five_new_filtered) * 20 < whole_table


def test_list_news_since_ts_is_exact_beyond_the_probed_range(db: Database) -> None:
    # More matches than the lower-end probe looks at: the walk must stay unbounded, down to the oldest match.
    total = repo_news._SINCE_PROBE + 50
    db.executemany(
        "INSERT INTO news (uid, title_key, source, url, title, fetched_ts) VALUES (?, ?, 'feed', 'https://x', 't', ?)",
        ((f"n{n}", f"k{n}", T0 + n) for n in range(1, total + 1)),  # item n has id n
    )

    def ids(**filters: Any) -> list[int | None]:
        return [item.id for item, _ in repo_news.list_news(db, **filters)]

    assert ids(since_ts=T0, limit=3) == [total, total - 1, total - 2]
    assert ids(since_ts=T0, before_id=4) == [3, 2, 1]  # the oldest matches are not cut off
    assert ids(since_ts=T0 + 2, before_id=4) == [3, 2]
    # as many matches as the probe covers, and just below and above that
    for matches in (repo_news._SINCE_PROBE + step for step in (-1, 0, 1, 2)):
        oldest = total - matches + 1
        assert ids(since_ts=T0 + oldest, before_id=oldest + 2) == [oldest + 1, oldest], matches
        assert ids(since_ts=T0 + oldest, limit=1) == [total]
    # a page that is larger than the standard probe is complete as well
    assert ids(since_ts=T0 + 3, limit=total + 5) == list(range(total, 2, -1))


@pytest.mark.parametrize(
    "run",
    [
        lambda db: repo_news.symbol_news(db, "SAP.DE", asof_ts=T0, window_s=HOUR),
        lambda db: repo_news.pending_news(db, now_ts=T0, max_age_s=HOUR),
        lambda db: repo_news.news_counts(db, since_ts=T0),
        lambda db: repo_news.get_news(db, 1),
    ],
    ids=["symbol_news", "pending_news", "news_counts", "get_news"],
)
def test_frequent_readers_never_scan_a_whole_table(db: Database, run: Any) -> None:
    plan = query_plan(db, lambda: run(db))

    assert not [step for step in plan if step.startswith("SCAN") and step not in ("SCAN page", "SCAN CONSTANT ROW")]


def test_symbol_news_relevance_threshold_is_inclusive(db: Database) -> None:
    low, high = repo_news.insert_news(db, [make_item("low"), make_item("high")])
    repo_news.save_analysis(db, make_analysis(low.id, relevance=2, symbols={"SAP.DE": 0.1}))
    repo_news.save_analysis(db, make_analysis(high.id, relevance=4, symbols={"SAP.DE": 0.9}))

    def ids(min_relevance: int) -> list[int]:
        found = repo_news.symbol_news(db, "SAP.DE", asof_ts=T0 + 60, window_s=HOUR, min_relevance=min_relevance)
        return [news.news_id for news in found]

    assert (ids(2), ids(3), ids(4), ids(5)) == ([2, 1], [2], [2], [])
    assert ids(1) == [2, 1]  # the default threshold lets everything through


def test_symbol_news_shows_a_reanalysis_only_from_the_time_it_was_made(db: Database) -> None:
    [item] = repo_news.insert_news(db, [make_item("a")])
    repo_news.save_analysis(
        db, make_analysis(item.id, analyzed_ts=T0 + 100, relevance=2, symbols={"SAP.DE": 0.5, "NVDA": 0.3})
    )
    repo_news.save_analysis(db, make_analysis(item.id, analyzed_ts=T0 + 600, relevance=5, symbols={"SAP.DE": -0.9}))

    def view(symbol: str, asof_ts: int) -> list[tuple[int, int, float]]:
        found = repo_news.symbol_news(db, symbol, asof_ts=asof_ts, window_s=HOUR)
        return [(news.analyzed_ts, news.relevance, news.sentiment) for news in found]

    # No look-ahead: the second verdict must not leak into the time before it existed (the first one is gone).
    assert view("SAP.DE", T0 + 599) == []
    assert view("SAP.DE", T0 + 600) == [(T0 + 600, 5, -0.9)]
    assert view("NVDA", T0 + 600) == []  # no longer named by the analysis


def test_symbol_news_window_bounds_do_not_depend_on_the_fetch_time(db: Database) -> None:
    # fetched late but analysed inside the window, and fetched inside the window but analysed after the as-of time
    late_fetch, late_analysis = repo_news.insert_news(
        db, [make_item("late-fetch", fetched_ts=T0 - 10 * DAY), make_item("late-analysis", fetched_ts=T0 - 60)]
    )
    repo_news.save_analysis(db, make_analysis(late_fetch.id, analyzed_ts=T0, symbols={"SAP.DE": 0.4}))
    repo_news.save_analysis(db, make_analysis(late_analysis.id, analyzed_ts=T0 + 1, symbols={"SAP.DE": 0.6}))

    found = repo_news.symbol_news(db, "SAP.DE", asof_ts=T0, window_s=60)

    assert [news.news_id for news in found] == [late_fetch.id]


def test_delete_news_before_keeps_the_references_between_surviving_items(db: Database) -> None:
    repo_news.insert_news(
        db,
        [
            make_item("old-original", title_key="old", fetched_ts=T0),
            make_item("young-duplicate-of-old", title_key="old", fetched_ts=T0 + DAY),
            make_item("young-original", title_key="young", fetched_ts=T0 + DAY),
            make_item("young-duplicate-of-young", title_key="young", fetched_ts=T0 + DAY + 5),
        ],
    )

    assert repo_news.delete_news_before(db, T0 + HOUR) == 1  # dependent rows are not counted

    rows = db.query("SELECT uid, status, duplicate_of FROM news ORDER BY id")
    assert [tuple(row) for row in rows] == [
        ("young-duplicate-of-old", "duplicate", None),
        ("young-original", "new", None),
        ("young-duplicate-of-young", "duplicate", 3),
    ]
    assert db.query("PRAGMA foreign_key_check") == []


def test_delete_news_before_joins_the_callers_transaction(db: Database) -> None:
    repo_news.insert_news(
        db,
        [make_item("old", title_key="same"), make_item("young", title_key="same", fetched_ts=T0 + DAY)],
    )
    repo_news.save_analysis(db, make_analysis(1, symbols={"SAP.DE": 0.5}))

    with pytest.raises(RuntimeError), db.write():
        assert repo_news.delete_news_before(db, T0 + HOUR) == 1
        assert db.scalar("SELECT duplicate_of FROM news WHERE uid = 'young'") is None
        raise RuntimeError("the housekeeping job failed after the purge")

    assert db.scalar("SELECT COUNT(*) FROM news") == 2
    assert db.scalar("SELECT duplicate_of FROM news WHERE uid = 'young'") == 1
    assert db.scalar("SELECT COUNT(*) FROM news_analysis") == 1
    assert db.scalar("SELECT COUNT(*) FROM news_symbols") == 1
