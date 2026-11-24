"""Crypto-derivatives repository: perpetual snapshots, liquidations and the derived metrics.

Time semantics (no look-ahead): ``received_ts`` is when this platform learned a row. ``metrics`` only uses
rows with ``ts <= asof_ts`` AND ``received_ts <= asof_ts``, so a backfill stored later never leaks into a
decision that was taken earlier.

Write rules: live rows replace what is stored for their instant (the newest answer is authoritative; the
earliest ``received_ts`` is kept); backfilled rows never overwrite a value, they only fill empty fields
(``replace=False``; funding and open-interest history share the full hours).
"""

import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

from trading_local.db.database import Database

FUNDING_WINDOW_S: Final = 30 * 86_400  # funding is ranked against the hourly funding of the last 30 days
MIN_FUNDING_HOURS: Final = 48  # fewer hourly values: no ranking
OI_TOLERANCE_1H_S: Final = 1800  # the reference value of the 1-hour change may be this much older
OI_TOLERANCE_24H_S: Final = 7200
LIQ_REFERENCE_S: Final = 7 * 86_400  # usual liquidations: 5-minute sums of the last 7 days
LIQ_BUCKET_S: Final = 300
STALE_S: Final = 15 * 60  # metrics with older snapshots are marked stale
HOURS_PER_YEAR: Final = 24 * 365


@dataclass(frozen=True, slots=True)
class DerivSnapshot:
    symbol: str
    ts: int
    received_ts: int
    funding_rate: float | None = None
    funding_pred: float | None = None
    open_interest: float | None = None
    mark_price: float | None = None
    index_price: float | None = None


@dataclass(frozen=True, slots=True)
class DerivMetrics:
    """The derivatives picture of one crypto instrument as known at ``asof_ts`` (None: not available)."""

    symbol: str
    asof_ts: int
    data_ts: int  # instant of the newest snapshot used
    stale: bool  # the newest snapshot is older than ``STALE_S``
    funding_rate: float | None  # relative funding per hour (fraction)
    funding_pred: float | None
    funding_annual_pct: float | None  # funding per hour x 24 x 365, in percent
    funding_pred_annual_pct: float | None
    funding_z: float | None  # current funding vs. the hourly funding of the last 30 days
    open_interest: float | None  # contracts (units of the base asset)
    oi_change_1h_pct: float | None
    oi_change_24h_pct: float | None
    mark_price: float | None
    index_price: float | None
    basis_pct: float | None  # mark / index - 1, in percent
    liq_15m: float | None  # liquidated units of the base asset in the last 15 complete minutes
    liq_60m: float | None
    liq_60m_usd: float | None
    liq_5m_ratio: float | None  # last 5 minutes / mean 5-minute sum of the last 7 days


_SNAPSHOT_COLUMNS = ("symbol", "ts", "funding_rate", "funding_pred", "open_interest", "mark_price", "index_price")


def upsert_snapshots(db: Database, rows: Iterable[DerivSnapshot], *, replace: bool = True) -> int:
    """Store snapshots. ``replace=True`` (live) replaces the values of the instant and keeps the earliest
    ``received_ts``; ``replace=False`` (backfill) only fills fields that are still empty - the row then counts
    as learned at the later time, so nothing becomes visible before it was known. Returns rows written."""
    values = [
        (r.symbol, int(r.ts), r.funding_rate, r.funding_pred, r.open_interest, r.mark_price, r.index_price,
         int(r.received_ts))
        for r in rows
    ]  # fmt: skip
    if not values:
        return 0
    if replace:
        conflict = (
            "DO UPDATE SET funding_rate = excluded.funding_rate, funding_pred = excluded.funding_pred, "
            "open_interest = excluded.open_interest, mark_price = excluded.mark_price, "
            "index_price = excluded.index_price, received_ts = MIN(received_ts, excluded.received_ts)"
        )
    else:
        fields = _SNAPSHOT_COLUMNS[2:]
        filled = " OR ".join(f"({name} IS NULL AND excluded.{name} IS NOT NULL)" for name in fields)
        conflict = (
            f"DO UPDATE SET received_ts = CASE WHEN {filled} THEN MAX(received_ts, excluded.received_ts) "
            "ELSE received_ts END, " + ", ".join(f"{name} = COALESCE({name}, excluded.{name})" for name in fields)
        )
    with db.write() as conn:
        before = conn.total_changes
        conn.executemany(
            f"INSERT INTO deriv_snapshots ({', '.join(_SNAPSHOT_COLUMNS)}, received_ts) "
            f"VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (symbol, ts) {conflict}",
            values,
        )
        return conn.total_changes - before


def upsert_liquidations(
    db: Database, symbol: str, points: Iterable[tuple[int, float]], *, received_ts: int, replace: bool = True
) -> int:
    """Store per-minute liquidation volumes (minute start, contracts). Returns rows written."""
    values = [(symbol, int(ts), float(volume), int(received_ts)) for ts, volume in points if volume >= 0]
    if not values:
        return 0
    conflict = (
        "DO UPDATE SET volume = excluded.volume, received_ts = MIN(received_ts, excluded.received_ts)"
        if replace
        else "DO NOTHING"
    )
    with db.write() as conn:
        before = conn.total_changes
        conn.executemany(
            "INSERT INTO deriv_liquidations (symbol, ts, volume, received_ts) VALUES (?, ?, ?, ?) "
            f"ON CONFLICT (symbol, ts) {conflict}",
            values,
        )
        return conn.total_changes - before


def newest_ts(db: Database, table: str, symbol: str, *, column: str | None = None) -> int | None:
    """Newest ``ts`` of ``symbol`` in 'deriv_snapshots' / 'deriv_liquidations' (with ``column`` not NULL)."""
    if table not in ("deriv_snapshots", "deriv_liquidations"):
        raise ValueError(f"unknown table {table!r}")
    if column is not None and column not in _SNAPSHOT_COLUMNS:
        raise ValueError(f"unknown column {column!r}")
    condition = f" AND {column} IS NOT NULL" if column else ""
    return db.scalar(f"SELECT MAX(ts) FROM {table} WHERE symbol = ?{condition}", (symbol,))


def first_liquidation_ts(db: Database, symbol: str, *, asof_ts: int) -> int | None:
    """The oldest stored liquidation minute of ``symbol`` known at ``asof_ts`` (the live poll and the backfill
    store every minute, also those without liquidations, so the history is complete from there on)."""
    return db.scalar(
        "SELECT MIN(ts) FROM deriv_liquidations WHERE symbol = ? AND received_ts <= ?", (symbol, asof_ts)
    )


def liquidation_buckets(db: Database, symbol: str, *, start_ts: int, end_ts: int, asof_ts: int) -> list[float]:
    """5-minute liquidation sums of the aligned buckets in ``[start_ts, end_ts)`` known at ``asof_ts``;
    buckets without any liquidation count as 0."""
    first, last = start_ts // LIQ_BUCKET_S, (end_ts - 1) // LIQ_BUCKET_S
    rows = db.query(
        "SELECT ts / ? AS bucket, SUM(volume) AS total FROM deriv_liquidations "
        "WHERE symbol = ? AND ts >= ? AND ts < ? AND received_ts <= ? GROUP BY bucket",
        (LIQ_BUCKET_S, symbol, first * LIQ_BUCKET_S, (last + 1) * LIQ_BUCKET_S, asof_ts),
    )
    totals = {int(row["bucket"]): float(row["total"]) for row in rows}
    return [totals.get(bucket, 0.0) for bucket in range(first, last + 1)]


def liquidations_between(db: Database, symbol: str, *, start_ts: int, end_ts: int, asof_ts: int) -> float:
    """Liquidated contracts in the minutes starting in ``[start_ts, end_ts)``, known at ``asof_ts``."""
    return float(
        db.scalar(
            "SELECT SUM(volume) FROM deriv_liquidations WHERE symbol = ? AND ts >= ? AND ts < ? AND received_ts <= ?",
            (symbol, start_ts, end_ts, asof_ts),
            default=0.0,
        )
    )


def _value_at(db: Database, symbol: str, column: str, *, at_ts: int, asof_ts: int, tolerance_s: int) -> float | None:
    """The newest value of ``column`` at or before ``at_ts`` (at most ``tolerance_s`` older), known at ``asof_ts``."""
    row = db.query_one(
        f"SELECT {column} FROM deriv_snapshots WHERE symbol = ? AND ts <= ? AND ts >= ? AND received_ts <= ? "
        f"AND {column} IS NOT NULL ORDER BY ts DESC LIMIT 1",
        (symbol, at_ts, at_ts - tolerance_s, asof_ts),
    )
    return None if row is None else float(row[0])


def _change_pct(now: float | None, then: float | None) -> float | None:
    if now is None or then is None or then <= 0:
        return None
    return (now / then - 1.0) * 100.0


def funding_z(current: float | None, history: Sequence[float]) -> float | None:
    """How unusual the current funding is: (current - mean) / standard deviation of ``history``."""
    if current is None or len(history) < MIN_FUNDING_HOURS:
        return None
    spread = statistics.pstdev(history)
    if not spread > 0:
        return None
    return (current - statistics.fmean(history)) / spread


def metrics(db: Database, symbol: str, *, asof_ts: int) -> DerivMetrics | None:
    """The derivatives picture of ``symbol`` as known at ``asof_ts``; None without any snapshot."""
    latest = db.query_one(
        "SELECT ts, funding_rate, funding_pred, open_interest, mark_price, index_price FROM deriv_snapshots "
        "WHERE symbol = ? AND ts <= ? AND received_ts <= ? AND mark_price IS NOT NULL ORDER BY ts DESC LIMIT 1",
        (symbol, asof_ts, asof_ts),
    )
    if latest is None:
        return None
    data_ts = int(latest["ts"])
    rate, pred = latest["funding_rate"], latest["funding_pred"]
    oi, mark, index = latest["open_interest"], latest["mark_price"], latest["index_price"]
    hourly = [
        float(row[0])
        for row in db.query(
            "SELECT funding_rate FROM deriv_snapshots WHERE symbol = ? AND ts % 3600 = 0 AND ts >= ? AND ts < ? "
            "AND received_ts <= ? AND funding_rate IS NOT NULL",
            (symbol, asof_ts - FUNDING_WINDOW_S, asof_ts, asof_ts),
        )
    ]
    minute_end = asof_ts // 60 * 60  # minutes starting before this are complete
    first_minute = first_liquidation_ts(db, symbol, asof_ts=asof_ts)
    has_liquidations = first_minute is not None
    liq_15 = liquidations_between(db, symbol, start_ts=minute_end - 900, end_ts=minute_end, asof_ts=asof_ts)
    liq_60 = liquidations_between(db, symbol, start_ts=minute_end - 3600, end_ts=minute_end, asof_ts=asof_ts)
    liq_5 = liquidations_between(db, symbol, start_ts=minute_end - 300, end_ts=minute_end, asof_ts=asof_ts)
    reference_start = max(first_minute or minute_end, minute_end - 300 - LIQ_REFERENCE_S)
    reference = (
        liquidation_buckets(db, symbol, start_ts=reference_start, end_ts=minute_end - 300, asof_ts=asof_ts)
        if reference_start < minute_end - 300
        else []
    )
    usual = statistics.fmean(reference) if reference else 0.0
    return DerivMetrics(
        symbol=symbol,
        asof_ts=asof_ts,
        data_ts=data_ts,
        stale=asof_ts - data_ts > STALE_S,
        funding_rate=rate,
        funding_pred=pred,
        funding_annual_pct=None if rate is None else rate * HOURS_PER_YEAR * 100.0,
        funding_pred_annual_pct=None if pred is None else pred * HOURS_PER_YEAR * 100.0,
        funding_z=funding_z(rate, hourly),
        open_interest=oi,
        oi_change_1h_pct=_change_pct(
            oi, _value_at(db, symbol, "open_interest", at_ts=data_ts - 3600, asof_ts=asof_ts,
                          tolerance_s=OI_TOLERANCE_1H_S)
        ),
        oi_change_24h_pct=_change_pct(
            oi, _value_at(db, symbol, "open_interest", at_ts=data_ts - 86_400, asof_ts=asof_ts,
                          tolerance_s=OI_TOLERANCE_24H_S)
        ),
        mark_price=mark,
        index_price=index,
        basis_pct=(mark / index - 1.0) * 100.0 if mark and index else None,
        liq_15m=liq_15 if has_liquidations else None,
        liq_60m=liq_60 if has_liquidations else None,
        liq_60m_usd=liq_60 * mark if has_liquidations and mark else None,
        liq_5m_ratio=liq_5 / usual if has_liquidations and usual > 0 else None,
    )  # fmt: skip


def prune(db: Database, *, before_ts: int) -> int:
    """Delete liquidations and minute snapshots older than ``before_ts`` (snapshots at full hours are kept: they
    carry the funding history). Returns the number of rows deleted."""
    with db.write() as conn:
        deleted = conn.execute("DELETE FROM deriv_liquidations WHERE ts < ?", (before_ts,)).rowcount
        deleted += conn.execute("DELETE FROM deriv_snapshots WHERE ts < ? AND ts % 3600 != 0", (before_ts,)).rowcount
    return deleted
