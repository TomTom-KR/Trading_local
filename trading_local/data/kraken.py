"""Kraken public market data (REST) for the pseudo venue KRAKEN - real time, no API key.

Every detail below was checked against docs.kraken.com and the live API on 2026-11-22/23 (probe scripts in
the work notes of the Kraken integration):

* REST base ``https://api.kraken.com/0/public/``. Every answer is ``{"error": [...], "result": {...}}``; an
  error is a string ``'E<category>:<message>'`` (a leading ``'W'`` marks a mere warning). Rate limits answer
  ``EAPI:Rate limit exceeded`` / ``EGeneral:Too many requests`` (or HTTP 429); ``EService:Throttled`` and
  ``EService:Busy`` ask the client to come back later. Public endpoints need no key - none is used here.
  A request that names several pairs is refused AS A WHOLE when one of them is unknown
  (``EQuery:Unknown asset pair``): a pair Kraken no longer lists is detected and left out of the Ticker batch.
* Pair names: our symbols ('BTC-EUR') map to the WebSocket v2 notation ('BTC/EUR'), which the REST endpoints
  accept as well (verified for AssetPairs, Ticker, OHLC and Trades); answers are then keyed by exactly that
  name. Kraken's own REST names differ - bitcoin is XBT, dogecoin XDG: altname 'XBTEUR', canonical result key
  'XXBTZEUR', WS v1 name 'XBT/EUR'. They are kept in ``KrakenPair`` (with the price/lot decimals, tick size
  and minimum order size) for a later live broker; this module itself contains no trading code.
* AssetPairs: ``pair_decimals`` (price decimals), ``lot_decimals`` (volume decimals), ``cost_decimals``,
  ``tick_size``, ``ordermin`` (base currency), ``costmin`` (quote currency), ``status`` ('online',
  'cancel_only', 'post_only', 'limit_only', 'reduce_only').
* Ticker (one request for all pairs): ``a``/``b`` = [price, whole lot volume, lot volume] of the best ask/bid,
  ``c`` = [last trade price, lot volume], ``v``/``p``/``t``/``l``/``h`` = [today, last 24 h], ``o`` = today's
  opening price; "today" starts at 00:00 UTC. The answer carries no timestamp.
* OHLC: rows ``[time, open, high, low, close, vwap, volume, count]`` (prices as strings), at most the newest
  720 rows whatever ``since`` says. The LAST row is the current, not-yet-committed candle; ``last`` is the open
  time of the newest committed one. Minutes without trades are filled with flat candles (count 0, vwap 0,
  volume 0): they are DROPPED here, so that REST candles, the trades history and the WebSocket candles follow
  one rule - no trade, no bar. ``interval=1440`` gives daily candles dated by the UTC day (up to 720 days).
* Trades: rows ``[price, volume, time, 'b'|'s', 'm'|'l', misc, trade_id]`` (time = epoch seconds with a
  fraction), at most 1000 per request, oldest first. ``since`` takes epoch seconds or the nanosecond ``last``
  value of the previous page and is INCLUSIVE (the trade at ``last`` comes again; duplicates are removed by
  trade id). Older 1-minute history than the 720 OHLC rows is built from these trades.

Quotes: ``change_pct`` and ``prev_close`` refer to the UTC-day open (Ticker ``o``): Kraken's daily candle opens
with the first trade after 00:00 UTC - in practice the previous UTC day's close - so the change is measured
against the same reference as the daily bars. The WebSocket ticker's own ``change_pct`` is a rolling 24 h
figure and is deliberately not used (see ``kraken_stream``).

Politeness: one token bucket for ALL requests of the provider (default 1 request per second sustained, burst
2; history pages of the trades backfill yield to every other request) and a circuit breaker that pauses REST
access after a rate-limit answer (60 s, doubling per repetition up to 15 minutes, reset by the next success).
"""

import asyncio
import contextlib
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Any, Final

import httpx
import numpy as np
import pandas as pd

from trading_local.clock import Clock
from trading_local.data.provider import BarHandler, ProviderError, RateLimitError, StateHandler, TickHandler
from trading_local.db.repo_market import BAR_COLUMNS, BAR_SECONDS, empty_bars_1d, empty_bars_1m
from trading_local.models import Instrument, Quote

log = logging.getLogger(__name__)

REST_URL: Final = "https://api.kraken.com/0/public/"
VENUE: Final = "KRAKEN"  # calendar key of the instruments served by this provider
KRAKEN_VENUES: Final = frozenset({VENUE})
OHLC_LIMIT: Final = 720  # OHLC serves only the newest 720 candles of any interval
OHLC_MARGIN_MIN: Final = 5  # windows starting less than 720 - 5 minutes back are served by OHLC
BACKFILL_DAYS: Final = 14  # 1-minute history built from trades reaches this many calendar days back
TRADES_PAGE: Final = 1000  # maximum trades per request
MAX_TRADE_PAGES: Final = 5_000  # safety cap per symbol and window
USER_AGENT: Final = "trading-local/0.1 (market data, public endpoints)"

_RATE_LIMIT_MARKERS: Final = (
    "EAPI:Rate limit exceeded", "EGeneral:Too many requests", "EService:Throttled", "EService:Busy",
)  # fmt: skip
_UNKNOWN_PAIR: Final = "EQuery:Unknown asset pair"  # the whole request is refused if one pair is unknown


# ------------------------------------------------------------------------------------------------
# Pair metadata
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class KrakenPair:
    """One Kraken spot pair: names and trading increments (for the data feed now, a live broker later)."""

    symbol: str  # our instrument symbol, e.g. 'BTC-EUR'
    ws_symbol: str  # WebSocket v2 / REST display name, e.g. 'BTC/EUR'
    altname: str  # Kraken REST altname, e.g. 'XBTEUR'
    rest_pair: str  # canonical REST result key, e.g. 'XXBTZEUR'
    base: str  # Kraken asset id of the base currency, e.g. 'XXBT'
    quote: str  # Kraken asset id of the quote currency, e.g. 'ZEUR'
    price_decimals: int  # AssetPairs pair_decimals
    lot_decimals: int  # AssetPairs lot_decimals (order volume in base currency)
    cost_decimals: int  # AssetPairs cost_decimals (order cost in quote currency)
    tick_size: Decimal  # minimum price increment
    ordermin: Decimal  # minimum order volume in base currency
    costmin: Decimal  # minimum order cost in quote currency
    status: str = "online"  # 'online' | 'cancel_only' | 'post_only' | 'limit_only' | 'reduce_only' | 'unknown'

    @property
    def lot_step(self) -> Decimal:
        """Smallest order-volume increment (10 ** -lot_decimals)."""
        return Decimal(1).scaleb(-self.lot_decimals)


def _pair(
    symbol: str, ws: str, alt: str, rest: str, base: str, price_dec: int, cost_dec: int, tick: str, omin: str
) -> KrakenPair:
    return KrakenPair(
        symbol=symbol, ws_symbol=ws, altname=alt, rest_pair=rest, base=base, quote="ZEUR",
        price_decimals=price_dec, lot_decimals=8, cost_decimals=cost_dec, tick_size=Decimal(tick),
        ordermin=Decimal(omin), costmin=Decimal("0.45"),
    )  # fmt: skip


# AssetPairs values of 2026-11-22 (all 'online'); refreshed from AssetPairs at run time (once a day), so a
# changed minimum order size reaches ``KrakenProvider.pairs()`` without a code change.
KRAKEN_PAIRS: Final[Mapping[str, KrakenPair]] = MappingProxyType({
    pair.symbol: pair
    for pair in (
        _pair("BTC-EUR", "BTC/EUR", "XBTEUR", "XXBTZEUR", "XXBT", 1, 5, "0.1", "0.00006"),
        _pair("ETH-EUR", "ETH/EUR", "ETHEUR", "XETHZEUR", "XETH", 2, 5, "0.01", "0.002"),
        _pair("SOL-EUR", "SOL/EUR", "SOLEUR", "SOLEUR", "SOL", 2, 5, "0.01", "0.05"),
        _pair("XRP-EUR", "XRP/EUR", "XRPEUR", "XXRPZEUR", "XXRP", 5, 8, "0.00001", "4"),
        _pair("ADA-EUR", "ADA/EUR", "ADAEUR", "ADAEUR", "ADA", 6, 6, "0.000001", "20"),
        _pair("DOGE-EUR", "DOGE/EUR", "XDGEUR", "XDGEUR", "XXDG", 7, 9, "0.0000001", "50"),
        _pair("AVAX-EUR", "AVAX/EUR", "AVAXEUR", "AVAXEUR", "AVAX", 3, 5, "0.001", "0.5"),
        _pair("LINK-EUR", "LINK/EUR", "LINKEUR", "LINKEUR", "LINK", 5, 6, "0.00001", "0.35"),
        _pair("DOT-EUR", "DOT/EUR", "DOTEUR", "DOTEUR", "DOT", 4, 5, "0.0001", "3.9"),
        _pair("LTC-EUR", "LTC/EUR", "LTCEUR", "XLTCZEUR", "XLTC", 2, 5, "0.01", "0.08"),
    )
})


def _decimal(value: Any, default: Decimal) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return default
    return number if number.is_finite() and number >= 0 else default


def _int(value: Any, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else default


def update_pairs(pairs: Mapping[str, KrakenPair], result: Mapping[str, Any]) -> tuple[dict[str, KrakenPair], list[str]]:
    """Apply an AssetPairs ``result`` to ``pairs``; returns (updated pairs, symbols missing in the answer).

    An entry is found under the requested WS name, or by its altname / canonical key. Fields that are missing
    or malformed keep their previous value; a pair missing in the answer gets status 'unknown'.
    """
    by_alt = {str(info.get("altname")): info for info in result.values() if isinstance(info, Mapping)}
    updated: dict[str, KrakenPair] = {}
    missing: list[str] = []
    for symbol, pair in pairs.items():
        info = result.get(pair.ws_symbol) or result.get(pair.rest_pair) or by_alt.get(pair.altname)
        if not isinstance(info, Mapping):
            missing.append(symbol)
            updated[symbol] = replace(pair, status="unknown")
            continue
        status = info.get("status")
        updated[symbol] = replace(
            pair,
            altname=str(info.get("altname") or pair.altname),
            base=str(info.get("base") or pair.base),
            quote=str(info.get("quote") or pair.quote),
            price_decimals=_int(info.get("pair_decimals"), pair.price_decimals),
            lot_decimals=_int(info.get("lot_decimals"), pair.lot_decimals),
            cost_decimals=_int(info.get("cost_decimals"), pair.cost_decimals),
            tick_size=_decimal(info.get("tick_size"), pair.tick_size),
            ordermin=_decimal(info.get("ordermin"), pair.ordermin),
            costmin=_decimal(info.get("costmin"), pair.costmin),
            status=status if isinstance(status, str) and status else pair.status,
        )
    return updated, missing


def result_entry(result: Mapping[str, Any], pair: KrakenPair) -> Any:
    """The part of a per-pair ``result`` that belongs to ``pair`` (keyed by the requested WS name, the canonical
    key or the altname; a single non-'last' key is taken as is). None if the pair is not in the answer."""
    for key in (pair.ws_symbol, pair.rest_pair, pair.altname):
        if key in result:
            return result[key]
    keys = [key for key in result if key != "last"]
    return result[keys[0]] if len(keys) == 1 else None


# ------------------------------------------------------------------------------------------------
# Parsing (pure)
# ------------------------------------------------------------------------------------------------
def _float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive(value: Any) -> float | None:
    number = _float(value)
    return number if number is not None and number > 0 else None


def _first(values: Any) -> Any:
    return values[0] if isinstance(values, Sequence) and not isinstance(values, str) and values else None


def _ohlc_table(rows: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(open times int64, OHLCV float64 (n, 5), trade counts int64) of valid OHLC rows, ascending and unique."""
    ts_list: list[int] = []
    values: list[tuple[float, float, float, float, float]] = []
    counts: list[int] = []
    for row in rows if isinstance(rows, Sequence) else ():
        if not isinstance(row, Sequence) or isinstance(row, str) or len(row) < 8:
            continue
        ts = row[0]
        prices = [_positive(value) for value in row[1:5]]
        volume = _float(row[6])
        count = row[7]
        if not isinstance(ts, int) or isinstance(ts, bool) or any(p is None for p in prices) or volume is None:
            continue
        open_, high, low, close = prices  # type: ignore[misc]
        if not (low <= min(open_, close) and high >= max(open_, close)):  # type: ignore[operator]
            continue
        ts_list.append(ts)
        values.append((open_, high, low, close, max(0.0, volume)))  # type: ignore[arg-type]
        counts.append(count if isinstance(count, int) and not isinstance(count, bool) else 0)
    ts_arr = np.asarray(ts_list, dtype=np.int64)
    val_arr = np.asarray(values, dtype=np.float64).reshape(-1, len(BAR_COLUMNS))
    cnt_arr = np.asarray(counts, dtype=np.int64)
    order = np.argsort(ts_arr, kind="stable")
    ts_arr, val_arr, cnt_arr = ts_arr[order], val_arr[order], cnt_arr[order]
    keep = np.r_[ts_arr[1:] != ts_arr[:-1], True] if len(ts_arr) else np.zeros(0, dtype=bool)  # last row wins
    return ts_arr[keep], val_arr[keep], cnt_arr[keep]


def ohlc_minute_frame(rows: Any, last: Any, *, start_ts: int | None = None, end_ts: int | None = None) -> pd.DataFrame:
    """OHLC (interval 1) rows -> canonical 1-minute frame of COMMITTED candles with at least one trade.

    Rows newer than ``last`` (the forming candle) and filler rows without trades (count 0) are dropped, as are
    rows outside ``[start_ts, end_ts)``. Without a valid ``last`` the newest row is treated as forming.
    """
    ts, values, counts = _ohlc_table(rows)
    if ts.size == 0:
        return empty_bars_1m()
    committed = int(last) if isinstance(last, int) and not isinstance(last, bool) else int(ts[-1]) - 1
    keep = (ts <= committed) & (counts > 0) & (ts % BAR_SECONDS == 0)
    if start_ts is not None:
        keep &= ts >= start_ts
    if end_ts is not None:
        keep &= ts < end_ts
    return pd.DataFrame(values[keep], index=pd.Index(ts[keep], name="ts"), columns=list(BAR_COLUMNS))


def ohlc_daily_frame(rows: Any, *, start_ts: int | None = None) -> pd.DataFrame:
    """OHLC (interval 1440) rows -> canonical daily frame keyed by the UTC date of the candle's open.

    The newest row may be the running UTC day (the caller keeps completed sessions only).
    """
    ts, values, _ = _ohlc_table(rows)
    keep = ts % 86_400 == 0
    if start_ts is not None:
        keep &= ts + 86_400 > start_ts
    if not keep.any():
        return empty_bars_1d()
    days = [datetime.fromtimestamp(int(value), UTC).strftime("%Y-%m-%d") for value in ts[keep]]
    return pd.DataFrame(values[keep], index=pd.Index(days, dtype="str", name="day"), columns=list(BAR_COLUMNS))


@dataclass(frozen=True, slots=True)
class TradePage:
    """One parsed Trades answer: (trade id, time, price, volume) rows plus the ``last`` cursor."""

    ids: np.ndarray  # int64
    times: np.ndarray  # float64 epoch seconds
    prices: np.ndarray  # float64
    volumes: np.ndarray  # float64
    last: str | None
    raw_count: int  # rows in the answer before validation (decides whether the source has caught up)


def parse_trades(rows: Any, last: Any) -> TradePage:
    """Trades rows -> ``TradePage`` (invalid rows dropped, order of the answer kept)."""
    ids: list[int] = []
    times: list[float] = []
    prices: list[float] = []
    volumes: list[float] = []
    raw = rows if isinstance(rows, Sequence) and not isinstance(rows, str) else ()
    for row in raw:
        if not isinstance(row, Sequence) or isinstance(row, str) or len(row) < 7:
            continue
        price, volume, at, trade_id = _positive(row[0]), _float(row[1]), _float(row[2]), row[6]
        if price is None or volume is None or volume < 0 or at is None or at <= 0:
            continue
        if not isinstance(trade_id, int) or isinstance(trade_id, bool):
            continue
        ids.append(trade_id)
        times.append(at)
        prices.append(price)
        volumes.append(volume)
    cursor = str(last) if isinstance(last, (str, int)) and not isinstance(last, bool) and str(last).isdigit() else None
    return TradePage(
        np.asarray(ids, dtype=np.int64),
        np.asarray(times, dtype=np.float64),
        np.asarray(prices, dtype=np.float64),
        np.asarray(volumes, dtype=np.float64),
        cursor,
        len(raw),
    )


def trades_to_minutes(
    ids: np.ndarray, times: np.ndarray, prices: np.ndarray, volumes: np.ndarray,
    *, start_ts: int | None = None, end_ts: int | None = None,
) -> pd.DataFrame:  # fmt: skip
    """Aggregate trades into canonical 1-minute bars (minute of the trade time; minutes without trades get no
    bar). Trades are ordered by (time, trade id): open = first, close = last, high/low = extremes, volume = sum.
    Only bars with an open time in ``[start_ts, end_ts)`` are returned. Duplicate trade ids count once."""
    if ids.size == 0:
        return empty_bars_1m()
    _, unique = np.unique(ids, return_index=True)
    ids, times, prices, volumes = ids[unique], times[unique], prices[unique], volumes[unique]
    order = np.lexsort((ids, times))
    times, prices, volumes = times[order], prices[order], volumes[order]
    minutes = (np.floor(times / BAR_SECONDS) * BAR_SECONDS).astype(np.int64)
    first = np.flatnonzero(np.r_[True, minutes[1:] != minutes[:-1]])
    last = np.r_[first[1:], minutes.size] - 1
    frame = pd.DataFrame(
        {
            "open": prices[first],
            "high": np.maximum.reduceat(prices, first),
            "low": np.minimum.reduceat(prices, first),
            "close": prices[last],
            "volume": np.add.reduceat(volumes, first),
        },
        index=pd.Index(minutes[first], name="ts"),
    )
    keep = np.ones(len(frame), dtype=bool)
    if start_ts is not None:
        keep &= frame.index.to_numpy() >= start_ts
    if end_ts is not None:
        keep &= frame.index.to_numpy() < end_ts
    return frame[keep]


def quote_from_ticker(symbol: str, raw: Any, received_ts: int) -> Quote | None:
    """One Ticker entry -> ``Quote`` (source 'quote'); None without a positive last price.

    The REST ticker has no timestamp: ``ts`` is the time of the request. Day values are "today" since 00:00
    UTC; ``prev_close`` and ``change_pct`` refer to today's opening price (see the module docstring).
    """
    if not isinstance(raw, Mapping):
        return None
    price = _positive(_first(raw.get("c")))
    if price is None:
        return None
    day_open = _positive(raw.get("o"))
    volume = _float(_first(raw.get("v")))
    return Quote(
        symbol=symbol,
        ts=received_ts,
        received_ts=received_ts,
        price=price,
        bid=_positive(_first(raw.get("b"))),
        ask=_positive(_first(raw.get("a"))),
        prev_close=day_open,
        day_open=day_open,
        day_high=_positive(_first(raw.get("h"))),
        day_low=_positive(_first(raw.get("l"))),
        day_volume=volume if volume is not None and volume >= 0 else None,
        change_pct=(price / day_open - 1.0) * 100.0 if day_open else None,
        market_state=None,
        source="quote",
    )


def api_errors(body: Any) -> list[str]:
    """The error strings of a Kraken answer ('W...' warnings are ignored)."""
    errors = body.get("error") if isinstance(body, Mapping) else None
    if not isinstance(errors, list):
        return []
    return [error for error in errors if isinstance(error, str) and not error.startswith("W")]


def is_rate_limit(errors: Sequence[str]) -> bool:
    return any(marker in error for error in errors for marker in _RATE_LIMIT_MARKERS)


# ------------------------------------------------------------------------------------------------
# Politeness
# ------------------------------------------------------------------------------------------------
class AsyncRateLimiter:
    """Token bucket for one event loop: ``rate`` requests per second, bursts up to ``capacity``.

    ``acquire(background=True)`` (history pages) only takes a token while no foreground request is waiting,
    so live work (quotes, reconciliation, daily bars) never queues behind a long backfill.
    """

    def __init__(
        self,
        rate: float = 1.0,
        capacity: float = 2.0,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        if rate <= 0 or capacity < 1:
            raise ValueError("rate must be positive and capacity at least 1")
        self._rate = rate
        self._capacity = capacity
        self._monotonic = monotonic
        self._sleep = sleep
        self._tokens = capacity
        self._stamp = monotonic()
        self._urgent = 0

    async def acquire(self, *, background: bool = False) -> None:
        if not background:
            self._urgent += 1
        try:
            while True:
                now = self._monotonic()
                self._tokens = min(self._capacity, self._tokens + (now - self._stamp) * self._rate)
                self._stamp = now
                if background and self._urgent:
                    await self._sleep(1.0 / self._rate)
                    continue
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                await self._sleep((1.0 - self._tokens) / self._rate)
        finally:
            if not background:
                self._urgent -= 1


class KrakenBreaker:
    """Pauses REST access after a rate-limit answer: ``initial_s``, doubled per further trip until a call
    succeeds, at most ``max_s``. A trip while paused does not escalate."""

    def __init__(self, clock: Clock, *, initial_s: int = 60, max_s: int = 900) -> None:
        self._clock = clock
        self._initial_s = initial_s
        self._max_s = max_s
        self._until: int | None = None
        self._level = 0
        self.reason = ""

    def paused_until(self) -> int | None:
        return self._until if self._until is not None and self._until > self._clock.ts() else None

    def trip(self, reason: str) -> int:
        now = self._clock.ts()
        if self._until is not None and self._until > now:
            return self._until
        self._level += 1
        self._until = now + min(self._initial_s * 2 ** (self._level - 1), self._max_s)
        self.reason = reason
        log.warning("Kraken REST access paused for %d s: %s", self._until - now, reason)
        return self._until

    def record_success(self) -> None:
        if self._until is None or self._until <= self._clock.ts():
            self._level = 0

    def check(self) -> None:
        until = self.paused_until()
        if until is not None:
            raise RateLimitError(f"Kraken REST access paused until {until} ({self.reason})")


# ------------------------------------------------------------------------------------------------
# Provider
# ------------------------------------------------------------------------------------------------
class KrakenProvider:
    """``Provider`` (and ``CandleProvider``) for the KRAKEN pairs of ``KRAKEN_PAIRS`` (see the module docstring).

    Construction touches neither network nor disk; the HTTP client is created on first use.
    """

    def __init__(
        self,
        *,
        clock: Clock,
        pairs: Mapping[str, KrakenPair] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        limiter: AsyncRateLimiter | None = None,
        breaker: KrakenBreaker | None = None,
        stream_options: Any = None,
        stream_connector: Any = None,
        base_url: str = REST_URL,
        timeout_s: float = 15.0,
        pairs_max_age_s: int = 86_400,
    ) -> None:
        self._clock = clock
        self._pairs: dict[str, KrakenPair] = dict(KRAKEN_PAIRS if pairs is None else pairs)
        self._transport = transport
        self._limiter = limiter or AsyncRateLimiter()
        self._breaker = breaker or KrakenBreaker(clock)
        self._stream_options = stream_options
        self._stream_connector = stream_connector
        self._base_url = base_url
        self._timeout_s = timeout_s
        self._pairs_max_age_s = pairs_max_age_s
        self._pairs_checked_ts: int | None = None
        self._pairs_lock = asyncio.Lock()
        self._client: httpx.AsyncClient | None = None
        self._closed = False
        self.requests = 0  # HTTP requests made (status and tests)

    # -- metadata -------------------------------------------------------------------------------
    def pairs(self) -> dict[str, KrakenPair]:
        """Current pair metadata by our symbol (copy): names, decimals, tick size, minimum order size."""
        return dict(self._pairs)

    def pair(self, symbol: str) -> KrakenPair | None:
        return self._pairs.get(symbol)

    async def refresh_pairs(self) -> list[str]:
        """Re-read the pair metadata from AssetPairs (one request); returns the symbols Kraken did not list.

        Kraken refuses a request that names ONE unknown pair as a whole ('EQuery:Unknown asset pair', checked
        live): then the unfiltered pair list (one larger answer) shows which pair is gone; it gets status
        'unknown' and is left out of the batched Ticker request."""
        try:
            result = await self._get("AssetPairs", {"pair": ",".join(p.ws_symbol for p in self._pairs.values())})
        except RateLimitError:
            raise
        except ProviderError as exc:
            if _UNKNOWN_PAIR not in str(exc):
                raise
            log.warning("Kraken no longer knows one of the pairs; reading the full pair list")
            result = await self._get("AssetPairs", {})
        self._pairs, missing = update_pairs(self._pairs, result)
        self._pairs_checked_ts = self._clock.ts()
        offline = [p.symbol for p in self._pairs.values() if p.status not in ("online", "unknown")]
        if missing:
            log.warning("Kraken did not list the pairs %s", ", ".join(missing))
        if offline:
            log.warning("Kraken pairs not fully online: %s", ", ".join(f"{s} ({self._pairs[s].status})" for s in offline))
        return missing

    async def ensure_pairs(self) -> None:
        """Validate the pair metadata once a day (failures keep the known table and retry after an hour)."""
        now = self._clock.ts()
        if self._pairs_checked_ts is not None and now - self._pairs_checked_ts < self._pairs_max_age_s:
            return
        async with self._pairs_lock:
            if self._pairs_checked_ts is not None and self._clock.ts() - self._pairs_checked_ts < self._pairs_max_age_s:
                return
            try:
                await self.refresh_pairs()
            except RateLimitError:
                raise
            except ProviderError as exc:
                log.warning("Kraken pair metadata could not be refreshed: %s", exc)
                self._pairs_checked_ts = self._clock.ts() - self._pairs_max_age_s + 3_600

    # -- Provider -------------------------------------------------------------------------------
    async def fetch_minutes(
        self, symbols: Sequence[str], *, start_ts: int, end_ts: int | None = None, prepost: bool = False
    ) -> dict[str, pd.DataFrame]:
        """Closed 1-minute bars with open time in ``[start_ts, end_ts)``.

        Windows that start within the newest 720 minutes come from ONE OHLC request per pair (committed candles
        with trades only); older windows are built from the trades endpoint (paginated, at background priority).
        ``prepost`` has no meaning for a 24/7 market. Pairs that fail individually are logged and left out; a
        rate limit aborts the call.
        """
        if end_ts is not None and end_ts <= start_ts:
            return {}
        known = [symbol for symbol in dict.fromkeys(symbols) if symbol in self._pairs]
        if not known:
            return {}
        await self.ensure_pairs()
        now_minute = self._clock.ts() // BAR_SECONDS * BAR_SECONDS
        recent = start_ts >= now_minute - (OHLC_LIMIT - OHLC_MARGIN_MIN) * BAR_SECONDS
        frames: dict[str, pd.DataFrame] = {}
        failures: list[str] = []
        for symbol in known:
            pair = self._pairs[symbol]
            try:
                if recent:
                    frame = await self._ohlc_minutes(pair, start_ts, end_ts)
                else:
                    frame = await self._trade_minutes(pair, start_ts, end_ts)
            except RateLimitError:
                raise
            except ProviderError as exc:
                log.warning("Kraken 1-minute bars of %s failed: %s", symbol, exc)
                failures.append(symbol)
                continue
            if not frame.empty:
                frames[symbol] = frame
        if failures and len(failures) == len(known):
            raise ProviderError(f"Kraken 1-minute bars failed for {', '.join(failures)}")
        return frames

    async def fetch_daily(self, symbols: Sequence[str], *, start_ts: int) -> dict[str, pd.DataFrame]:
        """UTC-dated daily bars from OHLC interval 1440 (one request per pair, at most the newest 720 days)."""
        known = [symbol for symbol in dict.fromkeys(symbols) if symbol in self._pairs]
        if not known:
            return {}
        await self.ensure_pairs()
        frames: dict[str, pd.DataFrame] = {}
        failures: list[str] = []
        for symbol in known:
            pair = self._pairs[symbol]
            try:
                result = await self._get("OHLC", {"pair": pair.ws_symbol, "interval": 1440})
            except RateLimitError:
                raise
            except ProviderError as exc:
                log.warning("Kraken daily bars of %s failed: %s", symbol, exc)
                failures.append(symbol)
                continue
            frame = ohlc_daily_frame(result_entry(result, pair), start_ts=start_ts)
            if not frame.empty:
                frames[symbol] = frame
        if failures and len(failures) == len(known):
            raise ProviderError(f"Kraken daily bars failed for {', '.join(failures)}")
        return frames

    async def fetch_quotes(self, symbols: Sequence[str]) -> list[Quote]:
        """Snapshot quotes of all known pairs in ONE Ticker request.

        Pairs that Kraken no longer lists (status 'unknown') are left out: a single unknown pair makes Kraken
        refuse the whole request. Such a refusal schedules a fresh pair validation for the next call."""
        known = [symbol for symbol in dict.fromkeys(symbols) if symbol in self._pairs]
        if not known:
            return []
        await self.ensure_pairs()
        pairs = [self._pairs[symbol] for symbol in known if self._pairs[symbol].status != "unknown"]
        if not pairs:
            return []
        try:
            result = await self._get("Ticker", {"pair": ",".join(pair.ws_symbol for pair in pairs)})
        except RateLimitError:
            raise
        except ProviderError as exc:
            if _UNKNOWN_PAIR in str(exc):
                self._pairs_checked_ts = None  # find the pair that is gone before the next request
            raise
        received = self._clock.ts()
        quotes = [quote_from_ticker(pair.symbol, result_entry(result, pair), received) for pair in pairs]
        return [quote for quote in quotes if quote is not None]

    async def lookup(self, symbol: str) -> Instrument | None:
        """The default instrument of a known Kraken pair that Kraken still lists; None otherwise."""
        from trading_local.universe import DEFAULT_INSTRUMENTS  # the instrument rows live in the universe

        key = symbol.strip().upper()
        if key not in self._pairs:
            return None
        await self.ensure_pairs()
        if self._pairs[key].status == "unknown":
            return None
        return next((i for i in DEFAULT_INSTRUMENTS if i.symbol == key and i.exchange in KRAKEN_VENUES), None)

    def open_stream(self, on_tick: TickHandler, on_state: StateHandler) -> Any:
        """Ticker-only stream (quotes, no candles)."""
        return self._make_stream(on_tick, None, on_state)

    def open_candle_stream(self, on_tick: TickHandler, on_bar: BarHandler, on_state: StateHandler) -> Any:
        """Stream of ticker quotes plus the exchange's closed 1-minute candles (``ohlc`` channel)."""
        return self._make_stream(on_tick, on_bar, on_state)

    def paused_until(self) -> int | None:
        return self._breaker.paused_until()

    def trip(self, reason: str) -> None:
        self._breaker.trip(reason)

    async def close(self) -> None:
        self._closed = True
        client, self._client = self._client, None
        if client is not None:
            with contextlib.suppress(Exception):
                await client.aclose()

    # -- internals ------------------------------------------------------------------------------
    def _make_stream(self, on_tick: TickHandler, on_bar: BarHandler | None, on_state: StateHandler) -> Any:
        from trading_local.data.kraken_stream import KrakenStream

        def resolve(symbol: str) -> str | None:
            pair = self._pairs.get(symbol)
            return None if pair is None else pair.ws_symbol

        return KrakenStream(
            on_tick, on_bar, on_state, clock=self._clock, resolve=resolve, options=self._stream_options,
            connector=self._stream_connector,
        )  # fmt: skip

    async def _ohlc_minutes(self, pair: KrakenPair, start_ts: int, end_ts: int | None) -> pd.DataFrame:
        # 'since' only trims the answer; the window is cut locally (its boundary semantics do not matter).
        params = {"pair": pair.ws_symbol, "interval": 1, "since": max(0, start_ts - BAR_SECONDS)}
        result = await self._get("OHLC", params)
        return ohlc_minute_frame(result_entry(result, pair), result.get("last"), start_ts=start_ts, end_ts=end_ts)

    async def _trade_minutes(self, pair: KrakenPair, start_ts: int, end_ts: int | None) -> pd.DataFrame:
        """Bars of ``[start_ts, end_ts)`` from the trades endpoint: pages from ``start_ts`` on until a trade at or
        after the window end (or the present) is reached. Without ``end_ts`` the last two minutes before the
        request are left out (their trades may still be in flight)."""
        if end_ts is None:
            end_ts = (self._clock.ts() // BAR_SECONDS - 1) * BAR_SECONDS
        since = str(int(start_ts))
        parts: list[TradePage] = []
        newest_id = -1
        for _ in range(MAX_TRADE_PAGES):
            result = await self._get("Trades", {"pair": pair.ws_symbol, "since": since, "count": TRADES_PAGE},
                                     background=True)  # fmt: skip
            page = parse_trades(result_entry(result, pair), result.get("last"))
            fresh = page.ids > newest_id
            if fresh.any():
                newest_id = int(page.ids[fresh].max())
                parts.append(page)
            reached_end = bool(page.times.size) and float(page.times.max()) >= end_ts
            caught_up = page.raw_count < TRADES_PAGE
            if reached_end or caught_up or page.last is None:
                break
            # 'since' is inclusive: the next page starts with the trade at 'last'. A full page of trades
            # sharing one timestamp would never advance - skip that nanosecond then.
            since = page.last if page.last != since else str(int(page.last) + 1)
        else:
            log.warning("Kraken trades of %s: stopped after %d pages", pair.symbol, MAX_TRADE_PAGES)
        if not parts:
            return empty_bars_1m()
        return trades_to_minutes(
            np.concatenate([p.ids for p in parts]), np.concatenate([p.times for p in parts]),
            np.concatenate([p.prices for p in parts]), np.concatenate([p.volumes for p in parts]),
            start_ts=start_ts, end_ts=end_ts,
        )  # fmt: skip

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout_s,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                transport=self._transport,
            )
        return self._client

    async def _get(self, endpoint: str, params: Mapping[str, Any], *, background: bool = False) -> dict[str, Any]:
        """One public GET; returns ``result``. Raises ``RateLimitError`` (breaker tripped) or ``ProviderError``."""
        if self._closed:
            raise ProviderError("provider is closed")
        self._breaker.check()
        await self._limiter.acquire(background=background)
        self._breaker.check()
        self.requests += 1
        try:
            response = await self._http().get(endpoint, params=dict(params))
        except httpx.HTTPError as exc:
            raise ProviderError(f"{endpoint} request failed: {type(exc).__name__}: {exc}") from exc
        if response.status_code == 429:
            raise RateLimitError(f"rate limited, paused until {self._breaker.trip(f'{endpoint}: HTTP 429')}")
        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderError(f"{endpoint}: HTTP {response.status_code}, no JSON answer") from exc
        errors = api_errors(body)
        if is_rate_limit(errors):
            until = self._breaker.trip(f"{endpoint}: {'; '.join(errors)}")
            raise RateLimitError(f"rate limited by Kraken, paused until {until}")
        if errors:
            raise ProviderError(f"{endpoint}: {'; '.join(errors)}")
        if response.status_code != 200:
            raise ProviderError(f"{endpoint}: HTTP {response.status_code}")
        result = body.get("result") if isinstance(body, Mapping) else None
        if not isinstance(result, dict):
            raise ProviderError(f"{endpoint}: answer without result")
        self._breaker.record_success()
        return result


__all__ = [
    "AsyncRateLimiter",
    "BACKFILL_DAYS",
    "KRAKEN_PAIRS",
    "KRAKEN_VENUES",
    "KrakenBreaker",
    "KrakenPair",
    "KrakenProvider",
    "OHLC_LIMIT",
    "REST_URL",
    "TradePage",
    "VENUE",
    "api_errors",
    "is_rate_limit",
    "ohlc_daily_frame",
    "ohlc_minute_frame",
    "parse_trades",
    "quote_from_ticker",
    "result_entry",
    "trades_to_minutes",
    "update_pairs",
]
