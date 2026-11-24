"""Yahoo Finance REST access through yfinance 1.8.0 (behaviour verified by the market-data probe).

Verified facts this module is built on
* yfinance is not safe for concurrent use: every blocking call runs in ONE dedicated worker thread.
* ONE curl_cffi session is registered with yfinance's ``YfData`` singleton and passed to every call
  (``yf.download`` without ``session=`` replaces the singleton's session and costs extra crumb requests).
* ``yf.download`` costs one request per symbol; it HIDES rate-limit and network errors behind empty frames
  and only logs them to the 'yfinance' logger - a handler on that logger turns them into exceptions here.
* Snapshot quotes for all symbols come from ``/v7/finance/quote`` in ONE request (needs the crumb).
* 1-minute history: at most 8 days per request and only the last 30 days; epoch ints for start/end (date
  strings are read in each exchange's time zone); never ``period='max'``, ``repair=True`` or
  ``ignore_tz=True`` for intraday data.
* The last 1-minute row of a live market is a flat snapshot of the current minute, not a bar (the caller
  drops rows that are not closed).
* Seeding yfinance's time-zone cache from quote responses avoids one extra request per new symbol.

Politeness: a token bucket caps the request rate, and a circuit breaker pauses ALL REST work after a
rate-limit error (10 minutes, doubling on repetition, reset by the next successful call).
"""

import asyncio
import contextlib
import functools
import logging
import math
import re
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_local.clock import Clock
from trading_local.data.provider import ProviderError, RateLimitError, StateHandler, TickHandler
from trading_local.data.stream import StreamOptions, YahooStream
from trading_local.db.repo_market import BAR_COLUMNS, BAR_SECONDS, empty_bars_1d, empty_bars_1m
from trading_local.models import AssetClass, EarningsDate, Instrument, Quote
from trading_local.universe import (
    REGION_SORT_BASE,
    USER_SORT_OFFSET,
    VENUE_REGION,
    VENUE_TIMEZONE,
    YAHOO_EXCHANGE_TO_VENUE,
    default_benchmark,
    is_tradable,
    normalize_currency,
)

log = logging.getLogger(__name__)

QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
QUOTE_CHUNK = 200  # symbols per snapshot request (1,083 in one request were verified to work)
DOWNLOAD_THREADS = 4  # parallel chart requests inside one yf.download call
MAX_MINUTE_WINDOW_S = 8 * 86_400 - 60  # Yahoo: at most 8 days of 1-minute bars per request
MAX_MINUTE_AGE_S = 30 * 86_400 - 3_600  # Yahoo: 1-minute bars only for the last 30 days (one hour of margin)
EARNINGS_LIMIT = 8  # earnings releases per request: the next one and the latest seven

# Requested quote fields (Yahoo returns ~50 always-on keys besides; unrequested keys can be wrong).
SNAPSHOT_FIELDS: tuple[str, ...] = (
    "regularMarketPrice", "regularMarketTime", "regularMarketVolume", "regularMarketOpen",
    "regularMarketDayHigh", "regularMarketDayLow", "regularMarketPreviousClose", "regularMarketChangePercent",
    "bid", "ask", "marketState", "exchangeDataDelayedBy", "currency", "exchange", "exchangeTimezoneName",
    "quoteType",
)  # fmt: skip
LOOKUP_FIELDS: tuple[str, ...] = (*SNAPSHOT_FIELDS, "shortName", "longName", "fullExchangeName")

_QUOTE_TYPES: Mapping[str, AssetClass] = {
    "EQUITY": "equity",
    "ETF": "equity",
    "INDEX": "index",
    "CRYPTOCURRENCY": "crypto",
    "CURRENCY": "fx",
    "FUTURE": "future",
}
_YF_FIELDS: Mapping[str, str] = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
_RATE_LIMIT_MARKERS: tuple[str, ...] = ("Too Many Requests", "YFRateLimitError", "Rate limited")
_LEGAL_SUFFIX = re.compile(
    r"[\s,]+(inc\.?|incorporated|corp\.?|corporation|co\.?|company|ag|se|plc|n\.?v\.?|s\.?a\.?|ltd\.?|limited|"
    r"a/s|aktiengesellschaft|holdings?|k\.?k\.?)$",
    re.IGNORECASE,
)


# ------------------------------------------------------------------------------------------------
# Request budget and circuit breaker
# ------------------------------------------------------------------------------------------------
class TokenBucket:
    """Thread-safe token bucket: ``rate`` tokens per second, at most ``capacity`` tokens stored.

    ``acquire(n)`` waits until ``min(n, capacity)`` tokens are available and then takes ``n`` (the balance may
    go negative, so a call larger than the capacity is paid off by the following callers' waits).
    """

    def __init__(self, rate: float, capacity: float, *, monotonic: Callable[[], float] = time.monotonic) -> None:
        if rate <= 0 or capacity <= 0:
            raise ValueError("rate and capacity must be positive")
        self._rate = rate
        self._capacity = capacity
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._tokens = capacity
        self._stamp = monotonic()

    def wait_time(self, tokens: float) -> float:
        """Seconds until ``tokens`` could be acquired (0.0 if right now)."""
        with self._lock:
            self._refill()
            return self._wait_time(tokens)

    def acquire(self, tokens: float, stop: threading.Event) -> bool:
        """Block until the tokens are taken; False if ``stop`` was set while waiting."""
        while True:
            with self._lock:
                self._refill()
                delay = self._wait_time(tokens)
                if delay <= 0:
                    self._tokens -= tokens
                    return True
            if stop.wait(delay):
                return False

    def _wait_time(self, tokens: float) -> float:
        needed = min(tokens, self._capacity)
        return max(0.0, (needed - self._tokens) / self._rate)

    def _refill(self) -> None:
        now = self._monotonic()
        self._tokens = min(self._capacity, self._tokens + (now - self._stamp) * self._rate)
        self._stamp = now


class CircuitBreaker:
    """Pauses REST access after a rate-limit error: ``initial_s`` seconds, doubled for every further trip
    until a call succeeds again, at most ``max_s``. A trip while already paused does not escalate."""

    def __init__(self, clock: Clock, *, initial_s: int = 600, max_s: int = 7_200) -> None:
        self._clock = clock
        self._initial_s = initial_s
        self._max_s = max_s
        self._lock = threading.Lock()
        self._until: int | None = None
        self._level = 0
        self.reason = ""

    def paused_until(self) -> int | None:
        with self._lock:
            return self._until if self._until is not None and self._until > self._clock.ts() else None

    def trip(self, reason: str) -> int:
        """Start (or keep) the pause; returns its end (epoch seconds)."""
        with self._lock:
            now = self._clock.ts()
            if self._until is not None and self._until > now:
                return self._until
            self._level += 1
            self._until = now + min(self._initial_s * 2 ** (self._level - 1), self._max_s)
            self.reason = reason
            log.warning("Yahoo REST access paused for %d s: %s", self._until - now, reason)
            return self._until

    def record_success(self) -> None:
        with self._lock:
            if self._until is None or self._until <= self._clock.ts():
                self._level = 0

    def check(self) -> None:
        """Raise ``RateLimitError`` while paused."""
        until = self.paused_until()
        if until is not None:
            raise RateLimitError(f"REST access paused until {until} ({self.reason})")


# ------------------------------------------------------------------------------------------------
# Frame conversion (pure)
# ------------------------------------------------------------------------------------------------
def split_download(raw: pd.DataFrame | None, symbols: Sequence[str]) -> dict[str, pd.DataFrame]:
    """Split a ``yf.download`` result into one frame per symbol with yfinance's field columns.

    Handles both column layouts (``group_by='ticker'``: levels Ticker/Price, ``group_by='column'``:
    levels Price/Ticker) and the flat columns of a single ticker with ``multi_level_index=False``.
    Keys are the requested symbols (yfinance upper-cases tickers; matching ignores case).
    """
    if raw is None or raw.empty:
        return {}
    by_upper = {symbol.upper(): symbol for symbol in symbols}
    columns = raw.columns
    if not isinstance(columns, pd.MultiIndex):
        return {symbols[0]: raw} if len(symbols) == 1 else {}
    field_level = 0 if "Close" in set(map(str, columns.get_level_values(0))) else 1
    ticker_level = 1 - field_level
    parts: dict[str, pd.DataFrame] = {}
    for ticker in dict.fromkeys(map(str, columns.get_level_values(ticker_level))):
        symbol = by_upper.get(ticker.upper())
        if symbol is not None:
            parts[symbol] = raw.xs(ticker, axis=1, level=ticker_level)
    return parts


def _field_arrays(sub: pd.DataFrame) -> np.ndarray:
    """(n, 5) float64 array open/high/low/close/volume from a yfinance per-symbol frame (no Volume -> 0)."""
    columns = {str(column): column for column in sub.columns}
    arrays = []
    for yf_name in _YF_FIELDS:
        column = columns.get(yf_name)
        if column is None:
            if yf_name != "Volume":
                return np.empty((0, len(BAR_COLUMNS)))
            arrays.append(np.zeros(len(sub)))
        else:
            arrays.append(pd.to_numeric(sub[column], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan))
    return np.column_stack(arrays) if arrays else np.empty((0, len(BAR_COLUMNS)))


def _valid_rows(values: np.ndarray) -> np.ndarray:
    prices = values[:, :4]
    return np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)


def round_significant(values: np.ndarray, digits: int = 7) -> np.ndarray:
    """Round positive finite values to ``digits`` significant digits.

    Yahoo's chart prices are 32-bit floats (60740.66 arrives as 60740.66015625); seven significant digits
    restore the quoted value without cutting FX quotes (1.0974539518 -> 1.097454).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        magnitude = np.floor(np.log10(np.abs(values)))
    factor = 10.0 ** np.where(np.isfinite(magnitude), digits - 1 - magnitude, 0.0)
    return np.round(values * factor) / factor


def _finish(values: np.ndarray) -> np.ndarray:
    """Storage form of valid rows: prices rounded (see ``round_significant``), missing/negative volume 0."""
    values[:, :4] = round_significant(values[:, :4])
    values[:, 4] = np.where(np.isfinite(values[:, 4]) & (values[:, 4] > 0), values[:, 4], 0.0)
    return values


def to_minute_frame(sub: pd.DataFrame, *, before_ts: int | None = None) -> pd.DataFrame:
    """yfinance 1-minute frame of one symbol -> canonical 1-minute frame.

    The (tz-aware) index is converted to UTC epoch seconds; a tz-naive index is taken as UTC. Rows without
    four finite positive prices are dropped (the union index of multi-symbol downloads has NaN rows for
    other symbols' minutes), as are unaligned rows (not on a full minute) and, with ``before_ts``, rows with
    an open time ``>= before_ts`` (the still-forming bar). Prices are rounded to seven significant digits,
    NaN volume becomes 0; duplicate minutes keep the last row.
    """
    if sub.empty or not isinstance(sub.index, pd.DatetimeIndex):
        return empty_bars_1m()
    index = sub.index if sub.index.tz is not None else sub.index.tz_localize("UTC")
    ts = index.tz_convert("UTC").as_unit("s").asi8
    values = _field_arrays(sub)
    if len(values) == 0:
        return empty_bars_1m()
    keep = _valid_rows(values) & (ts % BAR_SECONDS == 0)
    if before_ts is not None:
        keep &= ts < before_ts
    values = _finish(values[keep])
    frame = pd.DataFrame(values, index=pd.Index(ts[keep], name="ts", dtype=np.int64), columns=list(BAR_COLUMNS))
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def to_daily_frame(sub: pd.DataFrame) -> pd.DataFrame:
    """yfinance daily frame of one symbol -> canonical daily frame (day = the index's calendar date).

    ``yf.download`` gives a tz-naive index at local midnight, ``Ticker.history`` a tz-aware one; in both
    cases the date of the index is the venue's session date. Invalid rows are dropped as for minutes.
    """
    if sub.empty or not isinstance(sub.index, pd.DatetimeIndex):
        return empty_bars_1d()
    values = _field_arrays(sub)
    if len(values) == 0:
        return empty_bars_1d()
    keep = _valid_rows(values)
    days = sub.index.strftime("%Y-%m-%d")[keep]
    values = _finish(values[keep])
    frame = pd.DataFrame(values, index=pd.Index(days, dtype="str", name="day"), columns=list(BAR_COLUMNS))
    return frame[~frame.index.duplicated(keep="last")].sort_index()


# ------------------------------------------------------------------------------------------------
# Quote and instrument mapping (pure)
# ------------------------------------------------------------------------------------------------
def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _positive(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and number > 0 else None


def quote_from_yahoo(raw: Mapping[str, Any], received_ts: int) -> Quote | None:
    """One ``/v7/finance/quote`` result -> ``Quote`` (source 'quote'); None for stubs without price or time.

    Zero bid/ask/open/high/low (indices report 0.0) count as unknown. ``regularMarketVolume`` is the
    session's cumulative volume (crypto: rolling 24 h); ``regularMarketChangePercent`` is in percent.
    """
    symbol = raw.get("symbol")
    price = _positive(raw.get("regularMarketPrice"))
    ts = _number(raw.get("regularMarketTime"))
    if not isinstance(symbol, str) or price is None or ts is None:
        return None
    volume = _number(raw.get("regularMarketVolume"))
    state = raw.get("marketState")
    return Quote(
        symbol=symbol,
        ts=int(ts),
        received_ts=received_ts,
        price=price,
        bid=_positive(raw.get("bid")),
        ask=_positive(raw.get("ask")),
        prev_close=_positive(raw.get("regularMarketPreviousClose")),
        day_open=_positive(raw.get("regularMarketOpen")),
        day_high=_positive(raw.get("regularMarketDayHigh")),
        day_low=_positive(raw.get("regularMarketDayLow")),
        day_volume=volume if volume is not None and volume >= 0 else None,
        change_pct=_number(raw.get("regularMarketChangePercent")),
        market_state=state if isinstance(state, str) and state else None,
        source="quote",
    )


def clean_name(name: Any) -> str:
    """Yahoo names without the padding junk of ``shortName`` ('SAP SE                        I' -> 'SAP SE')."""
    if not isinstance(name, str):
        return ""
    return re.split(r"\s{2,}", name.strip())[0].strip()


def aliases_from_names(*names: Any) -> tuple[str, ...]:
    """News aliases from the long and short name: each cleaned name plus its form without legal suffixes
    ('Toyota Motor Corporation' -> 'Toyota Motor Corporation', 'Toyota Motor'). Order-preserving, unique,
    at least two characters."""
    aliases: list[str] = []
    for name in names:
        cleaned = clean_name(name)
        stripped = cleaned
        while (shorter := _LEGAL_SUFFIX.sub("", stripped).strip(" ,")) != stripped:
            stripped = shorter
        aliases.extend(alias for alias in (cleaned, stripped) if len(alias) >= 2)
    return tuple(dict.fromkeys(aliases))


def instrument_from_yahoo(raw: Mapping[str, Any]) -> Instrument | None:
    """Derive an instrument from one ``/v7/finance/quote`` result with ``LOOKUP_FIELDS``.

    None for stubs without a price, unsupported quote types (options, funds ...), venues without a
    calendar key (see ``universe.YAHOO_EXCHANGE_TO_VENUE``) or a missing currency.
    """
    symbol = raw.get("symbol")
    asset_class = _QUOTE_TYPES.get(str(raw.get("quoteType", "")))
    venue = YAHOO_EXCHANGE_TO_VENUE.get(str(raw.get("exchange", "")))
    currency_code = raw.get("currency")
    if (
        not isinstance(symbol, str)
        or asset_class is None
        or venue is None
        or not isinstance(currency_code, str)
        or not currency_code
        or _positive(raw.get("regularMarketPrice")) is None
    ):
        return None
    currency, price_scale = normalize_currency(currency_code)
    region = VENUE_REGION[venue]
    delay = _number(raw.get("exchangeDataDelayedBy"))
    long_name, short_name = raw.get("longName"), raw.get("shortName")
    return Instrument(
        symbol=symbol,
        name=clean_name(long_name) or clean_name(short_name) or symbol,
        asset_class=asset_class,
        region=region,
        exchange=venue,
        currency=currency,
        price_scale=price_scale,
        timezone=VENUE_TIMEZONE[venue],
        delay_min=max(0, int(delay)) if delay is not None else 0,
        tradable=is_tradable(symbol, asset_class),
        active=True,
        benchmark=default_benchmark(venue, asset_class),
        aliases=aliases_from_names(long_name, short_name),
        sort_order=REGION_SORT_BASE[region] + USER_SORT_OFFSET,
    )


# ------------------------------------------------------------------------------------------------
# yfinance log capture
# ------------------------------------------------------------------------------------------------
class _YfLogCapture(logging.Handler):
    """Collects the messages of the 'yfinance' logger while a call is running (yf.download reports its
    failures only there). Installed once; the logger stops propagating so the root log stays readable -
    this module logs a summary instead."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self._messages: list[str] | None = None

    def emit(self, record: logging.LogRecord) -> None:
        messages = self._messages
        if messages is not None:
            messages.append(record.getMessage())

    @contextlib.contextmanager
    def capture(self) -> Iterator[list[str]]:
        messages: list[str] = []
        self._messages = messages
        try:
            yield messages
        finally:
            self._messages = None

    def install(self) -> None:
        logger = logging.getLogger("yfinance")
        if self not in logger.handlers:
            logger.addHandler(self)
            logger.propagate = False


def _is_rate_limited(messages: Sequence[str]) -> bool:
    return any(marker in message for message in messages for marker in _RATE_LIMIT_MARKERS)


# ------------------------------------------------------------------------------------------------
# Earnings dates
# ------------------------------------------------------------------------------------------------
def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def earnings_from_yahoo(symbol: str, frame: pd.DataFrame | None) -> list[EarningsDate]:
    """Earnings releases from ``Ticker.get_earnings_dates``, newest first.

    The index holds the release times (tz-aware, New York time; a tz-naive index is read as New York time);
    columns 'EPS Estimate', 'Reported EPS', 'Surprise(%)'. A release at local midnight has no known time.
    The surprise is computed from estimate and actual when the source leaves it out.
    """
    if frame is None or frame.empty:
        return []
    found: list[EarningsDate] = []
    for stamp, row in frame.iterrows():
        try:
            moment = pd.Timestamp(stamp)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if moment is pd.NaT:
            continue
        if moment.tzinfo is None:
            moment = moment.tz_localize("America/New_York")
        estimate = _finite_or_none(row.get("EPS Estimate"))
        actual = _finite_or_none(row.get("Reported EPS"))
        surprise = _finite_or_none(row.get("Surprise(%)"))
        if surprise is None and actual is not None and estimate:
            surprise = (actual - estimate) / abs(estimate) * 100.0
        found.append(
            EarningsDate(
                symbol=symbol,
                ts=int(moment.timestamp()),
                time_known=not (moment.hour == 0 and moment.minute == 0),
                eps_estimate=estimate,
                eps_actual=actual,
                surprise_pct=surprise if actual is not None else None,
            )
        )
    return sorted(found, key=lambda release: release.ts, reverse=True)


# ------------------------------------------------------------------------------------------------
# Provider
# ------------------------------------------------------------------------------------------------
class YahooProvider:
    """``Provider`` implementation for Yahoo Finance (see the module docstring for the verified rules)."""

    def __init__(
        self,
        *,
        clock: Clock,
        cache_dir: Path | None = None,
        bucket: TokenBucket | None = None,
        breaker: CircuitBreaker | None = None,
        stream_options: StreamOptions | None = None,
    ) -> None:
        self._clock = clock
        self._cache_dir = cache_dir
        # 1.5 requests/s sustained leaves room for the live cadence (<= 26/min) next to a 1/s backfill.
        self._bucket = bucket or TokenBucket(rate=1.5, capacity=10)
        self._breaker = breaker or CircuitBreaker(clock)
        self._stream_options = stream_options or StreamOptions()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yahoo-rest")
        self._stop = threading.Event()
        self._capture = _YfLogCapture()
        self._session: Any = None  # curl_cffi session, created in the worker thread on first use
        self._tz_seeded: dict[str, str] = {}

    # -- Provider -------------------------------------------------------------------------------
    async def fetch_minutes(
        self, symbols: Sequence[str], *, start_ts: int, end_ts: int | None = None, prepost: bool = False
    ) -> dict[str, pd.DataFrame]:
        now = self._clock.ts()
        if (end_ts if end_ts is not None else now) - start_ts > MAX_MINUTE_WINDOW_S:
            raise ValueError("Yahoo serves at most 8 days of 1-minute bars per request")
        if start_ts < now - MAX_MINUTE_AGE_S:
            raise ValueError("Yahoo serves 1-minute bars only for the last 30 days")
        if not symbols:
            return {}
        return await self._run(self._minutes_job, tuple(symbols), int(start_ts), end_ts, prepost)

    async def fetch_daily(self, symbols: Sequence[str], *, start_ts: int) -> dict[str, pd.DataFrame]:
        if not symbols:
            return {}
        return await self._run(self._daily_job, tuple(symbols), int(start_ts))

    async def fetch_quotes(self, symbols: Sequence[str]) -> list[Quote]:
        if not symbols:
            return []
        return await self._run(self._quotes_job, tuple(dict.fromkeys(symbols)))

    async def lookup(self, symbol: str) -> Instrument | None:
        return await self._run(self._lookup_job, symbol)

    async def fetch_earnings(self, symbol: str) -> list[EarningsDate]:
        """Past and upcoming earnings releases (``Ticker.get_earnings_dates``, one request), newest first."""
        return await self._run(self._earnings_job, symbol)

    def open_stream(self, on_tick: TickHandler, on_state: StateHandler) -> YahooStream:
        return YahooStream(on_tick, on_state, clock=self._clock, options=self._stream_options)

    def paused_until(self) -> int | None:
        return self._breaker.paused_until()

    def trip(self, reason: str) -> None:
        self._breaker.trip(reason)

    async def close(self) -> None:
        self._stop.set()
        self._executor.shutdown(wait=False, cancel_futures=True)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(asyncio.to_thread(self._executor.shutdown, True), timeout=15)
        if self._session is not None:
            with contextlib.suppress(Exception):
                self._session.close()
            self._session = None

    # -- worker plumbing ------------------------------------------------------------------------
    async def _run[T](self, func: Callable[..., T], *args: Any) -> T:
        self._breaker.check()
        if self._stop.is_set():
            raise ProviderError("provider is closed")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, functools.partial(func, *args))

    def _prepare(self, requests: int) -> None:
        """Worker-thread preamble of every HTTP call: breaker check, session, request budget."""
        self._breaker.check()
        if self._session is None:
            self._open_session()
        if not self._bucket.acquire(requests, self._stop):
            raise ProviderError("provider is closed")

    def _open_session(self) -> None:
        import yfinance as yf
        from curl_cffi import requests as curl_requests
        from yfinance.data import YfData

        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            yf.set_tz_cache_location(str(self._cache_dir))
        yf.config.network.retries = 2  # retries transport errors (1 s, 2 s); never 429
        self._capture.install()
        self._session = curl_requests.Session(impersonate="chrome")
        YfData(session=self._session)

    def _rate_limited(self, detail: str) -> RateLimitError:
        until = self._breaker.trip(f"Yahoo rate limit ({detail})")
        return RateLimitError(f"rate limited by Yahoo, paused until {until}")

    def _download(self, symbols: Sequence[str], **kwargs: Any) -> pd.DataFrame | None:
        import yfinance as yf
        from yfinance.exceptions import YFException, YFRateLimitError

        self._prepare(len(symbols))
        with self._capture.capture() as messages:
            try:
                raw = yf.download(
                    list(symbols),
                    session=self._session,
                    threads=min(DOWNLOAD_THREADS, len(symbols)),
                    progress=False,
                    group_by="ticker",
                    auto_adjust=False,
                    actions=False,
                    multi_level_index=True,
                    **kwargs,
                )
            except YFRateLimitError as exc:
                raise self._rate_limited("download") from exc
            except (OSError, YFException) as exc:
                raise ProviderError(f"download failed: {exc}") from exc
        if _is_rate_limited(messages):
            raise self._rate_limited("download")
        if messages:
            log.debug("yfinance reported for %d symbols: %s", len(symbols), " | ".join(messages)[:500])
        if raw is not None and not raw.empty:
            self._breaker.record_success()
        return raw

    def _quote_request(self, symbols: Sequence[str], fields: Sequence[str]) -> list[dict[str, Any]]:
        from yfinance.data import YfData
        from yfinance.exceptions import YFException, YFRateLimitError

        results: list[dict[str, Any]] = []
        for start in range(0, len(symbols), QUOTE_CHUNK):
            chunk = symbols[start : start + QUOTE_CHUNK]
            self._prepare(1)
            params = {
                "symbols": ",".join(chunk),
                "formatted": "false",
                "lang": "en-US",
                "region": "US",
                "fields": ",".join(fields),
            }
            try:
                response = YfData(session=self._session).get(QUOTE_URL, params=params, timeout=10)
            except YFRateLimitError as exc:
                raise self._rate_limited("quote") from exc
            except (OSError, YFException) as exc:
                raise ProviderError(f"quote request failed: {exc}") from exc
            if response.status_code == 429:
                raise self._rate_limited("quote HTTP 429")
            if response.status_code != 200:
                raise ProviderError(f"quote request failed: HTTP {response.status_code}")
            try:
                body = response.json()["quoteResponse"]
            except (ValueError, KeyError, TypeError) as exc:
                raise ProviderError("unexpected quote response") from exc
            if body.get("error"):
                raise ProviderError(f"quote request failed: {body['error']}")
            results.extend(item for item in body.get("result") or [] if isinstance(item, dict))
        self._breaker.record_success()
        return results

    def _seed_timezones(self, results: Sequence[Mapping[str, Any]]) -> None:
        """Store each symbol's exchange time zone in yfinance's cache (saves a request per new symbol)."""
        from yfinance import cache as yf_cache

        tz_cache = yf_cache.get_tz_cache()
        for item in results:
            symbol, tz = item.get("symbol"), item.get("exchangeTimezoneName")
            if isinstance(symbol, str) and isinstance(tz, str) and self._tz_seeded.get(symbol) != tz:
                tz_cache.store(symbol, tz)
                self._tz_seeded[symbol] = tz

    # -- jobs (worker thread) -------------------------------------------------------------------
    def _minutes_job(
        self, symbols: tuple[str, ...], start_ts: int, end_ts: int | None, prepost: bool
    ) -> dict[str, pd.DataFrame]:
        raw = self._download(symbols, interval="1m", start=start_ts, end=end_ts, prepost=prepost)
        frames = {symbol: to_minute_frame(sub) for symbol, sub in split_download(raw, symbols).items()}
        return {symbol: frame for symbol, frame in frames.items() if not frame.empty}

    def _daily_job(self, symbols: tuple[str, ...], start_ts: int) -> dict[str, pd.DataFrame]:
        raw = self._download(symbols, interval="1d", start=start_ts)
        frames = {symbol: to_daily_frame(sub) for symbol, sub in split_download(raw, symbols).items()}
        return {symbol: frame for symbol, frame in frames.items() if not frame.empty}

    def _quotes_job(self, symbols: tuple[str, ...]) -> list[Quote]:
        results = self._quote_request(symbols, SNAPSHOT_FIELDS)
        self._seed_timezones(results)
        received = self._clock.ts()
        return [quote for item in results if (quote := quote_from_yahoo(item, received)) is not None]

    def _earnings_job(self, symbol: str) -> list[EarningsDate]:
        import yfinance as yf
        from yfinance.exceptions import YFException, YFRateLimitError

        self._prepare(1)
        with self._capture.capture() as messages:
            try:
                frame = yf.Ticker(symbol, session=self._session).get_earnings_dates(limit=EARNINGS_LIMIT)
            except YFRateLimitError as exc:
                raise self._rate_limited("earnings") from exc
            except (OSError, YFException, KeyError, ValueError, TypeError, AttributeError) as exc:
                raise ProviderError(f"earnings dates of {symbol} failed: {exc}") from exc
        if _is_rate_limited(messages):
            raise self._rate_limited("earnings")
        self._breaker.record_success()
        return earnings_from_yahoo(symbol, frame)

    def _lookup_job(self, symbol: str) -> Instrument | None:
        results = self._quote_request([symbol], LOOKUP_FIELDS)
        match = next((item for item in results if str(item.get("symbol", "")).upper() == symbol.upper()), None)
        if match is None:
            return None
        self._seed_timezones([match])
        return instrument_from_yahoo(match)


__all__ = [
    "CircuitBreaker",
    "LOOKUP_FIELDS",
    "SNAPSHOT_FIELDS",
    "TokenBucket",
    "YahooProvider",
    "aliases_from_names",
    "clean_name",
    "earnings_from_yahoo",
    "instrument_from_yahoo",
    "quote_from_yahoo",
    "round_significant",
    "split_download",
    "to_daily_frame",
    "to_minute_frame",
]
