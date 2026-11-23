"""Currency conversion from the FX pairs of the instrument universe (implements ``contracts.FxRates``).

A pair is an instrument named '<BASE><CCY>=X' whose price is the number of CCY per ONE unit of BASE:
'EURUSD=X' at 1.10 means 1 EUR = 1.10 USD, so ``rate('EUR', 'USD') = 1.10`` and
``rate('USD', 'EUR') = 1 / 1.10``. For ``rate(currency, base)`` the pair '<base><currency>=X' is tried
first (inverted), then '<currency><base>=X'. If neither exists, a cross through a pivot currency is used
(USD first, then the configured base currency), e.g. JPY -> USD = JPY -> EUR -> USD.

Price sources
* latest (``at_ts=None``): the in-memory quote, else the newest stored 1-minute bar, else the newest daily close.
  Results are cached for a few seconds because valuations ask for the same rates many times per second.
* historical (``at_ts`` given): the close of the newest 1-minute bar that was CLOSED at ``at_ts``, else the
  close of the newest daily bar of a UTC day that had ended by then (FX days are UTC days). Both rules use
  only data that was known at ``at_ts`` - never a later price.
"""

import logging
import math
import threading
import time
from collections.abc import Callable

import pandas as pd

from trading_local.barstore import BarReader
from trading_local.contracts import MarketData
from trading_local.db import repo_market
from trading_local.db.database import Database
from trading_local.db.repo_market import InstrumentBook

log = logging.getLogger(__name__)

_PIVOT_CURRENCY = "USD"
DEFAULT_CACHE_TTL_S = 5.0


class FxTable:
    """Thread-safe ``FxRates`` implementation backed by the market-data quote book and the stored bars."""

    def __init__(
        self,
        *,
        db: Database,
        instruments: InstrumentBook,
        data: MarketData,
        base_currency: str,
        cache_ttl_s: float = DEFAULT_CACHE_TTL_S,
        timer: Callable[[], float] = time.monotonic,
    ) -> None:
        self._db = db
        self._instruments = instruments
        self._data = data
        self._pivots = tuple(dict.fromkeys((_PIVOT_CURRENCY, base_currency)))
        self._cache_ttl_s = cache_ttl_s
        self._timer = timer  # monotonic seconds; only ages cache entries, never decides about prices
        self._lock = threading.Lock()  # guards _cache; never held during I/O
        self._cache: dict[tuple[str, str], tuple[float, float | None]] = {}  # (ccy, base) -> (expiry, rate)

    def rate(self, currency: str, base: str, at_ts: int | None = None) -> float | None:
        """Units of ``base`` per ONE unit of ``currency`` (see the module docstring); 1.0 for equal
        currencies, None when no rate is known. Never raises: database errors are logged and yield None."""
        if currency == base:
            return 1.0
        if at_ts is not None:
            return self._resolve_safely(currency, base, int(at_ts))
        key = (currency, base)
        now = self._timer()
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None and cached[0] > now:
            return cached[1]
        value = self._resolve_safely(currency, base, None)
        with self._lock:
            self._cache[key] = (now + self._cache_ttl_s, value)
        return value

    # -- resolution -----------------------------------------------------------------------------
    def _resolve_safely(self, currency: str, base: str, at_ts: int | None) -> float | None:
        try:
            return self._resolve(currency, base, at_ts)
        except Exception:  # noqa: BLE001 - a conversion problem must not break valuations
            log.exception("fx rate %s -> %s (at %s) could not be determined", currency, base, at_ts)
            return None

    def _resolve(self, currency: str, base: str, at_ts: int | None) -> float | None:
        direct = self._pair_rate(currency, base, at_ts)
        if direct is not None:
            return direct
        for pivot in self._pivots:
            if pivot in (currency, base):
                continue
            to_pivot = self._pair_rate(currency, pivot, at_ts)
            if to_pivot is None:
                continue
            from_pivot = self._pair_rate(pivot, base, at_ts)
            if from_pivot is not None:
                return to_pivot * from_pivot
        return None

    def _pair_rate(self, currency: str, base: str, at_ts: int | None) -> float | None:
        """Rate from one pair instrument (either orientation), or None if no pair has a usable price."""
        for symbol, inverted in ((f"{base}{currency}=X", True), (f"{currency}{base}=X", False)):
            if symbol not in self._instruments:
                continue
            price = self._price(symbol, at_ts)
            if price is not None:
                return 1.0 / price if inverted else price
        return None

    def _price(self, symbol: str, at_ts: int | None) -> float | None:
        """First usable price of the pair from the sources of the module docstring (in that order)."""
        return self._latest_price(symbol) if at_ts is None else self._price_at(symbol, at_ts)

    def _latest_price(self, symbol: str) -> float | None:
        quote = self._data.quote(symbol)
        if quote is not None and _usable(quote.price):
            return float(quote.price)
        bar = repo_market.last_bar(self._db, symbol)
        if bar is not None and _usable(bar.close):
            return bar.close
        return _daily_close(repo_market.read_bars_1d(self._db, symbol, limit=1))

    def _price_at(self, symbol: str, at_ts: int) -> float | None:
        # No calendar: daily bars count as complete once their UTC day has ended (the FX convention).
        reader = BarReader(self._db, self._instruments, None, at_ts)
        price = reader.last_price(symbol)
        if price is not None and _usable(price):
            return price
        return _daily_close(reader.daily(symbol, limit=1))


def _usable(price: float) -> bool:
    return math.isfinite(price) and price > 0


def _daily_close(frame: pd.DataFrame) -> float | None:
    """Close of the last row of a daily frame, None if the frame is empty or the price unusable."""
    if frame.empty:
        return None
    close = float(frame["close"].iloc[-1])
    return close if _usable(close) else None
