"""Data-source abstraction of the market-data module.

``MarketDataService`` talks to its source only through ``Provider`` and ``TickStream``: tests inject a fake,
and further (real-time) sources can be added without touching the service.

Shapes
* 1-minute frames: canonical bar frames of ``trading_local.db.repo_market`` - index ``ts`` = int64 epoch
  seconds of the bar OPEN time (UTC), ascending and unique; float64 columns open, high, low, close, volume.
  Rows without valid prices are removed, nothing is invented. Volume is the source's value (the service
  decides per asset class whether it is meaningful).
* daily frames: index ``day`` = 'YYYY-MM-DD' session date of the venue (UTC date for 24h markets), same columns.
* A symbol the source returned nothing for is missing from the result dict.
* All timestamps are epoch seconds UTC; prices are in the unit the source quotes (London shares in pence).
"""

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

from trading_local.models import Bar, Instrument, Quote


class ProviderError(Exception):
    """A request to the data source failed (network, unexpected answer). Per-call; the caller skips it."""


class RateLimitError(ProviderError):
    """The source rate-limited us, or REST access is paused by the circuit breaker. No request was made
    (or the request was refused): retry after ``Provider.paused_until()``."""


@dataclass(frozen=True, slots=True)
class Tick:
    """One price update from a streaming source.

    ``ts`` is the source's own time of the update (epoch seconds, may carry a fraction), ``received_ts``
    the local arrival time. Optional fields are None when the message did not carry them. ``day_volume``
    is the source's cumulative volume of the current session (it can also be a rolling 24 h figure for
    crypto - the consumer decides whether to use it). ``change_pct`` is in percent versus the previous close.
    ``market_state`` uses the source's vocabulary ('PRE', 'REGULAR', 'POST', 'EXTENDED').
    """

    symbol: str
    ts: float
    price: float
    received_ts: float
    day_volume: float | None = None
    change_pct: float | None = None
    market_state: str | None = None
    day_open: float | None = None
    day_high: float | None = None
    day_low: float | None = None
    prev_close: float | None = None
    bid: float | None = None
    ask: float | None = None


type TickHandler = Callable[[Tick], None]
type StateHandler = Callable[[bool], None]
type BarHandler = Callable[[Bar], None]


@runtime_checkable
class TickStream(Protocol):
    """A streaming connection (possibly several sharded connections) that keeps itself alive.

    The handlers given to ``Provider.open_stream`` are called on the event loop that called ``connect``:
    ``on_tick`` for every decoded update, ``on_state(connected)`` whenever the overall connection state
    changes. Handler exceptions are logged and never stop the stream.
    """

    @property
    def connected(self) -> bool:
        """True while every connection of the stream is established."""
        ...

    async def connect(self) -> None:
        """Start the background connection tasks and return immediately (reconnects happen automatically)."""
        ...

    async def subscribe(self, symbols: Collection[str]) -> None:
        """Replace the subscribed symbol set (applied on the live connections and after every reconnect)."""
        ...

    async def close(self) -> None:
        """Stop all connections and wait for their tasks to finish (bounded by a few seconds)."""
        ...


@runtime_checkable
class Provider(Protocol):
    """REST access plus a stream factory. Methods raise ``RateLimitError`` while REST access is paused or the
    source rate-limits, ``ProviderError`` for other failures of the whole call, and ``ValueError`` for
    arguments the source cannot serve (programming errors)."""

    async def fetch_minutes(
        self, symbols: Sequence[str], *, start_ts: int, end_ts: int | None = None, prepost: bool = False
    ) -> dict[str, pd.DataFrame]:
        """1-minute bars with open time in ``[start_ts, end_ts)`` (``end_ts=None``: up to now).

        The last row may still be forming; the caller drops rows that are not closed. ``prepost=True`` adds
        pre/post-market bars, including the closing-auction print stamped at the session end.
        """
        ...

    async def fetch_daily(self, symbols: Sequence[str], *, start_ts: int) -> dict[str, pd.DataFrame]:
        """Daily bars from ``start_ts`` up to today; the newest row may be a session that is still running."""
        ...

    async def fetch_quotes(self, symbols: Sequence[str]) -> list[Quote]:
        """Snapshot quotes for all ``symbols`` in as few requests as the source allows.

        Symbols the source does not know (or that carry no price) are missing from the result.
        """
        ...

    async def lookup(self, symbol: str) -> Instrument | None:
        """Validate ``symbol`` at the source and derive its instrument record; None if unknown or unsupported."""
        ...

    def open_stream(self, on_tick: TickHandler, on_state: StateHandler) -> TickStream:
        """Create (but do not connect) the streaming connection."""
        ...

    def paused_until(self) -> int | None:
        """Epoch seconds until which REST access is paused by the circuit breaker; None if not paused."""
        ...

    def trip(self, reason: str) -> None:
        """Open the circuit breaker as after a rate-limit error. For callers that detect failures the source
        library hides (e.g. empty answers for most symbols of open markets)."""
        ...

    async def close(self) -> None:
        """Release the resources (worker threads, sessions). The provider is unusable afterwards."""
        ...


@runtime_checkable
class CandleProvider(Provider, Protocol):
    """A ``Provider`` whose stream delivers the exchange's own 1-minute candles besides price ticks.

    ``on_bar`` receives every candle once its minute is over (a CLOSED bar with source 'stream'; a later
    correction of the same minute arrives as another call). The service then builds no bars from ticks or
    snapshot quotes for this provider's symbols: ticks only update the quote book.
    """

    def open_candle_stream(self, on_tick: TickHandler, on_bar: BarHandler, on_state: StateHandler) -> TickStream:
        """Create (but do not connect) the streaming connection with candle delivery."""
        ...


__all__ = [
    "BarHandler",
    "CandleProvider",
    "Provider",
    "ProviderError",
    "RateLimitError",
    "StateHandler",
    "Tick",
    "TickHandler",
    "TickStream",
]
