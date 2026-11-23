"""Thread-safe in-memory book of the latest quote per symbol.

Updates come from stream ticks and from batched snapshot quotes. Merge rules:
* An update at least as new as the stored quote (``ts`` >= stored ``ts``) wins: every field it carries
  replaces the stored value, fields it does not carry (None) keep their old value; ``ts``, ``received_ts``,
  ``price`` and ``source`` always come from the update.
* An OLDER update never overwrites anything; it only fills fields the stored quote does not know yet
  (e.g. the previous close from a snapshot that is a few seconds older than the newest tick).
* ``change_pct`` (percent versus the previous close) is computed from ``prev_close`` whenever the update
  changes the price without delivering a change itself.

The book remembers, per consumer ('publish' for the GUI events, 'persist' for the database), which
symbols changed since that consumer last called ``take_changed``. All returned quotes are copies.
"""

import threading
from collections.abc import Iterable, Sequence
from dataclasses import fields, replace

from trading_local.data.provider import Tick
from trading_local.models import Quote

CONSUMERS: tuple[str, ...] = ("publish", "persist")

# Optional fields that an update may or may not carry (None = not delivered).
_OPTIONAL_FIELDS: tuple[str, ...] = tuple(
    field.name for field in fields(Quote) if field.name not in ("symbol", "ts", "received_ts", "price", "source")
)


def quote_from_tick(tick: Tick) -> Quote:
    """The quote a stream tick represents (``source`` 'stream', times floored to whole seconds)."""
    return Quote(
        symbol=tick.symbol,
        ts=int(tick.ts),
        received_ts=int(tick.received_ts),
        price=tick.price,
        bid=tick.bid,
        ask=tick.ask,
        prev_close=tick.prev_close,
        day_open=tick.day_open,
        day_high=tick.day_high,
        day_low=tick.day_low,
        day_volume=tick.day_volume,
        change_pct=tick.change_pct,
        market_state=tick.market_state,
        source="stream",
    )


def _change_pct(price: float, prev_close: float | None) -> float | None:
    if prev_close is None or prev_close <= 0:
        return None
    return (price / prev_close - 1.0) * 100.0


def merge_quotes(stored: Quote | None, update: Quote) -> Quote:
    """The quote that results from applying ``update`` to ``stored`` under the module's merge rules."""
    if stored is None:
        if update.change_pct is None:
            return replace(update, change_pct=_change_pct(update.price, update.prev_close))
        return replace(update)
    if update.ts >= stored.ts:
        merged = replace(
            update,
            **{name: getattr(stored, name) for name in _OPTIONAL_FIELDS if getattr(update, name) is None},
        )
        if update.change_pct is None:
            computed = _change_pct(merged.price, merged.prev_close)
            if computed is not None:
                merged.change_pct = computed
            elif merged.price != stored.price:
                merged.change_pct = None  # the stored change belonged to another price
        return merged
    filled = {name: getattr(update, name) for name in _OPTIONAL_FIELDS if getattr(stored, name) is None}
    merged = replace(stored, **{name: value for name, value in filled.items() if value is not None})
    if merged.change_pct is None:
        merged.change_pct = _change_pct(merged.price, merged.prev_close)
    return merged


class QuoteBook:
    def __init__(self, consumers: Sequence[str] = CONSUMERS) -> None:
        self._lock = threading.Lock()
        self._quotes: dict[str, Quote] = {}
        self._changed: dict[str, set[str]] = {consumer: set() for consumer in consumers}

    def load(self, quotes: Iterable[Quote]) -> None:
        """Add persisted quotes (merged like updates) WITHOUT marking them as changed."""
        with self._lock:
            for quote in quotes:
                self._quotes[quote.symbol] = merge_quotes(self._quotes.get(quote.symbol), quote)

    def update(self, quote: Quote) -> bool:
        """Merge one quote; returns True if the stored quote changed."""
        with self._lock:
            stored = self._quotes.get(quote.symbol)
            merged = merge_quotes(stored, quote)
            if merged == stored:
                return False
            self._quotes[quote.symbol] = merged
            for changed in self._changed.values():
                changed.add(quote.symbol)
            return True

    def update_from_tick(self, tick: Tick) -> bool:
        return self.update(quote_from_tick(tick))

    def get(self, symbol: str) -> Quote | None:
        with self._lock:
            quote = self._quotes.get(symbol)
            return None if quote is None else replace(quote)

    def snapshot(self) -> dict[str, Quote]:
        with self._lock:
            return {symbol: replace(quote) for symbol, quote in self._quotes.items()}

    def take_changed(self, consumer: str) -> list[Quote]:
        """Quotes that changed since ``consumer``'s previous call (sorted by symbol); resets its change set.

        Raises ``KeyError`` for a consumer that was not declared in the constructor.
        """
        with self._lock:
            symbols = sorted(self._changed[consumer])
            self._changed[consumer] = set()
            return [replace(self._quotes[symbol]) for symbol in symbols if symbol in self._quotes]

    def remove(self, symbol: str) -> None:
        with self._lock:
            self._quotes.pop(symbol, None)
            for changed in self._changed.values():
                changed.discard(symbol)


__all__ = ["CONSUMERS", "QuoteBook", "merge_quotes", "quote_from_tick"]
