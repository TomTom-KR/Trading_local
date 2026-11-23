"""Paper broker for the portfolios 'ai' (KI) and 'me' (Ich) - implements ``contracts.Broker``.

Both portfolios follow IDENTICAL rules, which is what makes the comparison fair: same start capital, base
currency, costs, fill rules and leverage limit. 'ai' is traded through ``execute_signal``; 'me' through the
manual order API.

Accounting (all money in the portfolio base currency)
* value = qty * price * price_scale * fx, fx = base units per unit of the instrument currency;
  entry_value = the value at the entry price and the entry rate ``entry_fx``
* price result of a position at price P and rate fx (before fees):
    LONG:  qty * price_scale * (P * fx - entry * entry_fx)   (bought with base currency: price AND currency risk)
    SHORT: -qty * price_scale * (P - entry) * fx            (the sale proceeds stay in the instrument currency, as
           in a real short or CFD account: only the result is exposed to the exchange rate, never the notional)
* open:  LONG cash -= entry_value + fee        SHORT cash += entry_value - fee
* close: cash += sign * entry_value + price result(exit, exit_fx) - fee, i.e.
         LONG cash += exit value - fee
         SHORT cash -= entry_value + qty * price_scale * (exit - entry) * exit_fx + fee
* realised P&L of a closed position = price result(exit, exit_fx) - all fees
* unrealised P&L of an open position = price result(mark, fx_now) - entry fee
* equity = cash + sum(sign * entry_value + price result(mark, fx_now)) - per position the market value of a LONG
  and -(entry_value + qty * price_scale * (mark - entry) * fx_now) for a SHORT - so that
  equity - start capital = realised + unrealised at every moment
* exposure (leverage, buying power, snapshots) = qty * mark * price_scale * fx_now per position (signed: shorts < 0)
* mark = latest quote, else the close of the newest closed 1-minute bar, else the entry price
* buying power = max(0, equity * max_gross_leverage - (long + short exposure))

Fills (no look-ahead; the rules themselves live in ``trading_local.execution``)
* An order created at T fills at the OPEN of the first CLOSED 1-minute bar with open time >= T, read through
  a ``BarReader`` as of the processing time. Until that bar has arrived and is settled the order stays
  pending - on delayed feeds this is honestly at least delay + 1 minutes later.
* Fills and exits act on SETTLED bars only (``BarReader.settled_bars_after`` / ``first_settled_bar_at_or_after``).
  Live, stream/quote bars are provisional and REST replaces them minutes later; after a restart, a sleep or an
  outage REST also inserts the missed minutes behind newer bars. A bar is used only once it can no longer change
  and no earlier traded minute can still arrive; until then an order stays pending and a position unchecked.
* Fills, exits and protective levels happen at the bar's time (``filled_ts``/``opened_ts``/``closed_ts`` are
  bar open times) and use the FX rate known at that time, so waiting for settlement delays a booking but never
  changes its price or result.
* ``process`` checks every bar of an open position exactly once, including the entry bar: ``checked_ts`` only
  advances over settled bars. A closing order fills on the first bar from its creation on in the same settled
  walk, so every earlier bar is examined for a stop or target first.
* Rules on the wall clock never run ahead of unsettled bars: a pending order expires only when no bar inside its
  lifetime has arrived and none can still arrive (every traded minute up to ``expire_ts`` is settled by age),
  and an overdue time stop or session flatten is executed only once no traded minute after the deadline is
  still young and every arrived bar of the position was examined.
* A signal order follows the rules of the tracked signal (``trading_local.analysis.tracking``): it is created at
  the issue time, expires at the signal's entry deadline (``tracking.entry_deadline``), uses the drift limit
  and the flatten time of ``ENGINE_PARAMS`` and counts the holding period (time stop) from the entry bar, so the
  AI portfolio enters and exits exactly when the signal journal does.
* New protective levels (``update_position`` at U) act like an order placed at U: they apply to the bars that
  open at or after U; bars that opened earlier but are examined later (delayed feed, settlement) keep the
  previous levels.

All state lives in the database: a new broker instance after a restart continues seamlessly. That includes the
levels that applied before a change through ``update_position`` (settings key ``PRIOR_LEVELS_KEY``, kept until
the bars before the change are checked): bars are checked only once settled, minutes after they opened, and a
restart in between must not judge them by the new levels. (The running max drawdown per portfolio is only a
cache of the stored snapshots and is rebuilt from them.) One re-entrant lock serialises every public method
(API worker threads and the runtime thread).
"""

import logging
import math
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta

import numpy as np

from trading_local import execution
from trading_local.analysis import tracking
from trading_local.analysis.params import ENGINE_PARAMS
from trading_local.barstore import BarReader
from trading_local.clock import Clock, floor_ts, from_ts, to_ts
from trading_local.config import CostParams, Settings
from trading_local.contracts import FxRates, MarketCalendar, MarketData, OrderRejected
from trading_local.db import repo_market, repo_ops, repo_portfolio, repo_signals
from trading_local.db.database import Database
from trading_local.db.repo_market import BAR_SECONDS, InstrumentBook
from trading_local.db.repo_ops import SettingsStore
from trading_local.dto import (
    EquityCurvesOut,
    OrderOut,
    OrderRequest,
    PortfolioDetailOut,
    PortfolioSummaryOut,
    PositionOut,
)
from trading_local.events import TOPIC_ORDER, TOPIC_PORTFOLIO, EventBus
from trading_local.models import (
    Direction,
    EquityPoint,
    Instrument,
    Order,
    OrderOrigin,
    OrderStatus,
    Portfolio,
    Position,
    Signal,
)
from trading_local.sim import metrics, sizing

log = logging.getLogger(__name__)

PORTFOLIO_KEYS: tuple[str, ...] = ("ai", "me")
MANUAL_ORDER_LIFETIME_S = 7 * 86_400  # pending manual orders expire after a week
SNAPSHOT_INTERVAL_S = 300  # periodic equity snapshots (timestamps floored to this step)
REDUCE_RISK_FACTOR = 0.5  # position-size multiplier when the LLM verdict is 'reduce' (applied after the caps)
PRIOR_LEVELS_KEY = "sim_prior_levels"  # settings key: levels in force before each update_position change
_METRICS_POINTS = 2_000  # equity points used for the Sharpe ratio (the drawdown uses every snapshot)
_ALL_ROWS = 1_000_000  # "no limit" for the repository listings
_MONEY_EPS = 1e-9

_SIDE_DE: dict[str, str] = {"LONG": "Long", "SHORT": "Short"}
_OPEN_ACTION_DE: dict[str, str] = {"LONG": "Kauf", "SHORT": "Leerverkauf"}
_EXIT_REASON_DE: dict[str, str] = {
    "manual": "manuell",
    "stop": "Stopp",
    "target": "Ziel",
    "time_stop": "Zeitstopp",
    "session_close": "Handelsschluss",
}
_STATUS_DE: dict[str, str] = {"pending": "offen", "filled": "ausgeführt", "cancelled": "storniert", "rejected": "abgelehnt"}

_LevelChange = tuple[int, float | None, float | None]  # (change time, stop before it, target before it)


# ------------------------------------------------------------------------------------------------
# Helper types
# ------------------------------------------------------------------------------------------------
@dataclass(slots=True)
class _Holding:
    """An open position marked to market (money in base currency)."""

    position: Position
    instrument: Instrument
    mark: float  # quoted units
    mark_ts: int | None  # time of the mark; None when the entry price is used
    fx: float

    @property
    def market_value(self) -> float:
        """Unsigned market value at the mark and the current rate (exposure, buying power)."""
        return sizing.value_base(self.position.qty, self.mark, self.instrument.price_scale, self.fx)

    @property
    def exposure(self) -> float:
        """Signed market value (negative for shorts)."""
        return self.position.sign * self.market_value

    @property
    def entry_value(self) -> float:
        p = self.position
        return sizing.value_base(p.qty, p.entry_price, self.instrument.price_scale, p.entry_fx)

    @property
    def price_result(self) -> float:
        """Result at the mark before any fee (see the module docstring)."""
        p = self.position
        return _price_result(p, self.instrument, self.mark, self.fx)

    @property
    def equity_value(self) -> float:
        """What the position adds to the equity: the market value of a LONG; for a SHORT the liability
        -(entry_value + qty * price_scale * (mark - entry) * fx_now) against the proceeds booked at the entry."""
        return self.position.sign * self.entry_value + self.price_result

    @property
    def unrealized_pnl(self) -> float:
        """Result if closed at the mark without exit costs, after the entry fee."""
        return self.price_result - self.position.fees_base


@dataclass(slots=True)
class _Valuation:
    portfolio: Portfolio
    holdings: list[_Holding]

    @property
    def long_exposure(self) -> float:
        return sum(h.market_value for h in self.holdings if h.position.side == "LONG")

    @property
    def short_exposure(self) -> float:
        return sum(h.market_value for h in self.holdings if h.position.side == "SHORT")

    @property
    def gross_exposure(self) -> float:
        return self.long_exposure + self.short_exposure

    @property
    def equity(self) -> float:
        return self.portfolio.cash + sum(h.equity_value for h in self.holdings)

    def buying_power(self, max_gross_leverage: float) -> float:
        return max(0.0, self.equity * max_gross_leverage - self.gross_exposure)


@dataclass(slots=True)
class _Notices:
    """What one public call announces once its changes are committed."""

    orders: list[OrderOut] = field(default_factory=list)
    changed: set[str] = field(default_factory=set)  # portfolio keys
    activities: list[tuple[str, str, str | None, int | None]] = field(default_factory=list)

    def order(self, out: OrderOut) -> None:
        self.orders.append(out)
        self.changed.add(out.portfolio_key)

    def activity(self, level: str, message: str, ref_type: str | None = None, ref_id: int | None = None) -> None:
        self.activities.append((level, message, ref_type, ref_id))


@dataclass(slots=True)
class _DrawdownCache:
    """Running drawdown over the equity snapshots of one comparison period that are older than the newest one."""

    reset_ts: int
    folded_until: int  # the snapshots with ts < folded_until are folded into ``state``
    state: metrics.Drawdown


class _Skip(Exception):
    """A signal is not traded; ``str(exc)`` is the German reason (lower-case sentence fragment)."""


# ------------------------------------------------------------------------------------------------
# Broker
# ------------------------------------------------------------------------------------------------
class PaperBroker:
    """``contracts.Broker`` for the two paper portfolios; see the module docstring for the rules."""

    def __init__(
        self,
        *,
        settings: Settings,
        db: Database,
        clock: Clock,
        bus: EventBus,
        calendar: MarketCalendar,
        instruments: InstrumentBook,
        runtime_settings: SettingsStore,
        costs: CostParams,
        data: MarketData,
        fx: FxRates,
    ) -> None:
        self._settings = settings
        self._db = db
        self._clock = clock
        self._bus = bus
        self._calendar = calendar
        self._instruments = instruments
        self._runtime_settings = runtime_settings
        self._costs = costs
        self._data = data
        self._fx = fx
        self._lock = threading.RLock()
        portfolios = repo_portfolio.ensure_portfolios(
            db, base_currency=settings.base_currency, start_cash=settings.start_capital, now_ts=clock.ts()
        )
        self._ids: dict[str, int] = {key: _stored_id(portfolios[key].id) for key in PORTFOLIO_KEYS}
        self._keys: dict[int, str] = {pid: key for key, pid in self._ids.items()}
        # Neither the name nor the base currency of a portfolio ever changes (a reset keeps both).
        self._names: dict[str, str] = {key: portfolios[key].name for key in PORTFOLIO_KEYS}
        self._bases: dict[int, str] = {self._ids[key]: portfolios[key].base_currency for key in PORTFOLIO_KEYS}
        # Position id -> [(change_ts, stop, target)], ascending: the levels that were in force BEFORE each change
        # through update_position. Kept until a bar opening at or after the change has been examined (see
        # _levels_per_bar) and stored under PRIOR_LEVELS_KEY, so a restart judges late bars by the same levels.
        self._prior_levels: dict[int, list[_LevelChange]] = self._load_prior_levels()
        # Portfolio id -> running max drawdown of its snapshots (see _max_drawdown_pct); rebuilt from the
        # database after a restart or a reset.
        self._drawdowns: dict[int, _DrawdownCache] = {}

    # ==========================================================================================
    # Manual orders
    # ==========================================================================================
    def place_order(self, portfolio_key: str, req: OrderRequest) -> OrderOut:
        with self._lock:
            now_ts = self._clock.ts()
            portfolio = self._portfolio_for_order(portfolio_key)
            instrument = self._tradable_instrument(req.symbol)
            if (req.amount_base is None) == (req.qty is None):
                raise OrderRejected("Gib entweder einen Betrag oder eine Stückzahl an – genau eines von beiden.")
            self._reject_symbol_conflict(portfolio, instrument)
            reader = self._reader(now_ts)
            last = self._last_price(reader, instrument.symbol)
            if last is None:
                raise OrderRejected(
                    f"Für {instrument.name} ist noch kein Kurs bekannt. Warte, bis Kursdaten eingetroffen sind."
                )
            price = last[0]
            _check_levels(instrument, req.side, price, req.stop_price, req.target_price)
            base = portfolio.base_currency
            fx = self._fx.rate(instrument.currency, base)
            if fx is None:
                raise OrderRejected(
                    f"Für {instrument.currency} → {base} ist kein Wechselkurs bekannt, daher kann die Order "
                    "nicht bewertet werden. Versuche es später erneut."
                )
            qty, value = _order_size(instrument, req, price, fx, base)
            if req.signal_id is not None and repo_signals.get_signal(self._db, req.signal_id) is None:
                raise OrderRejected(f"Das Signal #{req.signal_id} wurde nicht gefunden.")
            buying_power = self._valuation(portfolio, reader).buying_power(self._max_leverage())
            if value > buying_power + _MONEY_EPS:
                raise OrderRejected(
                    f"Der Auftragswert ({_money(value, base)}) übersteigt die verfügbare Kaufkraft "
                    f"({_money(buying_power, base)}). Wähle einen kleineren Betrag oder schließe zuerst andere Positionen."
                )
            order = repo_portfolio.insert_order(
                self._db,
                Order(
                    portfolio_id=self._ids[portfolio.key],
                    created_ts=now_ts,
                    symbol=instrument.symbol,
                    intent="open",
                    side=req.side,
                    origin="manual",
                    qty=qty,
                    notional_base=req.amount_base if qty is None else None,
                    ref_price=price,
                    stop_price=req.stop_price,
                    target_price=req.target_price,
                    signal_id=req.signal_id,
                    expire_ts=now_ts + MANUAL_ORDER_LIFETIME_S,
                ),
            )
            return self._announce(order, now_ts)

    def cancel_order(self, portfolio_key: str, order_id: int) -> OrderOut:
        with self._lock:
            now_ts = self._clock.ts()
            portfolio = self._portfolio_for_order(portfolio_key)
            order = repo_portfolio.get_order(self._db, order_id)
            if order is None or order.portfolio_id != self._ids[portfolio.key]:
                raise OrderRejected(f"Die Order #{order_id} wurde in diesem Depot nicht gefunden.")
            if order.status != "pending":
                raise OrderRejected(
                    f"Die Order #{order_id} kann nicht mehr storniert werden (Status: {_STATUS_DE[order.status]})."
                )
            order.status, order.status_reason = "cancelled", "storniert"
            repo_portfolio.update_order(self._db, order)
            return self._announce(order, now_ts)

    def close_position(self, portfolio_key: str, position_id: int) -> OrderOut:
        with self._lock:
            now_ts = self._clock.ts()
            portfolio = self._portfolio_for_order(portfolio_key)
            position = self._open_position_of(portfolio, position_id)
            instrument = self._instrument(position.symbol)
            pending = repo_portfolio.pending_orders(self._db, self._ids[portfolio.key])
            if any(o.intent == "close" and o.position_id == position.id for o in pending):
                raise OrderRejected(
                    f"Für die Position in {instrument.name} liegt bereits eine Schließungs-Order vor. "
                    "Warte, bis sie ausgeführt ist."
                )
            last = self._last_price(self._reader(now_ts), position.symbol)
            order = repo_portfolio.insert_order(
                self._db,
                Order(
                    portfolio_id=position.portfolio_id,
                    created_ts=now_ts,
                    symbol=position.symbol,
                    intent="close",
                    side=position.side,
                    origin="manual",
                    qty=position.qty,
                    ref_price=None if last is None else last[0],
                    position_id=position.id,
                    expire_ts=now_ts + MANUAL_ORDER_LIFETIME_S,
                ),
            )
            return self._announce(order, now_ts)

    def update_position(
        self, portfolio_key: str, position_id: int, *, stop_price: float | None, target_price: float | None
    ) -> PositionOut:
        with self._lock:
            now_ts = self._clock.ts()
            portfolio = self._portfolio_for_order(portfolio_key)
            position = self._open_position_of(portfolio, position_id)
            instrument = self._instrument(position.symbol)
            holding = self._holding(position, instrument, portfolio.base_currency, self._reader(now_ts))
            _check_levels(instrument, position.side, holding.mark, stop_price, target_price)
            stored_id = _stored_id(position.id)
            prior: _LevelChange = (now_ts, position.stop_price, position.target_price)
            unchanged = prior[1:] == (stop_price, target_price)
            history = None if unchanged else [*self._prior_levels.get(stored_id, []), prior]
            position.stop_price, position.target_price = stop_price, target_price
            with self._db.write():  # the new levels and the memory of the old ones in one transaction
                repo_portfolio.update_position(self._db, position)
                if history is not None:
                    self._store_prior_levels({**self._prior_levels, stored_id: history}, now_ts)
            if history is not None:
                self._prior_levels[stored_id] = history
            notices = _Notices()
            notices.changed.add(portfolio.key)
            self._flush(notices, now_ts)
            return self._position_out(position, instrument, holding)

    # ==========================================================================================
    # AI portfolio
    # ==========================================================================================
    def execute_signal(self, signal: Signal) -> OrderOut | None:
        with self._lock:
            now = self._clock.now()
            now_ts = to_ts(now)
            notices = _Notices()
            out: OrderOut | None = None
            label = f"Signal #{signal.id} ({signal.symbol} {_SIDE_DE.get(signal.direction, signal.direction)})"
            try:
                out = self._execute_signal(signal, now, notices)
            except _Skip as skip:
                notices.activity("info", f"Depot {self._names['ai']}: {label} nicht gehandelt – {skip}.", "signal", signal.id)
            except Exception:  # noqa: BLE001 - never raises: the engine must keep running
                log.exception("execute_signal failed for signal %s (%s)", signal.id, signal.symbol)
                notices.activity(
                    "error",
                    f"Depot {self._names['ai']}: {label} konnte wegen eines internen Fehlers nicht gehandelt werden.",
                    "signal",
                    signal.id,
                )
            self._flush(notices, now_ts)
            return out

    def _execute_signal(self, signal: Signal, now: datetime, notices: _Notices) -> OrderOut:
        if not signal.actionable or signal.status != "pending_entry":
            raise _Skip(f"das Signal ist nicht zum Handel freigegeben (Status: {signal.status})")
        instrument = self._instruments.get(signal.symbol)
        if instrument is None:
            raise _Skip(f"{signal.symbol} ist unbekannt")
        if not (instrument.active and instrument.tradable and sizing.is_tradable_class(instrument)):
            raise _Skip(f"{instrument.name} ist nicht handelbar")
        portfolio = self._portfolio("ai")
        pid = self._ids["ai"]
        existing = repo_portfolio.open_position_for_symbol(self._db, pid, instrument.symbol)
        if existing is not None:
            raise _Skip(f"für {instrument.name} besteht bereits eine {_SIDE_DE[existing.side]}-Position")
        pending = repo_portfolio.pending_orders(self._db, pid)
        if any(o.intent == "open" and o.symbol == instrument.symbol for o in pending):
            raise _Skip(f"für {instrument.name} liegt bereits eine offene Order vor")
        rules = self._runtime_settings.get()
        now_ts = to_ts(now)
        valuation = self._valuation(portfolio, self._reader(now_ts))
        occupied = len(valuation.holdings) + sum(1 for o in pending if o.intent == "open")
        if occupied >= rules.max_open_positions:
            raise _Skip(f"die Höchstzahl von {rules.max_open_positions} Positionen (inklusive offener Orders) ist erreicht")
        if not (_positive(signal.ref_price) and _positive(signal.stop_dist)):
            raise _Skip("kein gültiger Referenzkurs oder Stoppabstand")
        fx = self._fx.rate(instrument.currency, portfolio.base_currency)
        if fx is None:
            raise _Skip(f"kein Wechselkurs {instrument.currency} → {portfolio.base_currency} bekannt")
        # Size with the full risk and both caps, THEN scale for 'reduce': the position cap binds for typical
        # intraday stops, so reducing the risk percentage before the caps would change nothing.
        reduce = signal.llm_verdict == "reduce"
        raw_qty = sizing.risk_position_size(
            valuation.equity,
            rules.risk_per_trade_pct,
            signal.stop_dist,
            signal.ref_price,
            instrument.price_scale,
            fx,
            rules.max_position_pct,
            valuation.buying_power(rules.max_gross_leverage),
        )
        qty = sizing.round_qty(instrument, raw_qty * (REDUCE_RISK_FACTOR if reduce else 1.0))
        if qty <= 0:
            raise _Skip("die berechnete Positionsgröße ist kleiner als eine handelbare Einheit")
        risk = sizing.value_base(qty, signal.stop_dist, instrument.price_scale, fx)  # planned loss at the stop
        # expected_r is after slippage and proportional fees but before the FLAT order fees, which depend on the
        # position size: only those are checked here (counting the proportional fee again would double it).
        fee_r = 2.0 * self._costs.flat_fee(instrument) / risk
        if signal.expected_r - fee_r <= 0:
            raise _Skip(
                f"die Ordergebühren für Ein- und Ausstieg ({_de(fee_r, 2)} R) zehren den Erwartungswert "
                f"({_de(signal.expected_r, 2)} R) auf"
            )
        # The order follows the tracked signal: same first bar (created at the issue time), same entry deadline,
        # same drift limit and, once filled, the same time stop (counted from the entry bar), so the AI portfolio
        # and the signal journal always agree on the entry and the exit.
        issued_ts = now_ts if signal.issued_ts is None else int(signal.issued_ts)
        try:
            # provisional: the fill counts the holding period from the entry bar (see _position_time_stop)
            time_stop = self._calendar.add_trading_minutes(instrument.exchange, from_ts(issued_ts), signal.horizon_min)
        except KeyError as exc:
            raise _Skip(f"für den Handelsplatz {instrument.exchange} sind keine Handelszeiten bekannt") from exc
        deadline_ts = signal.entry_deadline_ts
        if deadline_ts is None:
            deadline_ts = tracking.entry_deadline(issued_ts, instrument, ENGINE_PARAMS)
        order = repo_portfolio.insert_order(
            self._db,
            Order(
                portfolio_id=pid,
                created_ts=issued_ts,
                symbol=instrument.symbol,
                intent="open",
                side=signal.direction,
                origin="signal",
                qty=qty,
                ref_price=signal.ref_price,
                stop_dist=signal.stop_dist,
                target_dist=signal.target_dist,
                max_drift=ENGINE_PARAMS.max_entry_drift * signal.stop_dist,
                time_stop_ts=to_ts(time_stop),
                flatten_at_close=signal.intraday,
                signal_id=signal.id,
                expire_ts=int(deadline_ts),
            ),
        )
        out = self._order_out(order)
        notices.order(out)
        effective_risk_pct = risk / valuation.equity * 100.0
        reduced = (
            f", Positionsgröße nach KI-Prüfung auf {_de(REDUCE_RISK_FACTOR * 100.0, 0)} % reduziert" if reduce else ""
        )
        notices.activity(
            "info",
            f"Depot {self._names['ai']}: Order für Signal #{signal.id} angelegt – {_OPEN_ACTION_DE[signal.direction]} "
            f"{_qty(qty)} {instrument.name} (Risiko {_de(effective_risk_pct, 2)} % des Depotwerts{reduced}).",
            "order",
            order.id,
        )
        return out

    # ==========================================================================================
    # Minute cycle
    # ==========================================================================================
    def process(self, now: datetime) -> list[Position]:
        with self._lock:
            notices = _Notices()
            closed: list[Position] = []
            now_ts = self._clock.ts()
            try:
                now_ts = to_ts(now)
                reader = self._reader(now_ts)
                close_times = self._pending_close_times()
                closed += self._protective_exits(reader, close_times, notices)
                closed += self._overdue_exits(reader, now, close_times, notices)
                closed += self._fill_closing_orders(reader, notices)
                self._fill_opening_orders(reader, notices)
                self._expire_orders(reader, notices)
                self._snapshots(reader, now_ts, notices)
            except Exception:  # noqa: BLE001 - never raises: the minute cycle must go on
                log.exception("broker pass at %s failed", now)
            self._flush(notices, now_ts)
            return closed

    def _pending_close_times(self) -> dict[int, int]:
        """Position id -> creation time of its oldest pending closing order.

        Such an order fills at the open of the first bar from its creation on, so automatic exits may only act
        on the bars before that one.
        """
        close_times: dict[int, int] = {}
        for order in repo_portfolio.pending_orders(self._db):
            if order.intent == "close" and order.position_id is not None:
                close_times[order.position_id] = min(order.created_ts, close_times.get(order.position_id, order.created_ts))
        return close_times

    # -- 1. protective exits ----------------------------------------------------------------------
    def _protective_exits(self, reader: BarReader, close_times: dict[int, int], notices: _Notices) -> list[Position]:
        closed: list[Position] = []
        for position in repo_portfolio.open_positions(self._db):
            try:
                result = self._check_protective_exit(reader, position, close_times.get(_stored_id(position.id)), notices)
            except Exception:  # noqa: BLE001 - one broken position must not stop the others
                log.exception("protective exit check failed for position %s (%s)", position.id, position.symbol)
                continue
            if result is not None:
                closed.append(result)
        return closed

    def _check_protective_exit(
        self, reader: BarReader, position: Position, close_order_ts: int | None, notices: _Notices
    ) -> Position | None:
        instrument = self._instrument(position.symbol)
        # Only settled bars: a provisional bar may still be replaced and a missing minute may still be inserted
        # behind it. checked_ts therefore never passes a bar whose final values are not known yet.
        bars = reader.settled_bars_after(position.symbol, position.checked_ts, start_ts=position.opened_ts)
        if close_order_ts is not None:
            bars = bars[bars.index < close_order_ts]
        if bars.empty:
            return None
        ts, values = repo_market.bar_arrays(bars)
        flatten_ts = self._flatten_ts(position, instrument)
        stops, targets = self._levels_per_bar(position, ts)
        for index in _exit_candidates(position.side, ts, values, stops, targets, (position.time_stop_ts, flatten_ts)):
            exit_ = execution.exit_on_bar(
                side=position.side,
                bar_ts=int(ts[index]),
                bar_open=float(values[index, 0]),
                bar_high=float(values[index, 1]),
                bar_low=float(values[index, 2]),
                stop=_level(stops[index]),
                target=_level(targets[index]),
                time_stop_ts=position.time_stop_ts,
                flatten_ts=flatten_ts,
            )
            if exit_ is not None:
                fill_price = execution.exit_fill_price(self._costs, instrument, position.side, exit_)
                return self._close(
                    position, instrument, exit_ts=int(ts[index]), fill_price=fill_price, reason=exit_.reason, notices=notices
                )
        position.checked_ts = int(ts[-1])
        repo_portfolio.update_position(self._db, position)
        self._forget_prior_levels(_stored_id(position.id), position.checked_ts, reader.asof_ts)
        return None

    def _levels_per_bar(self, position: Position, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Stop and target in force on each bar opening at ``ts`` (NaN: no level).

        A change through ``update_position`` at time U acts like an order placed at U: it applies to the bars
        that open at or after U. Earlier bars - on a delayed feed they arrive after the change - are judged by
        the levels that were in force before it, so a level set now can never fill on a price from before.
        """
        history = self._prior_levels.get(_stored_id(position.id), [])
        changes = np.array([change_ts for change_ts, _, _ in history], dtype=np.int64)
        period = np.searchsorted(changes, ts, side="right")  # number of changes at or before each bar
        stops = np.array([*(_nan(stop) for _, stop, _ in history), _nan(position.stop_price)], dtype=np.float64)
        targets = np.array([*(_nan(target) for _, _, target in history), _nan(position.target_price)], dtype=np.float64)
        return stops[period], targets[period]

    def _forget_prior_levels(self, position_id: int, checked_ts: int, now_ts: int) -> None:
        """Drop the level changes before which no bar is left to examine (a bar at or after them was checked)."""
        history = self._prior_levels.get(position_id)
        if not history:
            return
        pending = [entry for entry in history if entry[0] > checked_ts]
        if len(pending) == len(history):
            return
        if pending:
            self._prior_levels[position_id] = pending
        else:
            del self._prior_levels[position_id]
        self._persist_prior_levels(now_ts)

    def _load_prior_levels(self) -> dict[int, list[_LevelChange]]:
        """The stored level history of the open positions, without the changes that are no longer needed.

        Entries of closed positions or unreadable entries are dropped. A leftover entry could not do harm anyway:
        it only affects bars that open before its change time, and every bar a position (or a later position
        with a reused id) still examines opens after it.
        """
        stored = repo_ops.get_setting(self._db, PRIOR_LEVELS_KEY, {})
        if not isinstance(stored, dict):
            return {}
        positions = {position.id: position for position in repo_portfolio.open_positions(self._db)}
        history: dict[int, list[_LevelChange]] = {}
        for key, entries in stored.items():
            try:
                position = positions.get(int(key))
                changes = [(int(ts), _optional_level(stop), _optional_level(target)) for ts, stop, target in entries]
            except (TypeError, ValueError):
                log.warning("unreadable level history of position %s ignored", key)
                continue
            if position is None:
                continue
            examined = position.checked_ts
            pending = sorted((c for c in changes if examined is None or c[0] > examined), key=lambda c: c[0])
            if pending:
                history[_stored_id(position.id)] = pending
        return history

    def _store_prior_levels(self, history: dict[int, list[_LevelChange]], now_ts: int) -> None:
        """Write ``history`` (position id -> level changes) under ``PRIOR_LEVELS_KEY``."""
        value = {str(pid): [list(change) for change in changes] for pid, changes in history.items()}
        repo_ops.set_setting(self._db, PRIOR_LEVELS_KEY, value, now_ts)

    def _persist_prior_levels(self, now_ts: int) -> None:
        """Store the current memory after entries were dropped. A failure is only logged: the stored surplus
        entries are harmless (see ``_load_prior_levels``) and must never undo a booking or a check."""
        try:
            self._store_prior_levels(self._prior_levels, now_ts)
        except Exception:  # noqa: BLE001 - a booking or a check that already happened stays valid
            log.exception("level history could not be stored")

    # -- 2. overdue exits without bars ------------------------------------------------------------
    def _overdue_exits(
        self, reader: BarReader, now: datetime, close_times: dict[int, int], notices: _Notices
    ) -> list[Position]:
        closed: list[Position] = []
        for position in repo_portfolio.open_positions(self._db):
            try:
                result = self._check_overdue_exit(reader, position, now, close_times.get(_stored_id(position.id)), notices)
            except Exception:  # noqa: BLE001 - one broken position must not stop the others
                log.exception("overdue exit check failed for position %s (%s)", position.id, position.symbol)
                continue
            if result is not None:
                closed.append(result)
        return closed

    def _check_overdue_exit(
        self, reader: BarReader, position: Position, now: datetime, close_order_ts: int | None, notices: _Notices
    ) -> Position | None:
        """Close a position whose time stop or session flatten is overdue while its venue is closed and no
        bar at or after the deadline has arrived or can still arrive: market exit at the close of the newest
        closed bar, at ``now``.

        The rule never runs ahead of unsettled data. It waits until
        * the deadline minute (and every minute before it) is settled by age (``BarReader.settle_horizon``),
        * the venue traded in no minute after the deadline that is still young - REST may still deliver a bar
          at or after the deadline, on which step 1 exits at the bar's own time and open - and
        * every arrived bar of the position has been examined by step 1 (it hands them out once settled).
        A position whose pending closing order can fill on an arrived bar is left to step 3.
        """
        if close_order_ts is not None and reader.first_bar_at_or_after(position.symbol, close_order_ts) is not None:
            return None
        instrument = self._instrument(position.symbol)
        deadline = _earliest_deadline(position.time_stop_ts, self._flatten_ts(position, instrument))
        if deadline is None:
            return None
        deadline_ts, reason = deadline
        now_ts = to_ts(now)
        if deadline_ts + BAR_SECONDS > reader.settle_horizon(position.symbol):
            return None
        if self._calendar.is_open(instrument.exchange, now):
            return None
        if self._unsettled_trading(reader, instrument, deadline_ts, now_ts):
            return None
        bar = reader.last_bar(position.symbol)
        if bar is None:
            return None
        examined_until = position.opened_ts - 1 if position.checked_ts is None else position.checked_ts
        if bar.ts > examined_until:
            return None
        exit_ = execution.Exit(bar.close, reason, True)
        fill_price = execution.exit_fill_price(self._costs, instrument, position.side, exit_)
        return self._close(position, instrument, exit_ts=now_ts, fill_price=fill_price, reason=reason, notices=notices)

    # -- 3. pending closing orders ------------------------------------------------------------------
    def _fill_closing_orders(self, reader: BarReader, notices: _Notices) -> list[Position]:
        closed: list[Position] = []
        for order in repo_portfolio.pending_orders(self._db):
            if order.intent != "close":
                continue
            try:
                result = self._fill_closing_order(reader, order, notices)
            except Exception:  # noqa: BLE001 - one broken order must not stop the others
                log.exception("closing order %s (%s) could not be processed", order.id, order.symbol)
                continue
            if result is not None:
                closed.append(result)
        return closed

    def _fill_closing_order(self, reader: BarReader, order: Order, notices: _Notices) -> Position | None:
        position = None if order.position_id is None else repo_portfolio.get_position(self._db, order.position_id)
        if position is None or position.status != "open":
            self._finish(order, "cancelled", "Position bereits geschlossen", notices)
            return None
        bar = _closing_fill_bar(reader, position, order.created_ts)
        if bar is None:
            return None
        bar_ts, bar_open = bar
        if order.expire_ts is not None and bar_ts > order.expire_ts:
            self._expire(order, notices)
            return None
        instrument = self._instrument(order.symbol)
        fill_price = _market_close_price(self._costs, instrument, position.side, bar_open)
        return self._close(
            position, instrument, exit_ts=bar_ts, fill_price=fill_price, reason=order.origin, notices=notices, order=order
        )

    # -- 4. pending opening orders ------------------------------------------------------------------
    def _fill_opening_orders(self, reader: BarReader, notices: _Notices) -> None:
        for order in repo_portfolio.pending_orders(self._db):
            if order.intent != "open":
                continue
            try:
                self._fill_opening_order(reader, order, notices)
            except Exception:  # noqa: BLE001 - one broken order must not stop the others
                log.exception("opening order %s (%s) could not be processed", order.id, order.symbol)

    def _fill_opening_order(self, reader: BarReader, order: Order, notices: _Notices) -> None:
        # The fill bar is used only once it is final and no earlier traded minute can still arrive (that
        # minute would be the real fill bar); until then the order stays pending.
        bar = reader.first_settled_bar_at_or_after(order.symbol, order.created_ts)
        if bar is None:
            return
        if order.expire_ts is not None and bar.ts > order.expire_ts:
            self._expire(order, notices)
            return
        instrument = self._instrument(order.symbol)
        pid = order.portfolio_id
        portfolio = self._portfolio(self._keys[pid])
        base = portfolio.base_currency
        if repo_portfolio.open_position_for_symbol(self._db, pid, order.symbol) is not None:
            self._reject(order, f"für {instrument.name} besteht bereits eine Position", notices)
            return
        if order.ref_price is not None and execution.drift_exceeded(order.ref_price, bar.open, order.max_drift):
            self._reject(
                order,
                f"Kurs zu weit gelaufen (Eröffnung {_price(bar.open)} statt {_price(order.ref_price)}, "
                f"erlaubt ± {_price(order.max_drift or 0.0)})",
                notices,
            )
            return
        fill_price = execution.entry_fill_price(self._costs, instrument, order.side, bar.open)
        fx = self._fx_at(instrument.currency, base, bar.ts)
        if fx is None:
            self._reject(order, f"kein Wechselkurs {instrument.currency} → {base} bekannt", notices)
            return
        if order.qty is not None:
            qty = sizing.round_qty(instrument, order.qty)
        else:
            qty = sizing.qty_from_notional(instrument, order.notional_base or 0.0, fill_price, fx)
        if qty <= 0:
            self._reject(order, "Betrag reicht nicht für eine Einheit", notices)
            return
        value = sizing.value_base(qty, fill_price, instrument.price_scale, fx)
        buying_power = self._valuation(portfolio, reader).buying_power(self._max_leverage())
        if value > buying_power + _MONEY_EPS:
            self._reject(
                order,
                f"Kaufkraft reicht bei Ausführung nicht (Auftragswert {_money(value, base)}, "
                f"Kaufkraft {_money(buying_power, base)})",
                notices,
            )
            return
        fee = self._costs.order_fee(value, instrument)
        stop, target = _protective_levels(order, fill_price)
        position = Position(
            portfolio_id=pid,
            symbol=order.symbol,
            side=order.side,
            qty=qty,
            entry_price=fill_price,
            entry_fx=fx,
            opened_ts=bar.ts,
            stop_price=stop,
            target_price=target,
            time_stop_ts=self._position_time_stop(order, instrument, bar.ts),
            flatten_at_close=order.flatten_at_close,
            signal_id=order.signal_id,
            checked_ts=None,  # the next pass examines the entry bar itself (it happens after the fill)
            fees_base=fee,
        )
        with self._db.write():
            repo_portfolio.insert_position(self._db, position)
            repo_portfolio.adjust_cash(self._db, pid, cash_delta=-position.sign * value, fee=fee)
            order.status = "filled"
            order.position_id = position.id
            order.filled_ts, order.filled_qty, order.fill_price, order.fill_fx, order.fee_base = (
                bar.ts,
                qty,
                fill_price,
                fx,
                fee,
            )
            repo_portfolio.update_order(self._db, order)
        notices.order(self._order_out(order))
        notices.activity(
            "info",
            f"Depot {portfolio.name}: {_OPEN_ACTION_DE[order.side]} {_qty(qty)} {instrument.name} zu {_price(fill_price)} "
            f"ausgeführt (Wert {_money(value, base)}, Gebühr {_money(fee, base)}).",
            "position",
            position.id,
        )

    def _position_time_stop(self, order: Order, instrument: Instrument, bar_ts: int) -> int | None:
        """Time stop of the position that ``order`` opens on the bar opening at ``bar_ts``.

        A signal order counts the signal's holding period from the ENTRY BAR, exactly like the signal journal
        (``tracking`` sets ``expires_ts`` from the entry bar): when the first bar after the issue opens late
        (halt, sparse minutes) the time stop moves with it instead of cutting the trade short or closing it on
        its entry bar. Every other order keeps its own ``time_stop_ts``, as does a signal order whose signal or
        trading hours are unknown.
        """
        if order.origin != "signal" or order.signal_id is None:
            return order.time_stop_ts
        signal = repo_signals.get_signal(self._db, order.signal_id)
        if signal is None:
            return order.time_stop_ts
        try:
            return to_ts(self._calendar.add_trading_minutes(instrument.exchange, from_ts(bar_ts), signal.horizon_min))
        except KeyError:
            return order.time_stop_ts

    # -- 5. expiry ----------------------------------------------------------------------------------
    def _expire_orders(self, reader: BarReader, notices: _Notices) -> None:
        for order in repo_portfolio.pending_orders(self._db):
            try:
                if self._expiry_due(reader, order):
                    self._expire(order, notices)
            except Exception:  # noqa: BLE001 - one broken order must not stop the others
                log.exception("order %s (%s) could not be expired", order.id, order.symbol)

    def _expiry_due(self, reader: BarReader, order: Order) -> bool:
        """True when a pending order can be cancelled by the clock: ``expire_ts`` has passed, no fill bar has
        arrived inside the lifetime and none can still arrive - the venue traded in no minute up to
        ``expire_ts`` that is not settled by age yet.

        A fill bar that arrived in time is left to steps 3 and 4: they fill the order on it once it is settled.
        The bar-time rule of those steps (a fill bar opening after ``expire_ts`` expires the order) is the same
        test the signal tracker applies to the entry deadline.
        """
        if order.expire_ts is None or reader.asof_ts <= order.expire_ts:
            return False
        bar = reader.first_bar_at_or_after(order.symbol, order.created_ts)
        if bar is not None and bar.ts <= order.expire_ts:
            return False
        instrument = self._instrument(order.symbol)
        return not self._unsettled_trading(reader, instrument, order.created_ts, order.expire_ts + 1)

    def _unsettled_trading(self, reader: BarReader, instrument: Instrument, start_ts: int, end_ts: int) -> bool:
        """True if the venue of ``instrument`` traded during a minute in [start_ts, end_ts) that is not settled by
        age yet (``BarReader.settle_horizon``): a bar for that minute may still arrive or be corrected. Nights,
        weekends and breaks are no trading; a venue the calendar does not know counts as trading."""
        horizon = reader.settle_horizon(instrument.symbol)
        first = max(_ceil_minute(start_ts), _ceil_minute(horizon - BAR_SECONDS + 1))
        if first >= end_ts:
            return False
        try:
            return bool(self._calendar.segments(instrument.exchange, from_ts(first), from_ts(end_ts)))
        except KeyError:
            return True

    # -- 6. equity snapshots ------------------------------------------------------------------------
    def _snapshots(self, reader: BarReader, now_ts: int, notices: _Notices) -> None:
        """Snapshot every portfolio that changed in this pass (at ``now_ts``) and every portfolio whose newest
        snapshot is older than the current 5-minute slot (at the slot time, but never before the start of the
        comparison period: a portfolio created in the middle of a slot gets its first point at ``reset_ts``)."""
        slot = floor_ts(now_ts, SNAPSHOT_INTERVAL_S)
        for key, pid in self._ids.items():
            try:
                changed = key in notices.changed
                last = repo_portfolio.last_equity_point(self._db, pid)
                if not changed and last is not None and last.ts >= slot:
                    continue
                portfolio = self._portfolio(key)
                ts = now_ts if changed else max(slot, portfolio.reset_ts)
                self._save_snapshot(self._valuation(portfolio, reader), ts)
                notices.changed.add(key)
            except Exception:  # noqa: BLE001 - a failed snapshot must not stop the cycle
                log.exception("equity snapshot of portfolio %s failed", key)

    # -- closing, rejecting, expiring ---------------------------------------------------------------
    def _close(
        self,
        position: Position,
        instrument: Instrument,
        *,
        exit_ts: int,
        fill_price: float,
        reason: OrderOrigin,
        notices: _Notices,
        order: Order | None = None,
    ) -> Position:
        """Close ``position`` at ``fill_price`` (slippage already applied) in one transaction: book cash and
        fee, set the exit fields, mark ``order`` (a pending closing order) filled or record a new filled
        closing order with origin ``reason``, and cancel any other pending closing order of the position."""
        pid = position.portfolio_id
        base = self._bases[pid]
        fx = self._fx_at(instrument.currency, base, exit_ts)
        if fx is None:
            log.warning("no %s->%s rate at %s: closing %s with its entry rate", instrument.currency, base, exit_ts, position.symbol)
            fx = position.entry_fx
        value = sizing.value_base(position.qty, fill_price, instrument.price_scale, fx)
        fee = self._costs.order_fee(value, instrument)
        result = _price_result(position, instrument, fill_price, fx)
        # LONG: the exit value comes back; SHORT: the proceeds booked at the entry go out again together with
        # the loss (or come back reduced by the gain) converted at the exit rate - see the module docstring.
        cash_delta = position.sign * sizing.value_base(
            position.qty, position.entry_price, instrument.price_scale, position.entry_fx
        ) + result
        position.status = "closed"
        position.closed_ts, position.exit_price, position.exit_fx, position.exit_reason = exit_ts, fill_price, fx, reason
        position.fees_base += fee
        position.realized_pnl = result - position.fees_base
        closing = order if order is not None else Order(
            portfolio_id=pid,
            created_ts=exit_ts,
            symbol=position.symbol,
            intent="close",
            side=position.side,
            origin=reason,
            qty=position.qty,
            position_id=position.id,
        )
        closing.status = "filled"
        closing.filled_ts, closing.filled_qty, closing.fill_price, closing.fill_fx, closing.fee_base = (
            exit_ts,
            position.qty,
            fill_price,
            fx,
            fee,
        )
        obsolete = [
            o
            for o in repo_portfolio.pending_orders(self._db, pid)
            if o.intent == "close" and o.position_id == position.id and o.id != closing.id
        ]
        with self._db.write():
            repo_portfolio.update_position(self._db, position)
            repo_portfolio.adjust_cash(self._db, pid, cash_delta=cash_delta, fee=fee)
            if closing.id is None:
                repo_portfolio.insert_order(self._db, closing)
            else:
                repo_portfolio.update_order(self._db, closing)
            for other in obsolete:
                other.status, other.status_reason = "cancelled", "Position bereits geschlossen"
                repo_portfolio.update_order(self._db, other)
        if self._prior_levels.pop(_stored_id(position.id), None) is not None:
            self._persist_prior_levels(self._clock.ts())
        for changed in (closing, *obsolete):
            notices.order(self._order_out(changed))
        notices.activity(
            "info",
            f"Depot {self._names[self._keys[pid]]}: {instrument.name} {_SIDE_DE[position.side]} geschlossen "
            f"({_EXIT_REASON_DE.get(reason, reason)}) zu {_price(fill_price)} – Ergebnis {_money(position.realized_pnl, base)}.",
            "position",
            position.id,
        )
        return position

    def _reject(self, order: Order, reason: str, notices: _Notices) -> None:
        self._finish(order, "rejected", reason, notices)
        notices.activity(
            "warning",
            f"Depot {self._names[self._keys[order.portfolio_id]]}: Order #{order.id} für {self._display_name(order.symbol)} "
            f"abgelehnt – {reason}.",
            "order",
            order.id,
        )

    def _expire(self, order: Order, notices: _Notices) -> None:
        self._finish(order, "cancelled", "abgelaufen", notices)
        notices.activity(
            "info",
            f"Depot {self._names[self._keys[order.portfolio_id]]}: Order #{order.id} für {self._display_name(order.symbol)} "
            "abgelaufen, ohne ausgeführt zu werden.",
            "order",
            order.id,
        )

    def _finish(self, order: Order, status: OrderStatus, reason: str, notices: _Notices) -> None:
        order.status, order.status_reason = status, reason
        repo_portfolio.update_order(self._db, order)
        notices.order(self._order_out(order))

    # ==========================================================================================
    # Reports
    # ==========================================================================================
    def summary(self, portfolio_key: str) -> PortfolioSummaryOut:
        with self._lock:
            now_ts = self._clock.ts()
            valuation = self._valuation(self._portfolio(portfolio_key), self._reader(now_ts))
            return self._summary(valuation, now_ts)

    def detail(self, portfolio_key: str) -> PortfolioDetailOut:
        with self._lock:
            now_ts = self._clock.ts()
            portfolio = self._portfolio(portfolio_key)
            valuation = self._valuation(portfolio, self._reader(now_ts))
            pending = repo_portfolio.pending_orders(self._db, self._ids[portfolio.key])
            return PortfolioDetailOut(
                summary=self._summary(valuation, now_ts),
                positions=[self._position_out(h.position, h.instrument, h) for h in valuation.holdings],
                pending_orders=[self._order_out(order) for order in reversed(pending)],
            )

    def trades(self, portfolio_key: str, limit: int = 200) -> list[PositionOut]:
        with self._lock:
            portfolio = self._portfolio(portfolio_key)
            closed = repo_portfolio.closed_positions(self._db, self._ids[portfolio.key], limit=limit)
            return [self._position_out(position, self._instrument(position.symbol)) for position in closed]

    def orders(self, portfolio_key: str, limit: int = 200) -> list[OrderOut]:
        with self._lock:
            portfolio = self._portfolio(portfolio_key)
            return [self._order_out(o) for o in repo_portfolio.list_orders(self._db, self._ids[portfolio.key], limit=limit)]

    def equity_curves(self, since_ts: int | None = None, max_points: int = 600) -> EquityCurvesOut:
        with self._lock:
            now_ts = self._clock.ts()
            max_points = max(2, max_points)
            reference = self._portfolio("ai")
            reset_ts = max(self._portfolio(key).reset_ts for key in PORTFOLIO_KEYS)
            start_ts = reset_ts if since_ts is None else max(int(since_ts), reset_ts)
            curves = {
                key: [
                    (point.ts, point.equity)
                    for point in repo_portfolio.equity_series(self._db, pid, since_ts=start_ts, max_points=max_points)
                ]
                for key, pid in self._ids.items()
            }
            benchmarks: dict[str, list[tuple[int, float]]] = {}
            for symbol in self._runtime_settings.get().benchmarks:
                instrument = self._instruments.get(symbol)
                if instrument is None:
                    continue
                try:
                    curve = self._benchmark_curve(instrument, reference, now_ts, start_ts, max_points)
                except Exception:  # noqa: BLE001 - a broken benchmark must not hide the portfolio curves
                    log.exception("benchmark curve for %s failed", symbol)
                    continue
                if curve:
                    benchmarks[symbol] = curve
            return EquityCurvesOut(
                since_ts=start_ts,
                base_currency=reference.base_currency,
                start_cash=reference.start_cash,
                ai=curves["ai"],
                me=curves["me"],
                benchmarks=benchmarks,
            )

    def reset(self, start_capital: float | None = None) -> None:
        with self._lock:
            capital = self._settings.start_capital if start_capital is None else float(start_capital)
            if not _positive(capital):
                raise OrderRejected("Das Startkapital muss eine positive Zahl sein.")
            now_ts = self._clock.ts()
            repo_portfolio.reset_portfolios(self._db, start_cash=capital, now_ts=now_ts)
            if self._prior_levels:  # the positions are gone (and their ids are reused after the reset)
                self._prior_levels.clear()
                self._persist_prior_levels(now_ts)
            self._drawdowns.clear()  # the snapshots of the old period are gone
            notices = _Notices()
            reader = self._reader(now_ts)
            for key in PORTFOLIO_KEYS:
                self._save_snapshot(self._valuation(self._portfolio(key), reader), now_ts)
                notices.changed.add(key)
            base = self._bases[self._ids["ai"]]
            notices.activity(
                "info", f"Neuer Vergleichszeitraum: Beide Depots starten mit {_money(capital, base)} Startkapital."
            )
            self._flush(notices, now_ts)

    # -- report helpers -----------------------------------------------------------------------------
    def _summary(self, valuation: _Valuation, now_ts: int) -> PortfolioSummaryOut:
        portfolio = valuation.portfolio
        pid = self._ids[portfolio.key]
        equity = valuation.equity
        closed = repo_portfolio.closed_positions(self._db, pid, limit=_ALL_ROWS)
        stats = metrics.trade_stats(closed)
        series = repo_portfolio.equity_series(self._db, pid, since_ts=portfolio.reset_ts, max_points=_METRICS_POINTS)
        points = [(point.ts, point.equity) for point in series]
        if not points or points[-1][0] < now_ts:
            points.append((now_ts, equity))
        tz = self._settings.timezone
        reference = repo_portfolio.last_equity_point(
            self._db, pid, before_ts=metrics.local_midnight_ts(now_ts, tz)
        )
        before_today = [] if reference is None else [(reference.ts, reference.equity)]
        day_pnl, day_pct = metrics.day_pnl(equity, before_today, tz, portfolio.start_cash, now_ts=now_ts)
        gross = valuation.gross_exposure
        return PortfolioSummaryOut(
            key=portfolio.key,
            name=portfolio.name,
            base_currency=portfolio.base_currency,
            start_cash=portfolio.start_cash,
            cash=portfolio.cash,
            equity=equity,
            buying_power=valuation.buying_power(self._max_leverage()),
            long_exposure=valuation.long_exposure,
            short_exposure=valuation.short_exposure,
            gross_leverage=gross / equity if equity > 0 else 0.0,
            unrealized_pnl=sum(h.unrealized_pnl for h in valuation.holdings),
            realized_pnl=sum(p.realized_pnl or 0.0 for p in closed),
            fees_paid=portfolio.fees_paid,
            total_return_pct=(equity / portfolio.start_cash - 1.0) * 100.0 if portfolio.start_cash > 0 else 0.0,
            day_pnl=day_pnl,
            day_return_pct=day_pct,
            open_positions=len(valuation.holdings),
            pending_orders=len(repo_portfolio.pending_orders(self._db, pid)),
            closed_trades=stats.closed_trades,
            wins=stats.wins,
            win_rate=stats.win_rate,
            profit_factor=stats.profit_factor,
            avg_win=stats.avg_win,
            avg_loss=stats.avg_loss,
            max_drawdown_pct=self._max_drawdown_pct(portfolio, pid, equity, now_ts),
            sharpe=metrics.sharpe_daily(points, tz),
            reset_ts=portfolio.reset_ts,
            as_of_ts=now_ts,
        )

    def _max_drawdown_pct(self, portfolio: Portfolio, pid: int, equity: float, now_ts: int) -> float | None:
        """Max drawdown over EVERY equity snapshot of the comparison period plus the current equity (a thinned
        series would lose troughs that last only a few snapshots).

        Snapshots are only appended - the newest one may still be replaced at the same time - so the running
        peak and drawdown of all snapshots before the newest one are kept per portfolio and each call reads
        only the newest snapshot and the ones written since.
        """
        cache = self._drawdowns.get(pid)
        if cache is None or cache.reset_ts != portfolio.reset_ts:
            cache = _DrawdownCache(portfolio.reset_ts, portfolio.reset_ts, metrics.Drawdown())
            self._drawdowns[pid] = cache
        fresh = repo_portfolio.equity_series(self._db, pid, since_ts=cache.folded_until, max_points=_ALL_ROWS)
        if len(fresh) > 1:
            cache.state = metrics.extend_drawdown(cache.state, [point.equity for point in fresh[:-1]])
            cache.folded_until = fresh[-1].ts
        tail = [point.equity for point in fresh[-1:]]
        if not fresh or fresh[-1].ts < now_ts:
            tail.append(equity)
        return metrics.extend_drawdown(cache.state, tail).max_drawdown_pct

    def _benchmark_curve(
        self, instrument: Instrument, portfolio: Portfolio, now_ts: int, start_ts: int, max_points: int
    ) -> list[tuple[int, float]]:
        """Buy-and-hold of ``instrument`` bought for the start capital at the reset, in base currency.

        Points: the reset itself, the close of every session that ended after the reset (at the session's
        end), and the latest price; each converted with the FX rate known at that point.
        """
        symbol, currency, base = instrument.symbol, instrument.currency, portfolio.base_currency
        reset_ts = portfolio.reset_ts
        at_reset = self._reader(reset_ts)
        ref_price = at_reset.last_price(symbol)
        if ref_price is None:
            daily_ref = at_reset.daily(symbol, limit=1)
            ref_price = None if daily_ref.empty else float(daily_ref["close"].iloc[-1])
        ref_fx = self._fx_at(currency, base, reset_ts)
        if ref_price is None or ref_fx is None or not _positive(ref_price):
            return []
        units = portfolio.start_cash / (ref_price * ref_fx)  # price_scale cancels out of the ratio
        points: list[tuple[int, float]] = [(reset_ts, portfolio.start_cash)]
        reader = self._reader(now_ts)
        closes = self._session_closes(instrument.exchange, reset_ts, now_ts)
        daily = reader.daily(symbol, limit=(now_ts - reset_ts) // 86_400 + 7)
        for day, close in zip(daily.index.tolist(), daily["close"].tolist(), strict=True):
            close_ts = closes.get(day) or _utc_day_end(day)
            if close_ts <= reset_ts or close_ts > now_ts:  # a completed session cannot close in the future
                continue
            fx = self._fx_at(currency, base, close_ts)
            if fx is not None:
                points.append((close_ts, close * fx * units))
        last = self._last_price(reader, symbol)
        fx_now = self._fx.rate(currency, base)
        if last is not None and fx_now is not None and last[1] > points[-1][0]:
            points.append((last[1], last[0] * fx_now * units))
        return _thin([point for point in points if point[0] >= start_ts], max_points)

    def _session_closes(self, exchange: str, from_ts_: int, to_ts_: int) -> dict[str, int]:
        """Session date ('YYYY-MM-DD') -> end of its last regular segment, for the sessions around the range.

        Segments are cut at every UTC midnight and each piece counts for the trading date of its start (as in
        ``execution.session_flatten_ts``): a calendar may report a 24/7 venue as ONE segment spanning several
        days, whose end must not become the close of its first day.
        """
        closes: dict[str, int] = {}
        start, end = from_ts(from_ts_) - timedelta(days=2), from_ts(to_ts_) + timedelta(days=2)
        for segment in self._calendar.segments(exchange, start, end):
            for piece_start, piece_end in _split_at_utc_midnight(to_ts(segment.start), to_ts(segment.end)):
                closes[self._calendar.session_date(exchange, from_ts(piece_start)).isoformat()] = piece_end
        return closes

    # ==========================================================================================
    # Lookups and valuation
    # ==========================================================================================
    def _reader(self, asof_ts: int) -> BarReader:
        return BarReader(self._db, self._instruments, self._calendar, asof_ts)

    def _max_leverage(self) -> float:
        return self._runtime_settings.get().max_gross_leverage

    def _portfolio(self, key: str) -> Portfolio:
        """The stored portfolio; ``KeyError`` for keys other than 'ai' and 'me'."""
        portfolio = repo_portfolio.get_portfolio(self._db, key) if key in self._ids else None
        if portfolio is None:
            raise KeyError(f"unknown portfolio {key!r}")
        return portfolio

    def _portfolio_for_order(self, key: str) -> Portfolio:
        try:
            return self._portfolio(key)
        except KeyError:
            raise OrderRejected(f"Unbekanntes Depot '{key}'. Erlaubt sind 'ai' (KI) und 'me' (Ich).") from None

    def _open_position_of(self, portfolio: Portfolio, position_id: int) -> Position:
        position = repo_portfolio.get_position(self._db, position_id)
        if position is None or position.portfolio_id != self._ids[portfolio.key] or position.status != "open":
            raise OrderRejected(f"Die Position #{position_id} ist in diesem Depot nicht offen.")
        return position

    def _instrument(self, symbol: str) -> Instrument:
        """Instrument from the shared book, falling back to the table (the book may not be reloaded yet)."""
        instrument = self._instruments.get(symbol) or repo_market.get_instrument(self._db, symbol)
        if instrument is None:
            raise LookupError(f"unknown instrument {symbol}")
        return instrument

    def _display_name(self, symbol: str) -> str:
        instrument = self._instruments.get(symbol)
        return symbol if instrument is None else instrument.name

    def _tradable_instrument(self, symbol: str) -> Instrument:
        symbol = symbol.strip()
        instrument = self._instruments.get(symbol) or self._instruments.get(symbol.upper())
        if instrument is None:
            raise OrderRejected(f"Das Symbol {symbol} ist unbekannt. Füge es zuerst der Watchlist hinzu.")
        if not instrument.active:
            raise OrderRejected(f"{instrument.name} ist deaktiviert. Aktiviere das Symbol in der Watchlist, um es zu handeln.")
        if not sizing.is_tradable_class(instrument):
            raise OrderRejected(
                f"{instrument.name} ist nicht handelbar: Devisenpaare und Futures dienen nur als Marktinformation."
            )
        if not instrument.tradable:
            raise OrderRejected(
                f"{instrument.name} ist als nicht handelbar markiert. Ändere das in der Watchlist, um es zu handeln."
            )
        return instrument

    def _reject_symbol_conflict(self, portfolio: Portfolio, instrument: Instrument) -> None:
        pid = self._ids[portfolio.key]
        existing = repo_portfolio.open_position_for_symbol(self._db, pid, instrument.symbol)
        if existing is not None:
            raise OrderRejected(
                f"Für {instrument.name} besteht bereits eine {_SIDE_DE[existing.side]}-Position. Schließe sie zuerst."
            )
        if any(o.intent == "open" and o.symbol == instrument.symbol for o in repo_portfolio.pending_orders(self._db, pid)):
            raise OrderRejected(
                f"Für {instrument.name} liegt bereits eine offene Order vor. "
                "Storniere sie zuerst oder warte, bis sie ausgeführt ist."
            )

    def _last_price(self, reader: BarReader, symbol: str) -> tuple[float, int] | None:
        """Last known price and its time: the latest quote, else the close of the newest closed 1-minute bar."""
        quote = self._data.quote(symbol)
        if quote is not None and _positive(quote.price):
            return float(quote.price), int(quote.ts)
        bar = reader.last_bar(symbol)
        return None if bar is None else (bar.close, bar.ts + BAR_SECONDS)

    def _fx_at(self, currency: str, base: str, ts: int) -> float | None:
        """Rate known at ``ts``, else the latest rate (a fill must not fail only because FX history is missing)."""
        rate = self._fx.rate(currency, base, ts)
        return self._fx.rate(currency, base) if rate is None else rate

    def _flatten_ts(self, position: Position, instrument: Instrument) -> int | None:
        if not position.flatten_at_close:
            return None
        return execution.session_flatten_ts(
            self._calendar, instrument.exchange, int(position.opened_ts), ENGINE_PARAMS.flatten_before_close_min
        )

    def _holding(self, position: Position, instrument: Instrument, base: str, reader: BarReader) -> _Holding:
        last = self._last_price(reader, position.symbol)
        mark, mark_ts = (position.entry_price, None) if last is None else last
        fx = self._fx.rate(instrument.currency, base)
        return _Holding(position, instrument, mark, mark_ts, position.entry_fx if fx is None else fx)

    def _valuation(self, portfolio: Portfolio, reader: BarReader) -> _Valuation:
        holdings = [
            self._holding(position, self._instrument(position.symbol), portfolio.base_currency, reader)
            for position in repo_portfolio.open_positions(self._db, self._ids[portfolio.key])
        ]
        return _Valuation(portfolio, holdings)

    def _save_snapshot(self, valuation: _Valuation, ts: int) -> None:
        repo_portfolio.save_equity_point(
            self._db,
            EquityPoint(
                portfolio_id=self._ids[valuation.portfolio.key],
                ts=ts,
                equity=valuation.equity,
                cash=valuation.portfolio.cash,
                long_exposure=valuation.long_exposure,
                short_exposure=valuation.short_exposure,
                open_positions=len(valuation.holdings),
            ),
        )

    # ==========================================================================================
    # Presentation and notifications
    # ==========================================================================================
    def _order_out(self, order: Order) -> OrderOut:
        return OrderOut.model_validate(
            {**asdict(order), "portfolio_key": self._keys[order.portfolio_id], "name": self._display_name(order.symbol)}
        )

    def _position_out(self, position: Position, instrument: Instrument, holding: _Holding | None = None) -> PositionOut:
        values = {
            **asdict(position),
            "portfolio_key": self._keys[position.portfolio_id],
            "name": instrument.name,
            "currency": instrument.currency,
        }
        entry_value = sizing.value_base(position.qty, position.entry_price, instrument.price_scale, position.entry_fx)
        if holding is not None:
            values.update(
                last_price=holding.mark,
                last_ts=holding.mark_ts,
                exposure_base=holding.exposure,
                unrealized_pnl=holding.unrealized_pnl,
                unrealized_pct=holding.unrealized_pnl / entry_value * 100.0 if entry_value > 0 else None,
            )
        if position.status == "closed" and position.realized_pnl is not None and entry_value > 0:
            values["realized_pct"] = position.realized_pnl / entry_value * 100.0
        return PositionOut.model_validate(values)

    def _announce(self, order: Order, now_ts: int) -> OrderOut:
        """Present an order that was just written and announce it (order and portfolio events)."""
        out = self._order_out(order)
        notices = _Notices()
        notices.order(out)
        self._flush(notices, now_ts)
        return out

    def _flush(self, notices: _Notices, now_ts: int) -> None:
        """Write the activity entries and announce order and portfolio changes (after the commits)."""
        for level, message, ref_type, ref_id in notices.activities:
            try:
                repo_ops.log_activity(
                    self._db,
                    ts=now_ts,
                    level=level,
                    category="sim",
                    message=message,
                    ref_type=ref_type,
                    ref_id=ref_id,
                    bus=self._bus,
                )
            except Exception:  # noqa: BLE001 - the activity log must never undo a booking
                log.exception("activity entry could not be written: %s", message)
        for out in notices.orders:
            self._bus.publish(TOPIC_ORDER, out.model_dump())
        for key in PORTFOLIO_KEYS:
            if key in notices.changed:
                self._bus.publish(TOPIC_PORTFOLIO, {"key": key})


# ------------------------------------------------------------------------------------------------
# Pure helpers
# ------------------------------------------------------------------------------------------------
def _stored_id(value: int | None) -> int:
    if value is None:
        raise ValueError("row has no id: only stored rows can be referenced")
    return value


def _positive(value: float) -> bool:
    return math.isfinite(value) and value > 0


def _ceil_minute(ts: int) -> int:
    """``ts`` rounded up to a full minute."""
    return -(-int(ts) // BAR_SECONDS) * BAR_SECONDS


def _price_result(position: Position, instrument: Instrument, price: float, fx: float) -> float:
    """Result of ``position`` at the quoted ``price`` and the rate ``fx`` before fees, in base currency.

    LONG: qty * price_scale * (price * fx - entry * entry_fx) - bought with base currency, so the whole position
    carries the currency risk. SHORT: -qty * price_scale * (price - entry) * fx - the sale proceeds stay in the
    instrument currency, so only the result is converted (at the current rate), never the notional.
    """
    scale = instrument.price_scale
    if position.side == "LONG":
        return position.qty * scale * (price * fx - position.entry_price * position.entry_fx)
    return -position.qty * scale * (price - position.entry_price) * fx


def _closing_fill_bar(reader: BarReader, position: Position, created_ts: int) -> tuple[int, float] | None:
    """``(open time, open)`` of the bar a closing order placed at ``created_ts`` fills on, else None.

    It is the first bar with an open time >= ``created_ts`` in the settled walk over the position's unexamined
    bars - the walk of step 1 - so it is handed out only once it is final and every earlier bar could be
    examined for a stop or target first (a stop hit inside a minute that is inserted late wins over the close).
    """
    bars = reader.settled_bars_after(position.symbol, position.checked_ts, start_ts=position.opened_ts)
    ts = bars.index.to_numpy(dtype=np.int64)
    index = int(np.searchsorted(ts, int(created_ts), side="left"))
    if index >= ts.size:
        return None
    return int(ts[index]), float(bars["open"].to_numpy(dtype=np.float64)[index])


def _check_levels(
    instrument: Instrument, side: Direction, price: float, stop: float | None, target: float | None
) -> None:
    """Protective levels must lie on the correct side of the last known price (LONG: stop < price < target)."""
    for label, level in (("Der Stopp", stop), ("Das Ziel", target)):
        if level is not None and not _positive(level):
            raise OrderRejected(f"{label} muss ein positiver Kurs sein.")
    name, now = instrument.name, _price(price)
    if side == "LONG":
        if stop is not None and stop >= price:
            raise OrderRejected(
                f"Der Stopp ({_price(stop)}) muss bei einer Long-Position unter dem aktuellen Kurs von {name} ({now}) liegen."
            )
        if target is not None and target <= price:
            raise OrderRejected(
                f"Das Ziel ({_price(target)}) muss bei einer Long-Position über dem aktuellen Kurs von {name} ({now}) liegen."
            )
    else:
        if stop is not None and stop <= price:
            raise OrderRejected(
                f"Der Stopp ({_price(stop)}) muss bei einer Short-Position über dem aktuellen Kurs von {name} ({now}) liegen."
            )
        if target is not None and target >= price:
            raise OrderRejected(
                f"Das Ziel ({_price(target)}) muss bei einer Short-Position unter dem aktuellen Kurs von {name} ({now}) liegen."
            )


def _order_size(
    instrument: Instrument, req: OrderRequest, price: float, fx: float, base: str
) -> tuple[float | None, float]:
    """``(qty, estimated value in base currency)`` of a manual order; qty is None for an amount order."""
    if req.qty is not None:
        qty = sizing.round_qty(instrument, req.qty)
        if qty <= 0:
            lot = sizing.lot_size(instrument) or 0.0
            raise OrderRejected(
                f"Die Stückzahl {_qty(req.qty)} ist kleiner als die kleinste handelbare Einheit von "
                f"{instrument.name} ({_qty(lot)})."
            )
        return qty, sizing.value_base(qty, price, instrument.price_scale, fx)
    amount = float(req.amount_base or 0.0)
    if sizing.qty_from_notional(instrument, amount, price, fx) <= 0:
        raise OrderRejected(
            f"Der Betrag von {_money(amount, base)} reicht nicht für eine Einheit von {instrument.name} "
            f"(aktueller Kurs {_price(price)}). Wähle einen höheren Betrag."
        )
    return None, amount


def _protective_levels(order: Order, fill_price: float) -> tuple[float | None, float | None]:
    """Absolute levels of the order, else its distances applied to the fill price (levels <= 0 are dropped)."""
    sign = 1.0 if order.side == "LONG" else -1.0
    stop = order.stop_price
    if stop is None and order.stop_dist is not None:
        stop = fill_price - sign * order.stop_dist
    target = order.target_price
    if target is None and order.target_dist is not None:
        target = fill_price + sign * order.target_dist
    return (stop if stop is not None and stop > 0 else None), (target if target is not None and target > 0 else None)


def _market_close_price(costs: CostParams, instrument: Instrument, side: Direction, quoted: float) -> float:
    """Fill price of a closing MARKET order at ``quoted`` under the slippage rule of ``execution.exit_fill_price``.

    ``execution.Exit.reason`` has no value for a manual close; the reason does not influence the price, only
    ``market`` does, so any reason gives the same result.
    """
    return execution.exit_fill_price(costs, instrument, side, execution.Exit(quoted, "time_stop", True))


def _exit_candidates(
    side: Direction,
    ts: np.ndarray,
    values: np.ndarray,
    stops: np.ndarray,
    targets: np.ndarray,
    deadlines: tuple[int | None, ...],
) -> np.ndarray:
    """Ascending indices of the bars on which ``execution.exit_on_bar`` can act (vectorised pre-filter, so a long
    run of quiet bars after a restart costs no Python loop). ``stops``/``targets`` hold the level of every bar;
    NaN (no level) compares False."""
    open_, high, low = values[:, 0], values[:, 1], values[:, 2]
    mask = np.zeros(ts.size, dtype=bool)
    for deadline in deadlines:
        if deadline is not None:
            mask |= ts >= deadline
    if side == "LONG":
        mask |= (open_ <= stops) | (low <= stops) | (open_ >= targets) | (high >= targets)
    else:
        mask |= (open_ >= stops) | (high >= stops) | (open_ <= targets) | (low <= targets)
    return np.flatnonzero(mask)


def _nan(level: float | None) -> float:
    return math.nan if level is None else float(level)


def _optional_level(value: object) -> float | None:
    """A stored protective level: None or a finite number; anything else raises ``ValueError``/``TypeError``."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"not a price level: {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"not a finite price level: {value!r}")
    return float(value)


def _level(value: float) -> float | None:
    """A protective level from a level array (NaN: none) as plain float for ``execution.exit_on_bar``."""
    return None if math.isnan(value) else float(value)


def _earliest_deadline(time_stop_ts: int | None, flatten_ts: int | None) -> tuple[int, execution.ExitReason] | None:
    if time_stop_ts is not None and (flatten_ts is None or time_stop_ts <= flatten_ts):
        return time_stop_ts, "time_stop"
    if flatten_ts is not None:
        return flatten_ts, "session_close"
    return None


def _split_at_utc_midnight(start_ts: int, end_ts: int) -> list[tuple[int, int]]:
    """``[start_ts, end_ts)`` cut into consecutive pieces that do not cross a UTC midnight (epoch seconds)."""
    pieces: list[tuple[int, int]] = []
    while start_ts < end_ts:
        piece_end = min(end_ts, (start_ts // 86_400 + 1) * 86_400)
        pieces.append((start_ts, piece_end))
        start_ts = piece_end
    return pieces


def _utc_day_end(day: str) -> int:
    """End of a UTC calendar day ('YYYY-MM-DD'): the close of a 24/7 venue's session."""
    start = datetime.combine(date.fromisoformat(day), datetime.min.time(), tzinfo=UTC)
    return to_ts(start + timedelta(days=1))


def _thin(points: list[tuple[int, float]], max_points: int) -> list[tuple[int, float]]:
    """At most ``max_points`` points spread evenly over the list; the first and the last are always kept."""
    if len(points) <= max_points:
        return points
    picks = np.unique(np.linspace(0, len(points) - 1, max_points).round().astype(np.int64))
    return [points[int(i)] for i in picks]


# -- German formatting -------------------------------------------------------------------------------
def _de(value: float, decimals: int) -> str:
    """Number in German notation: 12345.6 -> '12.345,60' (with decimals=2)."""
    return f"{value:_.{decimals}f}".replace(".", ",").replace("_", ".")


def _money(value: float, currency: str) -> str:
    return f"{_de(value, 2)} {currency}"


def _price(value: float) -> str:
    """Quoted price with 2 decimals, small prices with 4 significant digits (at most 8 decimals)."""
    magnitude = abs(value)
    decimals = 2 if magnitude >= 1 or magnitude == 0 else min(8, 3 - math.floor(math.log10(magnitude)))
    return _de(value, decimals)


def _qty(value: float) -> str:
    """Quantity without superfluous decimals: 12 -> '12', 0.0105 -> '0,0105'."""
    text = _de(value, 6)
    return text.rstrip("0").rstrip(",") if "," in text else text
