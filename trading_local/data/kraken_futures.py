"""Kraken Futures public market data: funding rates, open interest and liquidations of the perpetuals - no key.

Checked against the live API on 2026-11-24 (probe scripts in the work notes):

* ``GET /derivatives/api/v3/tickers`` - one answer for all ~200 contracts: ``{"result": "success", "tickers":
  [...], "serverTime": ...}``. A perpetual (``tag`` 'perpetual', symbol ``PF_<BASE>USD``) carries ``markPrice``,
  ``indexPrice``, ``openInterest`` (contracts; one contract of a PF_ perpetual is one unit of the base asset),
  ``fundingRate`` / ``fundingRatePrediction`` (USD per contract and hour) and ``relativeFundingRate`` /
  ``relativeFundingRatePrediction`` (fraction of the price per hour: Kraken charges funding every hour).
  Only the relative values are used here (comparable across coins and over time).
* ``GET /api/charts/v1/analytics/<symbol>/<kind>?interval=<s>&since=<ts>[&to=<ts>]`` - time series with
  ``{"result": {"timestamp": [...], "data": ..., "more": bool}, "errors": []}``, at most 2000 points per answer
  (``more`` asks for the next page from the last timestamp on). Intervals 60, 300, 3600 ... seconds; a
  timestamp is the START of its interval (epoch seconds; ``funding`` answers in milliseconds).
  ``liquidation-volume``: data = list of strings (liquidated contracts in the interval).
  ``open-interest``: data = list of [open, high, low, close] strings (contracts).
* ``GET /derivatives/api/v4/historicalfundingrates?symbol=<symbol>`` - the realised hourly funding of about
  a year: ``rates`` = [{"timestamp": ISO, "fundingRate": .., "relativeFundingRate": ..}, ...], oldest first.

Our crypto instruments trade against EUR on Kraken spot; their perpetual is the USD contract of the same coin
(bitcoin is XBT on Kraken: BTC-EUR -> PF_XBTUSD). Funding, open interest and liquidations describe the
positioning in the coin, so the quote currency does not matter.

Politeness: one token bucket for all requests (2 per second sustained, burst 4) and the circuit breaker of the
spot provider (pause after HTTP 429). Every answer is checked for its shape; nothing here raises anything but
``ProviderError`` / ``RateLimitError``.
"""

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

import httpx

from trading_local.clock import Clock
from trading_local.data.kraken import AsyncRateLimiter, KrakenBreaker
from trading_local.data.provider import ProviderError, RateLimitError
from trading_local.models import Instrument

log = logging.getLogger(__name__)

BASE_URL: Final = "https://futures.kraken.com"
TICKERS_PATH: Final = "/derivatives/api/v3/tickers"
ANALYTICS_PATH: Final = "/api/charts/v1/analytics/{symbol}/{kind}"
FUNDING_HISTORY_PATH: Final = "/derivatives/api/v4/historicalfundingrates"
USER_AGENT: Final = "trading-local/0.1 (market data, public endpoints)"
MAX_PAGES: Final = 20  # safety cap of one analytics request (20 x 2000 points)
_BASE_ALIASES: Final[Mapping[str, str]] = {"BTC": "XBT"}  # our base asset -> Kraken Futures name


def perpetual_symbol(instrument: Instrument) -> str | None:
    """The Kraken Futures perpetual of a crypto instrument ('BTC-EUR' -> 'PF_XBTUSD'); None for other assets."""
    if instrument.asset_class != "crypto":
        return None
    base = instrument.symbol.split("-", 1)[0].upper()
    if not base.isalnum():
        return None
    return f"PF_{_BASE_ALIASES.get(base, base)}USD"


@dataclass(frozen=True, slots=True)
class PerpetualTicker:
    """The parts of one perpetual's ticker that are stored."""

    perpetual: str
    mark_price: float | None
    index_price: float | None
    open_interest: float | None  # contracts (units of the base asset)
    funding_rate: float | None  # relative funding per hour (fraction of the price), current period
    funding_pred: float | None  # relative funding per hour predicted for the next period
    suspended: bool = False


def _number(value: Any) -> float | None:
    """A finite float from a number or a numeric string; None otherwise (booleans included)."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and number > 0 else None


def _non_negative(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and number >= 0 else None


def parse_tickers(body: Any) -> dict[str, PerpetualTicker]:
    """The perpetual tickers of a tickers answer by symbol. Raises ``ProviderError`` for a malformed answer."""
    if not isinstance(body, Mapping) or body.get("result") != "success" or not isinstance(body.get("tickers"), list):
        raise ProviderError(f"tickers: unexpected answer {str(body)[:200]!r}")
    found: dict[str, PerpetualTicker] = {}
    for raw in body["tickers"]:
        if not isinstance(raw, Mapping) or raw.get("tag") != "perpetual":
            continue
        symbol = raw.get("symbol")
        if not isinstance(symbol, str) or not symbol.upper().startswith("PF_"):
            continue
        found[symbol.upper()] = PerpetualTicker(
            perpetual=symbol.upper(),
            mark_price=_positive(raw.get("markPrice")),
            index_price=_positive(raw.get("indexPrice")),
            open_interest=_non_negative(raw.get("openInterest")),
            funding_rate=_number(raw.get("relativeFundingRate")),
            funding_pred=_number(raw.get("relativeFundingRatePrediction")),
            suspended=raw.get("suspended") is True,
        )
    return found


def _seconds(value: Any) -> int | None:
    """Epoch seconds from seconds or milliseconds (the funding series answers in milliseconds)."""
    number = _number(value)
    if number is None or number <= 0:
        return None
    return int(number // 1000) if number > 1e11 else int(number)


def _close(value: Any) -> float | None:
    """The value of one analytics data point: a number, a numeric string or the close of [o, h, l, c]."""
    if isinstance(value, list | tuple):
        return _number(value[-1]) if value else None
    return _number(value)


def parse_series(result: Any) -> tuple[list[tuple[int, float]], bool]:
    """(points as (interval start, value), more pages) of an analytics ``result``; malformed points are left out.

    ``data`` is a list (liquidation volume: values; open interest: [o, h, l, c] -> close) or a mapping with one
    list (funding: {"rate": [[o, h, l, c], ...]}). Raises ``ProviderError`` when the shape is unknown.
    """
    if not isinstance(result, Mapping):
        raise ProviderError("analytics: answer without result")
    stamps, data = result.get("timestamp"), result.get("data")
    if isinstance(data, Mapping):
        lists = [value for value in data.values() if isinstance(value, list)]
        data = lists[0] if len(lists) == 1 else None
    if not isinstance(stamps, list) or not isinstance(data, list) or len(stamps) != len(data):
        raise ProviderError("analytics: timestamps and data do not match")
    points: list[tuple[int, float]] = []
    for stamp, value in zip(stamps, data, strict=True):
        ts, number = _seconds(stamp), _close(value)
        if ts is not None and number is not None:
            points.append((ts, number))
    return points, result.get("more") is True


def parse_funding_history(body: Any) -> list[tuple[int, float]]:
    """(hour, relative funding rate) of a historical funding answer, oldest first."""
    rates = body.get("rates") if isinstance(body, Mapping) else None
    if not isinstance(rates, list):
        raise ProviderError("historical funding: answer without rates")
    points: list[tuple[int, float]] = []
    for raw in rates:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("timestamp"), str):
            continue
        try:
            ts = int(datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00")).timestamp())
        except ValueError:
            continue
        rate = _number(raw.get("relativeFundingRate"))
        if rate is not None:
            points.append((ts, rate))
    points.sort()
    return points


class KrakenFuturesClient:
    """Async client for the public Kraken Futures endpoints above. Create per service; ``close`` at the end."""

    def __init__(
        self,
        clock: Clock,
        *,
        base_url: str = BASE_URL,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: float = 20.0,
        limiter: AsyncRateLimiter | None = None,
        breaker_factory: Callable[[Clock], KrakenBreaker] = KrakenBreaker,
    ) -> None:
        self._base_url = base_url
        self._transport = transport
        self._timeout_s = timeout_s
        self._limiter = limiter or AsyncRateLimiter(rate=2.0, capacity=4.0)
        self._breaker = breaker_factory(clock)
        self._client: httpx.AsyncClient | None = None
        self._closed = False
        self.requests = 0

    async def tickers(self) -> dict[str, PerpetualTicker]:
        return parse_tickers(await self._get(TICKERS_PATH, {}))

    async def series(
        self, perpetual: str, kind: str, *, interval_s: int, since_ts: int, background: bool = False
    ) -> list[tuple[int, float]]:
        """All points of one analytics series from ``since_ts`` on (pages followed up to ``MAX_PAGES``)."""
        path = ANALYTICS_PATH.format(symbol=perpetual, kind=kind)
        points: list[tuple[int, float]] = []
        since = int(since_ts)
        for _ in range(MAX_PAGES):
            body = await self._get(path, {"interval": interval_s, "since": since}, background=background)
            errors = body.get("errors")
            if isinstance(errors, list) and errors:
                raise ProviderError(f"{kind} {perpetual}: {'; '.join(map(str, errors))[:200]}")
            page, more = parse_series(body.get("result"))
            fresh = [point for point in page if not points or point[0] > points[-1][0]]
            points.extend(fresh)
            if not more or not fresh:
                break
            since = fresh[-1][0] + interval_s
        return points

    async def funding_history(self, perpetual: str) -> list[tuple[int, float]]:
        return parse_funding_history(await self._get(FUNDING_HISTORY_PATH, {"symbol": perpetual}, background=True))

    def paused_until(self) -> int | None:
        return self._breaker.paused_until()

    async def close(self) -> None:
        self._closed = True
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout_s,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                transport=self._transport,
            )
        return self._client

    async def _get(self, path: str, params: Mapping[str, Any], *, background: bool = False) -> Mapping[str, Any]:
        """One public GET; returns the JSON object. Raises ``RateLimitError`` or ``ProviderError``."""
        if self._closed:
            raise ProviderError("client is closed")
        self._breaker.check()
        await self._limiter.acquire(background=background)
        self._breaker.check()
        self.requests += 1
        try:
            response = await self._http().get(path, params=dict(params))
        except httpx.HTTPError as exc:
            raise ProviderError(f"{path} request failed: {type(exc).__name__}: {exc}") from exc
        if response.status_code == 429:
            raise RateLimitError(f"rate limited, paused until {self._breaker.trip(f'{path}: HTTP 429')}")
        if response.status_code != 200:
            raise ProviderError(f"{path}: HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderError(f"{path}: no JSON answer") from exc
        if not isinstance(body, Mapping):
            raise ProviderError(f"{path}: answer is not an object")
        if body.get("result") == "error":
            raise ProviderError(f"{path}: {str(body.get('error'))[:200]}")
        self._breaker.record_success()
        return body


__all__ = [
    "BASE_URL",
    "KrakenFuturesClient",
    "PerpetualTicker",
    "parse_funding_history",
    "parse_series",
    "parse_tickers",
    "perpetual_symbol",
]
