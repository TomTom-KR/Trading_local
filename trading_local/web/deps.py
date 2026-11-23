"""Shared plumbing of the web layer.

* ``ApiContext``  per-app state: the service container, the ONE small executor that runs every blocking call
                  of the API (repositories, ``BarReader``, pandas, broker calls) and the background tasks the
                  API started. With two worker threads at most two request threads touch SQLite at a time
                  (see the GIL rule in the architecture notes); FastAPI's 40-thread pool is never used.
* ``Ctx``         the FastAPI dependency that hands the context to a handler (all handlers are ``async def``).
* Helpers for symbols, 404 errors, a small TTL cache and the German texts of validation errors.
"""

import asyncio
import functools
import logging
import math
from collections.abc import Awaitable, Callable, Collection, Coroutine, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any

from fastapi import Depends, HTTPException
from pydantic import BaseModel
from starlette.requests import HTTPConnection

from trading_local.clock import Clock
from trading_local.models import Instrument
from trading_local.services import Services

log = logging.getLogger(__name__)

BLOCKING_WORKERS = 2  # threads of the API executor (= upper bound of API threads reading the database)
MAX_SYMBOL_LENGTH = 24  # same bound as dto.InstrumentCreate
# Upper bounds of id and time parameters. Larger values cannot match anything, but would overflow the
# SQLite driver (64-bit INTEGER) or datetime conversions and end as a 500 instead of a 422.
MAX_ID = 2**63 - 1
MAX_TS = 253_402_300_799  # 9999-12-31 23:59:59 UTC, the last second a datetime can represent


class ApiContext:
    """Per-app state of the web layer (created by ``create_app``, closed by its lifespan)."""

    def __init__(self, services: Services) -> None:
        self.services = services
        self._executor: ThreadPoolExecutor | None = None
        self._tasks: set[asyncio.Task[Any]] = set()

    async def run[**P, T](self, func: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
        """Run a blocking callable on the API executor and await its result (exceptions propagate)."""
        if self._executor is None:  # created lazily: the app may serve requests without a lifespan (tests)
            self._executor = ThreadPoolExecutor(max_workers=BLOCKING_WORKERS, thread_name_prefix="api-blocking")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, functools.partial(func, *args, **kwargs))

    def spawn(self, coro: Coroutine[Any, Any, Any], *, name: str) -> asyncio.Task[Any]:
        """Start a background task that outlives the request; failures are logged, ``aclose`` cancels it."""
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and (exc := task.exception()) is not None:
            log.error("background task %s failed", task.get_name(), exc_info=exc)

    async def aclose(self) -> None:
        """Cancel the background tasks and release the executor: queued blocking calls are dropped, running
        ones are awaited (off the event loop), so no API thread still uses the database when the runtime
        closes it after the server has stopped."""
        await cancel_and_wait(list(self._tasks))
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
            await asyncio.to_thread(executor.shutdown, wait=True)


async def cancel_and_wait(tasks: Collection[asyncio.Task[Any]]) -> None:
    """Cancel tasks and wait until all of them have finished (their results are not inspected).

    ``asyncio.wait`` instead of ``gather``: if the caller itself is cancelled meanwhile, ``gather`` raises a
    NEW CancelledError, which AnyIO cancel scopes (Starlette runs connections inside them) do not recognise
    as their own; the connection task would then end as cancelled instead of finishing cleanly.
    """
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.wait(tasks)


async def get_ctx(connection: HTTPConnection) -> ApiContext:
    """FastAPI dependency (``async`` so that it does not hop through the thread pool)."""
    ctx: ApiContext = connection.app.state.ctx
    return ctx


Ctx = Annotated[ApiContext, Depends(get_ctx)]


# ------------------------------------------------------------------------------------------------
# Symbols and errors
# ------------------------------------------------------------------------------------------------
def normalize_symbol(raw: str) -> str:
    """Symbols are stored in upper case ('sap.de' -> 'SAP.DE')."""
    return raw.strip().upper()


def not_found(detail: str) -> HTTPException:
    return HTTPException(status_code=404, detail=detail)


def require_storable_numbers(body: BaseModel) -> None:
    """422 if a top-level number of a request body cannot be used: NaN or infinity (Python's JSON parser
    accepts the non-standard literals ``NaN`` and ``Infinity``, and the request models in ``dto`` only bound
    their numbers from below) or an integer outside SQLite's 64-bit range."""
    errors: list[dict[str, Any]] = []
    for name, value in body:
        if isinstance(value, float) and not math.isfinite(value):
            errors.append({"loc": ("body", name), "type": "finite_number"})
        elif isinstance(value, int) and not isinstance(value, bool) and not -MAX_ID <= value <= MAX_ID:
            errors.append({"loc": ("body", name), "type": "less_than_equal", "ctx": {"le": MAX_ID}})
    if errors:
        raise HTTPException(status_code=422, detail=validation_detail(errors))


def require_instrument(services: Services, raw_symbol: str) -> Instrument:
    """The instrument (active or not) of a symbol from a path or query parameter; 404 if it is unknown."""
    symbol = normalize_symbol(raw_symbol)
    instrument = services.instruments.get(symbol)
    if instrument is None:
        raise not_found(f"Unbekanntes Symbol: {symbol[:MAX_SYMBOL_LENGTH]}")
    return instrument


class TtlCache[T]:
    """One cached value that is recomputed when it is older than ``ttl_s`` seconds of the given clock.

    Concurrent callers share one computation (asyncio lock); a failed computation leaves the cache empty.
    """

    def __init__(self, clock: Clock, ttl_s: int) -> None:
        self._clock = clock
        self._ttl_s = ttl_s
        self._lock = asyncio.Lock()
        self._value: T | None = None
        self._stored_ts = 0

    async def get(self, compute: Callable[[], Awaitable[T]]) -> T:
        async with self._lock:
            now_ts = self._clock.ts()
            if self._value is not None and 0 <= now_ts - self._stored_ts < self._ttl_s:
                return self._value
            value = await compute()
            self._value, self._stored_ts = value, now_ts
            return value


# ------------------------------------------------------------------------------------------------
# German validation messages
# ------------------------------------------------------------------------------------------------
_PLACES_DE: dict[str, str] = {
    "query": "Abfrageparameter",
    "path": "Pfadparameter",
    "body": "Feld",
    "header": "Header",
    "cookie": "Cookie",
}

# pydantic error type -> German reason; placeholders are filled from the error's ``ctx``.
_REASONS_DE: dict[str, str] = {
    "missing": "fehlt",
    "int_parsing": "muss eine ganze Zahl sein",
    "int_type": "muss eine ganze Zahl sein",
    "int_from_float": "muss eine ganze Zahl sein",
    "float_parsing": "muss eine Zahl sein",
    "float_type": "muss eine Zahl sein",
    "finite_number": "muss eine endliche Zahl sein",
    "bool_parsing": "muss true oder false sein",
    "bool_type": "muss true oder false sein",
    "string_type": "muss ein Text sein",
    "string_too_short": "ist zu kurz (mindestens {min_length} Zeichen)",
    "string_too_long": "ist zu lang (höchstens {max_length} Zeichen)",
    "greater_than": "muss größer als {gt} sein",
    "greater_than_equal": "muss mindestens {ge} sein",
    "less_than": "muss kleiner als {lt} sein",
    "less_than_equal": "darf höchstens {le} sein",
    "literal_error": "muss einer der Werte {expected} sein",
    "enum": "muss einer der Werte {expected} sein",
    "extra_forbidden": "ist kein bekanntes Feld",
    "json_invalid": "ist kein gültiges JSON",
    "dict_type": "muss ein JSON-Objekt sein",
    "model_type": "muss ein JSON-Objekt sein",
    "model_attributes_type": "muss ein JSON-Objekt sein",
    "list_type": "muss eine Liste sein",
}
_DEFAULT_REASON_DE = "ist ungültig"
_MAX_REPORTED_ERRORS = 3


def validation_detail(errors: Sequence[Mapping[str, Any]]) -> str:
    """One German sentence that names the offending fields, e.g.
    "Ungültige Eingabe: Abfrageparameter 'limit' darf höchstens 500 sein."

    ``errors`` are pydantic/FastAPI error dicts; locations without a FastAPI prefix ('query', 'body', ...)
    are reported as plain field names (``RuntimeSettings`` validation).
    """
    if not errors:
        return "Ungültige Eingabe."
    parts = [f"{_place_of(error)} {_reason(error)}" for error in errors[:_MAX_REPORTED_ERRORS]]
    suffix = " (weitere Fehler ausgelassen)" if len(errors) > _MAX_REPORTED_ERRORS else ""
    return f"Ungültige Eingabe: {'; '.join(parts)}{suffix}."


def _place_of(error: Mapping[str, Any]) -> str:
    if error.get("type") == "json_invalid":  # its location is a character offset, not a field
        return "Der Anfrageinhalt"
    return _place(error.get("loc", ()))


def _place(loc: Sequence[Any]) -> str:
    parts = [str(part) for part in loc]
    if parts and parts[0] in _PLACES_DE:
        name = ".".join(parts[1:])
        return f"{_PLACES_DE[parts[0]]} '{name}'" if name else "Der Anfrageinhalt"
    return f"Feld '{'.'.join(parts)}'" if parts else "Die Eingabe"


def _reason(error: Mapping[str, Any]) -> str:
    template = _REASONS_DE.get(str(error.get("type", "")), _DEFAULT_REASON_DE)
    values = {key: _format_value(value) for key, value in (error.get("ctx") or {}).items()}
    try:
        return template.format_map(values)
    except (KeyError, IndexError, ValueError):
        return _DEFAULT_REASON_DE


def _format_value(value: Any) -> str:
    """Bounds as a German reader expects them: 0.0 -> '0', 0.5 -> '0,5'; enumerations joined with 'oder'."""
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:g}".replace(".", ",")
    return str(value).replace(" or ", " oder ")
