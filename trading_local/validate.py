"""Command line: run a validation replay and store its result.

    python -m trading_local.validate [--days 30] [--run-id N]

The web API (``trading_local.web.routes.validation``) starts this module as a separate process, so the
CPU-bound replay never competes with the server's threads for the GIL; it can also be run by hand. With
``--run-id`` it finishes that existing ``validation_runs`` row (created by the API with status 'running'),
otherwise it creates its own row. The result is stored with status 'done'; any failure is stored as
'failed' with a German message. It works while the main application is running: the database is in WAL
mode, the replay only reads, and each write is one short transaction.

Exit codes: 0 when the run finished, 1 when it failed or the run id cannot be used, 2 for invalid arguments.
"""

import argparse
import logging
import sys
import time
from collections.abc import Callable, Sequence

from pydantic import ValidationError

from trading_local.analysis.params import ENGINE_PARAMS
from trading_local.analysis.validation import run_validation
from trading_local.clock import Clock, SystemClock
from trading_local.config import DEFAULT_COSTS, Settings
from trading_local.contracts import MarketCalendar
from trading_local.db import repo_ops
from trading_local.db.database import Database
from trading_local.db.repo_market import InstrumentBook
from trading_local.dto import ValidationRequest, ValidationResultOut

log = logging.getLogger("trading_local.validate")

EXIT_OK = 0
EXIT_FAILED = 1
_MAX_MESSAGE_CHARS = 500
_PROGRESS_STEP = 0.1


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m trading_local.validate",
        description="Validierung des Regelwerks auf der gespeicherten Minutenhistorie (ohne LLM)",
    )
    default_days = ValidationRequest.model_fields["days"].default
    parser.add_argument("--days", type=int, default=default_days, help=f"Kalendertage (Standard {default_days})")
    parser.add_argument("--run-id", type=int, default=None, help="vorhandenen Validierungslauf abschließen")
    args = parser.parse_args(argv)
    try:
        ValidationRequest(days=args.days)  # the same bounds as the web API
    except ValidationError:
        parser.error("--days muss zwischen 2 und 365 liegen")
    return args


def _number_de(value: int) -> str:
    """Integer with German thousands separators, e.g. '12.345'."""
    return f"{value:,}".replace(",", ".")


def _done_message(result: ValidationResultOut, duration_s: float) -> str:
    return (
        f"Abgeschlossen in {duration_s:.0f} s: {_number_de(result.overall.n)} ausgewählte Signale aus "
        f"{_number_de(result.decisions)} Entscheidungspunkten, {result.symbols} Instrumente."
    )


def _progress_printer() -> Callable[[float], None]:
    """Progress callback that prints every further 10 % to stdout."""
    printed = 0

    def show(fraction: float) -> None:
        nonlocal printed
        step = int(fraction / _PROGRESS_STEP + 1e-9)
        if step > printed:
            printed = step
            print(f"Fortschritt: {step * 10} %", flush=True)

    return show


def _print_summary(run_id: int, days: int, result: ValidationResultOut, duration_s: float) -> None:
    print(
        f"Validierung #{run_id} abgeschlossen in {duration_s:.0f} s "
        f"({days} Tage, {result.symbols} Instrumente, {result.trading_days} Handelstage)."
    )
    print(result.verdict_de)
    if result.warnings_de:
        print("Hinweise:")
        for warning in result.warnings_de:
            print(f"  - {warning}")


def _mark_failed(db: Database, run_id: int, message: str) -> None:
    """Store the failure; a database error here is only logged (the API marks runs without result itself)."""
    try:
        repo_ops.finish_validation_run(db, run_id, status="failed", message=message[:_MAX_MESSAGE_CHARS])
    except Exception:
        log.exception("could not mark validation run %d as failed", run_id)


def _resolve_run(db: Database, run_id: int | None, days: int, clock: Clock) -> int | None:
    """The id of the run to finish: the given one if it is still running, else a newly created one."""
    if run_id is None:
        return repo_ops.create_validation_run(
            db, ts=clock.ts(), engine_version=ENGINE_PARAMS.version, params={"days": days}
        )
    run = repo_ops.get_validation_run(db, run_id)
    if run is None:
        print(f"Validierungslauf #{run_id} existiert nicht.", file=sys.stderr)
        return None
    if run.status != "running":
        print(f"Validierungslauf #{run_id} ist nicht mehr aktiv (Status '{run.status}').", file=sys.stderr)
        return None
    return run.id


def _execute(db: Database, args: argparse.Namespace, *, calendar: MarketCalendar, clock: Clock) -> int:
    db.initialize()
    run_id = _resolve_run(db, args.run_id, args.days, clock)
    if run_id is None:
        return EXIT_FAILED
    started = time.perf_counter()
    try:
        result = run_validation(
            db,
            calendar=calendar,
            instruments=InstrumentBook(db).tradable(),
            costs=DEFAULT_COSTS,
            days=args.days,
            params=ENGINE_PARAMS,
            now_ts=clock.ts(),
            progress=_progress_printer(),
        )
        duration_s = time.perf_counter() - started
        repo_ops.finish_validation_run(
            db, run_id, status="done", result=result.model_dump(mode="json"), message=_done_message(result, duration_s)
        )
    except KeyboardInterrupt:
        _mark_failed(db, run_id, "Die Validierung wurde abgebrochen.")
        print("Validierung abgebrochen.", file=sys.stderr)
        return EXIT_FAILED
    except Exception as exc:
        log.exception("validation run %d failed", run_id)
        _mark_failed(db, run_id, f"Die Validierung ist fehlgeschlagen ({type(exc).__name__}): {exc}")
        print(f"Validierung #{run_id} fehlgeschlagen: {exc}", file=sys.stderr)
        return EXIT_FAILED
    _print_summary(run_id, args.days, result, duration_s)
    return EXIT_OK


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: Settings | None = None,
    calendar: MarketCalendar | None = None,
    clock: Clock | None = None,
) -> int:
    """Run one validation and return the exit code.

    Without ``settings`` they are read from the environment / ``.env`` and logging is configured; callers
    that pass their own settings (tests, embedding code) keep their logging setup. ``calendar`` defaults to
    the platform calendar, ``clock`` to the system clock.
    """
    args = _parse_args(argv)
    if settings is None:
        from trading_local.logging_setup import setup_logging

        settings = Settings()
        setup_logging(settings, file_name="validation.log")  # the server holds trading_local.log open
    try:
        if calendar is None:
            from trading_local.markets import build_calendar

            calendar = build_calendar()
        db = Database(settings.db_path)
    except Exception:
        log.exception("validation could not start")
        print("Die Validierung konnte nicht gestartet werden (Details im Log).", file=sys.stderr)
        return EXIT_FAILED
    try:
        return _execute(db, args, calendar=calendar, clock=clock or SystemClock())
    except Exception:
        log.exception("validation could not use the database")
        print("Die Validierung konnte nicht auf die Datenbank zugreifen (Details im Log).", file=sys.stderr)
        return EXIT_FAILED
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
