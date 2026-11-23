"""Validation runs (walk-forward replay on the stored history).

The replay is CPU-bound pure Python, so it runs in its own process (``python -m trading_local.validate``)
instead of sharing the GIL with the database threads. The process writes its result into its
``validation_runs`` row; this module creates the row, starts the process detached from the request,
watches it and announces the end on ``TOPIC_VALIDATION``. One run at a time: a new run is refused while a
watched process is alive or a 'running' row is younger than two hours. A row that is still 'running' when
its process has exited is marked 'failed'; so is a 'running' row older than two hours that this server does
not watch (left over from a server that was stopped mid-run, whose process may have died unseen).
"""

import asyncio
import logging
import subprocess
import sys
import threading
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request

from trading_local.config import PROJECT_ROOT
from trading_local.db import repo_ops
from trading_local.dto import ValidationRequest, ValidationRunOut
from trading_local.events import TOPIC_VALIDATION
from trading_local.web.deps import ApiContext

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

VALIDATION_MODULE = "trading_local.validate"
STALE_AFTER_S = 2 * 3600  # a 'running' row older than this without a live process is considered dead
POLL_INTERVAL_S = 2.0
LATEST_RUNS = 5
_RUNNING_SCAN = 20  # newest rows inspected for 'running' ones (a new run is refused while one is running)

# Detached: no console window, and Ctrl+C in the server console does not reach the replay.
_DETACH: dict[str, Any] = (
    {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    if sys.platform == "win32"
    else {"start_new_session": True}
)


def validation_command(days: int, run_id: int) -> list[str]:
    """Fixed argument list of the validation process (no shell involved)."""
    return [sys.executable, "-m", VALIDATION_MODULE, "--days", str(days), "--run-id", str(run_id)]


class ValidationRunner:
    """Starts and watches validation processes (one at a time). Thread-safe."""

    def __init__(self, ctx: ApiContext, *, poll_interval_s: float = POLL_INTERVAL_S) -> None:
        self.poll_interval_s = poll_interval_s
        self._ctx = ctx
        self._lock = threading.Lock()
        self._processes: dict[int, subprocess.Popen[bytes]] = {}

    async def latest(self) -> list[ValidationRunOut]:
        return await self._ctx.run(self._latest)

    async def start(self, days: int) -> ValidationRunOut:
        """Create the run and start its process; 409 while another run is active."""
        run, process = await self._ctx.run(self._create_and_spawn, days)
        self._ctx.spawn(self._watch(run.id, process), name=f"validation-run-{run.id}")
        return run

    # -- blocking parts (API executor) ----------------------------------------------------------
    def _latest(self) -> list[ValidationRunOut]:
        with self._lock:
            self._expire_stale(self._ctx.services.clock.ts())
        return repo_ops.list_validation_runs(self._ctx.services.db, limit=LATEST_RUNS)

    def _create_and_spawn(self, days: int) -> tuple[ValidationRunOut, subprocess.Popen[bytes]]:
        services = self._ctx.services
        db = services.db
        with self._lock:
            now_ts = services.clock.ts()
            self._expire_stale(now_ts)
            if self._busy(now_ts):
                raise HTTPException(status_code=409, detail="Es läuft bereits eine Validierung. Bitte warten.")
            run_id = repo_ops.create_validation_run(
                db, ts=now_ts, engine_version=services.engine.version, params={"days": days}
            )
            try:
                process = subprocess.Popen(
                    validation_command(days, run_id),
                    cwd=PROJECT_ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    close_fds=True,
                    **_DETACH,
                )
            except OSError:
                log.exception("could not start the validation process for run %d", run_id)
                repo_ops.finish_validation_run(
                    db, run_id, status="failed", message="Der Validierungsprozess konnte nicht gestartet werden."
                )
                raise HTTPException(status_code=503, detail="Die Validierung konnte nicht gestartet werden.") from None
            self._processes[run_id] = process
        run = repo_ops.get_validation_run(db, run_id)
        if run is None:  # the row was created a moment ago in this thread
            raise RuntimeError(f"validation run {run_id} vanished")
        return run, process

    def _finish(self, run_id: int, returncode: int) -> str:
        """Final status of a run whose process exited; a row left 'running' is marked 'failed'."""
        db = self._ctx.services.db
        with self._lock:
            self._processes.pop(run_id, None)
            run = repo_ops.get_validation_run(db, run_id)
            if run is None:
                return "failed"
            if run.status == "running":
                log.warning("validation run %d: process exited with code %d without a result", run_id, returncode)
                repo_ops.finish_validation_run(
                    db,
                    run_id,
                    status="failed",
                    message=f"Der Validierungsprozess endete ohne Ergebnis (Exit-Code {returncode}).",
                )
                return "failed"
            return run.status

    def _busy(self, now_ts: int) -> bool:
        """A run is active: a watched process is still alive, or a 'running' row is younger than
        ``STALE_AFTER_S`` (possibly started by a previous server process). Caller holds the lock."""
        if any(process.poll() is None for process in self._processes.values()):
            return True
        return any(now_ts - run.ts < STALE_AFTER_S for run in self._running_runs())

    def _running_runs(self) -> list[ValidationRunOut]:
        recent = repo_ops.list_validation_runs(self._ctx.services.db, limit=_RUNNING_SCAN)
        return [run for run in recent if run.status == "running"]

    def _expire_stale(self, now_ts: int) -> None:
        """Mark 'running' rows older than ``STALE_AFTER_S`` as failed unless this server still watches their
        live process (rows of a server that was restarted mid-run have no watcher). Caller holds the lock."""
        for run in self._running_runs():
            process = self._processes.get(run.id)
            if now_ts - run.ts >= STALE_AFTER_S and (process is None or process.poll() is not None):
                log.warning("validation run %d did not finish within %d s: marked failed", run.id, STALE_AFTER_S)
                repo_ops.finish_validation_run(
                    self._ctx.services.db,
                    run.id,
                    status="failed",
                    message="Keine Rückmeldung der Validierung innerhalb von zwei Stunden.",
                )

    # -- watcher --------------------------------------------------------------------------------
    async def _watch(self, run_id: int, process: subprocess.Popen[bytes]) -> None:
        while process.poll() is None:
            await asyncio.sleep(self.poll_interval_s)
        status = await self._ctx.run(self._finish, run_id, process.returncode)
        self._ctx.services.bus.publish(TOPIC_VALIDATION, {"id": run_id, "status": status})


async def get_runner(request: Request) -> ValidationRunner:
    runner: ValidationRunner = request.app.state.validation
    return runner


Runner = Annotated[ValidationRunner, Depends(get_runner)]


@router.get("/validation")
async def list_validation_runs(runner: Runner) -> list[ValidationRunOut]:
    return await runner.latest()


@router.post("/validation/run", status_code=202)
async def start_validation_run(runner: Runner, body: ValidationRequest) -> ValidationRunOut:
    return await runner.start(body.days)
