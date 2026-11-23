"""Review tests for ``trading_local.logging_setup``: idempotence when several threads configure logging."""

import logging
import threading
from collections.abc import Iterator
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from trading_local.logging_setup import (
    CONSOLE_HANDLER_NAME,
    FILE_HANDLER_NAME,
    LOG_FILE_NAME,
    NOISY_LOGGERS,
    setup_logging,
)
from trading_local.testing import make_settings

OWN_HANDLERS = (CONSOLE_HANDLER_NAME, FILE_HANDLER_NAME)


@pytest.fixture(autouse=True)
def restore_logging() -> Iterator[None]:
    """Undo everything setup_logging changes globally and release the log files of the temporary directories."""
    root = logging.getLogger()
    root_level = root.level
    foreign_handlers = list(root.handlers)
    library_levels = {name: logging.getLogger(name).level for name in NOISY_LOGGERS}
    yield
    for handler in [handler for handler in root.handlers if handler not in foreign_handlers]:
        root.removeHandler(handler)
        handler.close()
    root.setLevel(root_level)
    for name, level in library_levels.items():
        logging.getLogger(name).setLevel(level)


def test_concurrent_setup_calls_leave_exactly_one_pair_of_handlers(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, log_level="INFO", smtp_password="")
    callers = 8
    start = threading.Barrier(callers)
    created: list[RotatingFileHandler] = []
    errors: list[BaseException] = []
    original_init = RotatingFileHandler.__init__

    def recording_init(self: RotatingFileHandler, *args: object, **kwargs: object) -> None:
        original_init(self, *args, **kwargs)  # type: ignore[arg-type]
        created.append(self)

    def configure() -> None:
        try:
            start.wait()
            for _ in range(5):
                setup_logging(settings)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(RotatingFileHandler, "__init__", recording_init)
        threads = [threading.Thread(target=configure) for _ in range(callers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert not errors
    own = [handler for handler in logging.getLogger().handlers if handler.name in OWN_HANDLERS]
    assert [handler.name for handler in own] == list(OWN_HANDLERS)
    assert logging.getHandlerByName(CONSOLE_HANDLER_NAME) is own[0]
    assert logging.getHandlerByName(FILE_HANDLER_NAME) is own[1]
    # every file handler but the installed one was closed again: no leaked handle on the log file
    assert len(created) == callers * 5
    assert [handler for handler in created if handler.stream is not None] == [own[1]]

    logging.getLogger("trading_local.test").warning("genau einmal")
    text = (tmp_path / "logs" / LOG_FILE_NAME).read_text(encoding="utf-8")
    assert text.count("genau einmal") == 1


def test_setup_keeps_what_the_previous_configuration_already_wrote(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, log_level="INFO", smtp_password="")
    log = logging.getLogger("trading_local.test")

    setup_logging(settings)
    log.warning("vor dem Neustart")
    setup_logging(settings)
    log.warning("nach dem Neustart")

    lines = (tmp_path / "logs" / LOG_FILE_NAME).read_text(encoding="utf-8").splitlines()
    assert [line.rsplit(": ", 1)[1] for line in lines] == ["vor dem Neustart", "nach dem Neustart"]  # appended
