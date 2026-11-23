import logging
import re
from collections.abc import Iterator
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest
from pydantic import SecretStr

from trading_local.config import Settings
from trading_local.logging_setup import (
    CONSOLE_HANDLER_NAME,
    FILE_HANDLER_NAME,
    LOG_FILE_NAME,
    NOISY_LOGGERS,
    setup_logging,
)
from trading_local.testing import make_settings

OWN_HANDLERS = (CONSOLE_HANDLER_NAME, FILE_HANDLER_NAME)
SECRET = "hunter2-Geheim!"


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


def settings_in(directory: Path, **overrides: object) -> Settings:
    """Test settings with a pinned level and no secret, so that TL_* environment variables cannot interfere."""
    return make_settings(directory, **{"log_level": "INFO", "smtp_password": "", **overrides})


def own_handlers() -> list[logging.Handler]:
    return [handler for handler in logging.getLogger().handlers if handler.name in OWN_HANDLERS]


def log_text(directory: Path) -> str:
    return (directory / "logs" / LOG_FILE_NAME).read_text(encoding="utf-8")


def test_setup_installs_a_console_and_a_rotating_file_handler(tmp_path: Path) -> None:
    settings = settings_in(tmp_path)

    setup_logging(settings)

    console = logging.getHandlerByName(CONSOLE_HANDLER_NAME)
    file_handler = logging.getHandlerByName(FILE_HANDLER_NAME)
    assert own_handlers() == [console, file_handler]
    assert type(console) is logging.StreamHandler
    assert isinstance(file_handler, RotatingFileHandler)
    assert Path(file_handler.baseFilename) == settings.log_dir / "trading_local.log"
    assert file_handler.maxBytes == 5 * 1024 * 1024
    assert file_handler.backupCount == 5
    assert file_handler.encoding == "utf-8"
    assert (settings.log_dir / "trading_local.log").is_file()


def test_setup_itself_writes_nothing_to_the_log(tmp_path: Path) -> None:
    setup_logging(settings_in(tmp_path, smtp_user="jemand@example.invalid", smtp_password=SECRET))

    assert log_text(tmp_path) == ""


def test_setup_is_idempotent(tmp_path: Path) -> None:
    foreign = logging.NullHandler()
    logging.getLogger().addHandler(foreign)
    settings = settings_in(tmp_path)

    setup_logging(settings)
    first = own_handlers()
    setup_logging(settings)
    setup_logging(settings)

    assert [handler.name for handler in own_handlers()] == list(OWN_HANDLERS)
    assert all(handler not in logging.getLogger().handlers for handler in first)
    assert isinstance(first[1], RotatingFileHandler) and first[1].stream is None  # the old file handle is closed
    assert logging.getHandlerByName(FILE_HANDLER_NAME) is own_handlers()[1]
    assert foreign in logging.getLogger().handlers  # handlers of others are left alone

    logging.getLogger("trading_local.test").warning("nur einmal")
    assert log_text(tmp_path).count("nur einmal") == 1


def test_records_are_written_with_time_level_logger_name_and_message(tmp_path: Path) -> None:
    setup_logging(settings_in(tmp_path))

    logging.getLogger("trading_local.signals").info("Signal für %s: +%.1f %% in Zürich, 5 €", "SAP.DE", 1.25)

    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} INFO +trading_local\.signals: "
        r"Signal für SAP\.DE: \+1\.2 % in Zürich, 5 €\n",
        log_text(tmp_path),
    )


def test_console_shows_the_same_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(settings_in(tmp_path))

    logging.getLogger("trading_local.data").error("Datenfeed unterbrochen")

    assert capsys.readouterr().err.rstrip("\n") == log_text(tmp_path).rstrip("\n")
    assert "ERROR    trading_local.data: Datenfeed unterbrochen" in log_text(tmp_path)


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("DEBUG", logging.DEBUG),
        ("debug", logging.DEBUG),
        ("INFO", logging.INFO),
        (" warning ", logging.WARNING),
        ("Error", logging.ERROR),
    ],
)
def test_level_comes_from_the_settings(tmp_path: Path, configured: str, expected: int) -> None:
    setup_logging(settings_in(tmp_path, log_level=configured))
    log = logging.getLogger("trading_local.test")

    log.debug("level-debug")
    log.info("level-info")
    log.warning("level-warning")
    log.error("level-error")

    assert logging.getLogger().level == expected
    written = [name for name in ("debug", "info", "warning", "error") if f"level-{name}" in log_text(tmp_path)]
    assert written == ["debug", "info", "warning", "error"][expected // 10 - 1 :]


def test_unknown_level_falls_back_to_info_and_says_so(tmp_path: Path) -> None:
    setup_logging(settings_in(tmp_path, log_level="LOUD"))

    assert logging.getLogger().level == logging.INFO
    assert re.search(r"WARNING +trading_local\.logging_setup: unknown log level 'LOUD'", log_text(tmp_path))


def test_noisy_libraries_are_limited_to_warnings(tmp_path: Path) -> None:
    expected = "yfinance peewee httpx httpcore urllib3 matplotlib PIL asyncio websockets curl_cffi"
    assert sorted(NOISY_LOGGERS) == sorted(expected.split())
    setup_logging(settings_in(tmp_path, log_level="DEBUG"))

    for name in NOISY_LOGGERS:
        assert logging.getLogger(name).level == logging.WARNING
        logging.getLogger(name).info("chatter from %s", name)
        logging.getLogger(f"{name}.submodule").debug("chatter from a child of %s", name)
        logging.getLogger(name).warning("problem in %s", name)

    text = log_text(tmp_path)
    assert "chatter" not in text
    assert all(f"problem in {name}" in text for name in NOISY_LOGGERS)


def test_secret_settings_are_masked_in_file_and_console(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(settings_in(tmp_path, smtp_password=SECRET))
    log = logging.getLogger("trading_local.mail")

    log.warning("login failed with password " + SECRET)
    log.warning("arguments are covered too: %s / %r", SECRET, {"password": SECRET})
    try:
        raise RuntimeError(f"535 authentication failed for secret {SECRET}")
    except RuntimeError:
        log.exception("smtp error")

    for output in (log_text(tmp_path), capsys.readouterr().err):
        assert SECRET not in output
        assert "login failed with password ***" in output
        assert "arguments are covered too: *** / {'password': '***'}" in output
        assert "RuntimeError: 535 authentication failed for secret ***" in output
        assert "Traceback (most recent call last)" in output


def test_every_secret_setting_is_masked_and_the_longest_one_first(tmp_path: Path) -> None:
    class SettingsWithApiKey(Settings):
        api_key: SecretStr = SecretStr("")

    settings = SettingsWithApiKey(
        _env_file=None, data_dir=tmp_path, log_level="INFO", smtp_password="abc123", api_key="abc123-and-more"
    )
    setup_logging(settings)

    logging.getLogger("trading_local.test").warning("key=%s password=%s", "abc123-and-more", "abc123")

    assert "key=*** password=***\n" in log_text(tmp_path)  # not '***-and-more'


def test_an_empty_secret_masks_nothing(tmp_path: Path) -> None:
    setup_logging(settings_in(tmp_path))  # no SMTP password configured

    logging.getLogger("trading_local.test").warning("plain text stays plain")

    assert "plain text stays plain" in log_text(tmp_path)
    assert "***" not in log_text(tmp_path)


def test_new_settings_replace_the_old_configuration(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    setup_logging(settings_in(first, log_level="DEBUG", smtp_password=SECRET))
    logging.getLogger("trading_local.test").debug("to the first file")

    setup_logging(settings_in(second, log_level="WARNING"))
    logging.getLogger("trading_local.test").info("suppressed by the new level")
    logging.getLogger("trading_local.test").warning("to the second file: " + SECRET)

    assert "to the first file" in log_text(first)
    assert "second file" not in log_text(first)
    assert log_text(second).count("\n") == 1
    assert "to the second file: " + SECRET in log_text(second)  # the old secret is no longer configured
