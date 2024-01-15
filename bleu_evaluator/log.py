import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime, timezone


LOG_FORMATS = {
    "default": "%(asctime)s | %(levelname)s: %(message)s",
    "debug": "%(asctime)s | %(levelname)s: %(message)s @ %(name)s/%(funcName)s:%(lineno)d",
    "short": "%(levelname)s: %(message)s",
}


def format_logging_time(
    record: logging.LogRecord,
    datefmt: str | None = None,
) -> str:
    return (
        datetime.fromtimestamp(record.created, timezone.utc)
        .astimezone()
        .isoformat(sep="T", timespec="seconds")
    )


def setup_stream_logging(
    *,
    stream=sys.stderr,
    level: int | str | None = None,
    format: str = LOG_FORMATS["default"],
    capture_warnings: bool = True,
) -> None:
    logging.captureWarnings(capture=capture_warnings)

    root = logging.getLogger()

    formatter = logging.Formatter(format)
    formatter.formatTime = format_logging_time  # type: ignore[method-assign]

    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)

    root.addHandler(handler)


def setup_file_logging(
    logfile: Path,
    *,
    level: int | str = logging.INFO,
    format: str = LOG_FORMATS["default"],
    backup_count: int = 10,
    max_bytes: int = 1 * 1024 * 1024,
    capture_warnings: bool = True,
) -> None:
    logging.captureWarnings(capture=capture_warnings)

    root = logging.getLogger()

    handler = RotatingFileHandler(
        logfile,
        maxBytes=max_bytes,
        backupCount=backup_count,
    )

    formatter = logging.Formatter(format)
    formatter.formatTime = format_logging_time  # type: ignore[method-assign]

    handler.setLevel(level)
    handler.setFormatter(formatter)

    root.addHandler(handler)
