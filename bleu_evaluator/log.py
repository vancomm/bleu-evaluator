import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime, timezone


LOG_FORMATS = {
    "default": "%(asctime)s | %(levelname)s: %(message)s",
    "debug": "%(asctime)s | %(levelname)s: %(message)s @ %(name)s/%(funcName)s:%(lineno)d",
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


def setup_base_logging(
    *, level: int | str = logging.INFO, format: str = LOG_FORMATS["default"]
) -> None:
    logging.captureWarnings(capture=True)
    logging.basicConfig(level=level, format=format)
    logging.Formatter.formatTime = (  # type: ignore[method-assign]
        lambda self, record, datefmt=None: format_logging_time(record, datefmt)
    )


def setup_file_logging(
    logfile: Path,
    *,
    level: int | str = logging.INFO,
    format: str = LOG_FORMATS["default"],
    backup_count: int = 10,
    max_bytes: int = 1 * 1024 * 1024,
) -> None:
    fh = RotatingFileHandler(
        logfile,
        maxBytes=max_bytes,
        backupCount=backup_count,
    )

    formatter = logging.Formatter(format)
    formatter.formatTime = format_logging_time  # type: ignore[method-assign]

    fh.setLevel(level)
    fh.setFormatter(formatter)

    logging.getLogger().addHandler(fh)
