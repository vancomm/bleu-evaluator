import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime, timezone


LOGGING_FMT = (
    "%(asctime)s | %(levelname)s: %(message)s @ %(name)s/%(funcName)s:%(lineno)d"
)


def format_logging_time(
    record: logging.LogRecord,
    datefmt: str | None = None,
) -> str:
    return (
        datetime.fromtimestamp(record.created, timezone.utc)
        .astimezone()
        .isoformat(sep="T", timespec="seconds")
    )


def setup_base_logging(*, level: int | str = logging.INFO):
    logging.captureWarnings(capture=True)
    logging.basicConfig(level=level, format=LOGGING_FMT)
    logging.Formatter.formatTime = (
        lambda self, record, datefmt=None: format_logging_time(record, datefmt)
    )


def setup_file_logging(
    logfile: Path,
    *,
    level: int | str = logging.INFO,
    backup_count: int = 10,
    max_bytes: int = 5 * 1024 * 1024,
):
    fh = RotatingFileHandler(
        logfile,
        maxBytes=max_bytes,
        backupCount=backup_count,
    )

    formatter = logging.Formatter(LOGGING_FMT)
    formatter.formatTime = format_logging_time

    fh.setLevel(level)
    fh.setFormatter(formatter)

    logging.getLogger().addHandler(fh)
