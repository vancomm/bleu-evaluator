import logging

from .cli import cli
from .log import setup_base_logging


setup_base_logging(level=logging.DEBUG)
