import logging

import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError as e:
    nltk.download("punkt")

from .cli import cli
