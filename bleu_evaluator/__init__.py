import nltk
from .cli import cli
from .bleu import BLEU

try:
    nltk.data.find("tokenizers/punkt")
except LookupError as e:
    nltk.download("punkt")
