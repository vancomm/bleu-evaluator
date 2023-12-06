import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError as e:
    print(e)
    nltk.download("punkt")

from .cli import cli
