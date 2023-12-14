import nltk
from .cli import cli

try:
    nltk.data.find("tokenizers/punkt")
except LookupError as e:
    nltk.download("punkt")

if __name__ == "__main__":
    cli()
