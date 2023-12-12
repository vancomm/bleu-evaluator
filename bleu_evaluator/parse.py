import abc
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Sequence

import nltk
import docx2txt
import pypdf


@dataclass
class BaseParser(abc.ABC):
    file: Path

    @abc.abstractmethod
    def read(self) -> str:
        ...

    @classmethod
    def word_tokenize(
        cls, sentence: str, *, remove_chars: str | Sequence[str] = ".,;:-"
    ) -> list[str]:
        return [
            word.casefold()
            for word in nltk.word_tokenize(sentence)
            if word not in remove_chars
        ]

    @classmethod
    def parse_corpus(cls, text: str) -> list[list[str]]:
        return [cls.word_tokenize(sent) for sent in nltk.sent_tokenize(text.strip())]

    @classmethod
    def parse_corpora(cls, text: str) -> list[list[list[str]]]:
        return [
            [cls.word_tokenize(sent) for sent in nltk.sent_tokenize(corpus.strip())]
            for corpus in text.strip().split("\n\n")
        ]

    def to_corpus(self) -> list[list[str]]:
        return self.parse_corpus(self.read())

    def to_corpora(self) -> list[list[list[str]]]:
        return self.parse_corpora(self.read())


class PlainTextParser(BaseParser):
    def read(self) -> str:
        return self.file.read_text()


class DocParser(BaseParser):
    def read(self) -> str:
        return docx2txt.process(self.file)


class PdfParser(BaseParser):
    def read(self) -> str:
        reader = pypdf.PdfReader(self.file)
        text = "".join(page.extract_text() for page in reader.pages)
        return text


def get_parser(file: Path) -> BaseParser:
    match file.suffix.lower():
        case ".doc" | ".docx":
            return DocParser(file)
        case ".txt":
            return PlainTextParser(file)
        case ".pdf":
            return PdfParser(file)
        case _:
            return PlainTextParser(file)
