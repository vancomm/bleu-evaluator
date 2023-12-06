import pathlib
from collections.abc import Sequence
from typing import Callable

import nltk
import docx2txt


def read_text(text_file: pathlib.Path) -> str:
    return text_file.read_text()


def read_doc(doc_file: pathlib.Path) -> str:
    return docx2txt.process(doc_file)


def reader_factory(file: pathlib.Path) -> Callable[[pathlib.Path], str]:
    match file.suffix.lower():
        case ".doc" | ".docx":
            return read_doc
        case ".txt":
            return read_text
        case _:
            return read_text


def tokenize(
    sentence: str, *, remove_chars: str | Sequence[str] = ".,;:-"
) -> list[str]:
    return [
        word.lower()
        for word in nltk.tokenize.word_tokenize(sentence)
        if word not in remove_chars
    ]


def parse(file: pathlib.Path) -> list[list[str]]:
    read = reader_factory(file)
    corpus = read(file)

    return [tokenize(sent) for sent in nltk.sent_tokenize(corpus)]
