import pathlib
from typing import Callable

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
