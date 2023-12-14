import abc
from pathlib import Path
from dataclasses import dataclass

import docx2txt
import pypdf


@dataclass
class BaseParser(abc.ABC):
    file: Path

    @abc.abstractmethod
    def read_all(self) -> str:
        ...


class PlainTextParser(BaseParser):
    def read_all(self) -> str:
        return self.file.read_text()


class DocParser(BaseParser):
    def read_all(self) -> str:
        return docx2txt.process(self.file)


class PdfParser(BaseParser):
    def read_all(self) -> str:
        reader = pypdf.PdfReader(self.file)
        text = "".join(page.extract_text() for page in reader.pages)
        return text


def get_parser(file: Path) -> BaseParser:
    match file.suffix.lower():
        case ".txt":
            return PlainTextParser(file)
        case ".doc" | ".docx":
            return DocParser(file)
        case ".pdf":
            return PdfParser(file)
        case _:
            return PlainTextParser(file)
