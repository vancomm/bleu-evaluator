import abc
from pathlib import Path
from dataclasses import dataclass

import docx2txt
import pypdf


@dataclass
class BaseReader(abc.ABC):
    file: Path

    @abc.abstractmethod
    def read_all(self) -> str:
        ...


class PlainTextReader(BaseReader):
    def read_all(self) -> str:
        return self.file.read_text()


class DocReader(BaseReader):
    def read_all(self) -> str:
        return docx2txt.process(self.file)


class PdfReader(BaseReader):
    def read_all(self) -> str:
        reader = pypdf.PdfReader(self.file)
        text = "".join(page.extract_text() for page in reader.pages)
        return text


def get_reader(file: Path) -> BaseReader:
    match file.suffix.lower():
        case ".txt":
            return PlainTextReader(file)
        case ".doc" | ".docx":
            return DocReader(file)
        case ".pdf":
            return PdfReader(file)
        case _:
            return PlainTextReader(file)
