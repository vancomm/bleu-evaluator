import math
from collections import Counter
from collections.abc import Sequence
from typing import TypeVar

import nltk
import numpy as np


T = TypeVar("T")


def floor_log(num: float) -> float:
    if num == 0:
        return -1000000000
    return math.log(num)


def get_closest_len(hyp_len: int, ref_lens: list[int]) -> int:
    return ref_lens[np.absolute(np.array(ref_lens) - hyp_len).argmin()]


def sent_tokenize(text: str) -> list[str]:
    return nltk.sent_tokenize(text.strip())


def word_tokenize(sentence: str, *, casefold: bool = True) -> list[str]:
    return [
        word.casefold() if casefold else word for word in nltk.word_tokenize(sentence)
    ]


def count_ngrams(
    tokens: Sequence[str], min_order: int, max_order: int
) -> Counter[tuple[str, ...]]:
    ngrams = (
        tuple(tokens[i : i + n])
        for n in range(min_order, max_order + 1)
        for i in range(0, len(tokens) - n + 1)
    )
    return Counter(ngrams)
