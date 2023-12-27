import math
from operator import add
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypeVar

import nltk
import numpy as np

from .compat import Self


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


def extract_ngrams(
    tokens: Sequence[str], min_order: int, max_order: int
) -> Counter[tuple[str, ...]]:
    ngrams = (
        tuple(tokens[i : i + n])
        for n in range(min_order, max_order + 1)
        for i in range(0, len(tokens) - n + 1)
    )
    return Counter(ngrams)


@dataclass
class RefStats:
    lens: list[int] = field(default_factory=list)
    ngrams: Counter[tuple[str, ...]] = field(default_factory=Counter)

    @classmethod
    def compute(cls, refs: Sequence[str]) -> Self:
        stats = cls()

        for ref in refs:
            tokens = word_tokenize(ref)
            stats.lens.append(len(tokens))
            sent_ngrams = extract_ngrams(tokens, 1, 4)
            if not stats.ngrams:
                stats.ngrams = sent_ngrams
            else:
                for ngram, count in sent_ngrams.items():
                    stats.ngrams[ngram] = max(stats.ngrams[ngram], count)

        return stats


@dataclass
class HypothesisStats:
    hyp_len: int
    ref_len: int
    correct: list[int]
    total: list[int]

    @classmethod
    def compute(cls, sentence: str, ref_stats: RefStats) -> Self:
        hyp_tokens = word_tokenize(sentence)
        hyp_ngram_counts = extract_ngrams(hyp_tokens, min_order=1, max_order=4)
        ref_len = get_closest_len(len(hyp_tokens), ref_stats.lens)

        correct = [0] * 4
        total = [0] * 4
        for ngram, count in hyp_ngram_counts.items():
            n = len(ngram) - 1
            total[n] += count
            if ngram in ref_stats.ngrams:
                correct[n] += min(count, ref_stats.ngrams[ngram])

        return cls(
            hyp_len=len(hyp_tokens),
            ref_len=ref_len,
            correct=correct,
            total=total,
        )

    def __add__(self, other) -> Self:
        if not isinstance(other, HypothesisStats):
            return NotImplemented

        return self.__class__(
            hyp_len=self.hyp_len + other.hyp_len,
            ref_len=self.ref_len + other.ref_len,
            correct=list(map(add, self.correct, other.correct)),
            total=list(map(add, self.total, other.total)),
        )

    def __radd__(self, other) -> Self:
        return self + other
