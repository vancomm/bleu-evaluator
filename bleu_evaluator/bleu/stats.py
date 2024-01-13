import operator
from collections import Counter
from dataclasses import dataclass, field
from collections.abc import Sequence

from ..utils import word_tokenize, count_ngrams, get_closest_len
from ..compat import Self


@dataclass
class RefStats:
    lens: list[int] = field(default_factory=list)
    ngram_counts: Counter[tuple[str, ...]] = field(default_factory=Counter)

    @classmethod
    def compute(cls, refs: Sequence[str]) -> Self:
        stats = cls()

        for ref in refs:
            tokens = word_tokenize(ref)
            stats.lens.append(len(tokens))
            ngram_counts = count_ngrams(tokens, 1, 4)
            if not stats.ngram_counts:
                stats.ngram_counts = ngram_counts
            else:
                for ngram, count in ngram_counts.items():
                    stats.ngram_counts[ngram] = max(stats.ngram_counts[ngram], count)

        return stats


@dataclass
class HypothesisStats:
    hyp_len: int
    ref_len: int
    correct: list[int]
    total: list[int]

    @classmethod
    def compute(cls, sentence: str, ref_stats: RefStats) -> Self:
        tokens = word_tokenize(sentence)
        ngram_counts = count_ngrams(tokens, min_order=1, max_order=4)
        ref_len = get_closest_len(len(tokens), ref_stats.lens)

        correct = [0] * 4
        total = [0] * 4
        for ngram, count in ngram_counts.items():
            i = len(ngram) - 1
            total[i] += count
            if ngram in ref_stats.ngram_counts:
                correct[i] += min(count, ref_stats.ngram_counts[ngram])

        return cls(
            hyp_len=len(tokens),
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
            correct=list(map(operator.add, self.correct, other.correct)),
            total=list(map(operator.add, self.total, other.total)),
        )

    def __radd__(self, other) -> Self:
        return self + other
