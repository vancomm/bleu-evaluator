import math
import logging
import operator
import functools
import itertools
from collections import Counter
from dataclasses import dataclass, field
from collections.abc import Sequence

from .utils import (
    floor_log,
    sent_tokenize,
    word_tokenize,
    extract_ngrams,
    get_closest_len,
)
from .compat import Self


logger = logging.getLogger(__name__)


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
            correct=list(map(operator.add, self.correct, other.correct)),
            total=list(map(operator.add, self.total, other.total)),
        )

    def __radd__(self, other) -> Self:
        return self + other


@dataclass
class BLEUScore:
    score: float
    precisions: list[float]
    correct: list[int]
    total: list[int]
    bp: float
    hyp_len: int
    ref_len: int

    def __post_init__(self) -> None:
        precisions_str = "/".join(f"{p:.1f}" for p in self.precisions)
        ratio = self.hyp_len / self.ref_len
        self.verbose = (
            f"{precisions_str} (BP = {self.bp:.3f}, "
            f"ratio = {ratio:.3f}, hyp_len = {self.hyp_len}, ref_len = {self.ref_len})"
        )

    def format(self, *, width: int = 2, verbose: bool = False) -> str:
        text = f"BLEU = {self.score:.{width}f}"
        if verbose:
            text = "%s %s" % (text, self.verbose)
        return text


class BLEU:
    ref_cache: list[RefStats]

    def __init__(self, references: list[str]) -> None:
        assert (
            len(set(map(len, references))) == 1
        ), "all references must have the same number of sentences"

        self.ref_cache = list(
            map(RefStats.compute, zip(*map(sent_tokenize, references)))
        )

    @staticmethod
    def _compute_bleu(stats: HypothesisStats) -> BLEUScore:
        precisions = [float()] * 4
        for n in range(0, 4):
            if stats.total[n] == 0:
                continue
            precisions[n] = 100.0 * stats.correct[n] / stats.total[n]

        if stats.hyp_len > stats.ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - stats.ref_len / stats.hyp_len)

        score = bp * math.exp(sum(floor_log(p) for p in precisions) / 4)

        return BLEUScore(
            score=score,
            precisions=precisions,
            correct=stats.correct,
            total=stats.total,
            bp=bp,
            hyp_len=stats.hyp_len,
            ref_len=stats.ref_len,
        )

    def corpus_score(self, hypothesis: str) -> BLEUScore:
        sentences = sent_tokenize(hypothesis)
        assert len(sentences) == len(
            self.ref_cache
        ), "number of hypothesis sentences must be the same as number of reference sentences"

        stats = functools.reduce(
            operator.add,
            itertools.starmap(
                HypothesisStats.compute,
                zip(sentences, self.ref_cache),
            ),
        )

        return self._compute_bleu(stats)
