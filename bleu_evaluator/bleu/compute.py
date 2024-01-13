import math
import itertools
import functools
import operator

from .stats import RefStats, HypothesisStats
from .score import BLEUScore
from ..utils import sent_tokenize, floor_log


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
