import logging
from collections import Counter
from collections.abc import Sequence
from fractions import Fraction
import math
from typing import Callable

import numpy as np
import nltk.collocations as colloc
from nltk.util import ngrams

from .utils import find_best_matches_idx


logger = logging.getLogger(__name__)


def modified_ngram_precision_sentence(
    references: list[list[str]],
    hypothesis: list[str],
    *,
    n: int,
) -> Fraction:
    if n == 1:
        hyp_counter = Counter(hypothesis)
        ref_counters = list(map(Counter, references))
        return Fraction(
            sum(
                min(
                    hyp_counter[word],
                    max(ref_counter.get(word, 0) for ref_counter in ref_counters),
                )
                for word in hyp_counter
            ),
            len(hypothesis),
        )

    if n == 2:
        finder_cls = colloc.BigramCollocationFinder
    elif n == 3:
        finder_cls = colloc.TrigramCollocationFinder
    elif n == 4:
        finder_cls = colloc.QuadgramCollocationFinder
    else:
        raise ValueError("n must be in range [1; 4]")

    finder = finder_cls.from_words(hypothesis)
    hyp_ngrams = list(ngrams(hypothesis, n))
    ref_finders = list(map(finder_cls.from_words, references))
    return Fraction(
        sum(
            min(
                finder.ngram_fd[ngram],
                max(ref_finder.ngram_fd[ngram] for ref_finder in ref_finders),
            )
            for ngram in hyp_ngrams
        ),
        len(hyp_ngrams),
    )


def modified_ngram_precision(
    references: list[list[str]],
    hypothesis: list[list[str]],
    *,
    n: int,
) -> Fraction:
    if n == 1:
        ref_counters = list(map(Counter, references))
        hyp_counters = list(map(Counter, hypothesis))
        return Fraction(
            sum(
                sum(
                    min(
                        hyp_counter[word],
                        max(ref_counter.get(word, 0) for ref_counter in ref_counters),
                    )
                    for word in hyp_counter
                )
                for hyp_counter in hyp_counters
            ),
            sum(map(len, hypothesis)),
        )

    if n == 2:
        finder_cls = colloc.BigramCollocationFinder
    elif n == 3:
        finder_cls = colloc.TrigramCollocationFinder
    elif n == 4:
        finder_cls = colloc.QuadgramCollocationFinder
    else:
        raise ValueError("n must be in range [1; 4]")

    ref_finders = list(map(finder_cls.from_words, references))
    sum_clipped = 0
    for sentence in hypothesis:
        finder = finder_cls.from_words(sentence)
        sent_ngrams = list(ngrams(sentence, n))
        sum_clipped += sum(
            min(
                finder.ngram_fd[ngram],
                max(ref_finder.ngram_fd[ngram] for ref_finder in ref_finders),
            )
            for ngram in sent_ngrams
        )

    return Fraction(sum_clipped, sum(map(len, hypothesis)))


def bleu_score(
    references: list[list[str]],
    hypothesis: list[list[str]],
    *,
    n: int,
    weights: Sequence[float] | None = None,
    smoothing_function: Callable[[Fraction], float] | None = None,
) -> float:
    logger.info(f"{references = }")
    logger.info(f"{hypothesis = }")

    if n < 1 or n > 4:
        raise ValueError("n must be in range [1; 4]")

    if weights and len(weights) != n:
        raise ValueError(f"weights array must be of length {n = }")

    if not weights:
        weights = [1 / n] * n

    logger.info(f"{weights = }")

    ref_lens = np.array(list(map(len, references)))
    hyp_lens = np.array(list(map(len, hypothesis)))
    best_len_matches_idx = find_best_matches_idx(hyp_lens, ref_lens)
    r = ref_lens[best_len_matches_idx].sum()
    c = sum(map(len, hypothesis))

    brevity_penalty = 1 if c > r else np.e ** (1 - r / c)

    logger.info(f"{r = }, {c = }, {brevity_penalty = }")

    ps = [
        modified_ngram_precision(references, hypothesis, n=i) for i in range(1, n + 1)
    ]

    logger.info("ps = [%s]" % ", ".join(map(str, ps)))

    if not all(ps):
        if not smoothing_function:
            raise ValueError(
                "precision score of zero detected with no smoothing function"
            )
        logging.warning(
            "precision score of zero detected - applying a smoothing function"
        )
        ps = map(smoothing_function, ps)

    s = math.fsum(w * math.log(p) for w, p in zip(weights, ps))
    score = brevity_penalty * math.exp(s)

    logger.info(f"{score = }")

    return score
