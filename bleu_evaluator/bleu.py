import logging
from collections import Counter
from collections.abc import Sequence
from fractions import Fraction
import math
from typing import Callable

import numpy as np
import nltk.collocations as colloc
from nltk.util import ngrams


logger = logging.getLogger(__name__)


def modified_ngram_precision(
    references: Sequence[Sequence[str]], candidate: Sequence[str], *, n: int
) -> Fraction:
    if n == 1:
        cand_counter = Counter(candidate)
        ref_counters = list(map(Counter, references))
        return Fraction(
            sum(
                min(
                    cand_counter[word],
                    max(ref_counter.get(word, 0) for ref_counter in ref_counters),
                )
                for word in cand_counter
            ),
            len(candidate),
        )

    if n == 2:
        finder_cls = colloc.BigramCollocationFinder
    elif n == 3:
        finder_cls = colloc.TrigramCollocationFinder
    elif n == 4:
        finder_cls = colloc.QuadgramCollocationFinder
    else:
        raise ValueError("n must be in range [1; 4]")

    cand_finder = finder_cls.from_words(candidate)
    cand_ngrams = list(ngrams(candidate, n))
    ref_finders = list(map(finder_cls.from_words, references))
    return Fraction(
        sum(
            min(
                cand_finder.ngram_fd[ngram],
                max(ref_finder.ngram_fd[ngram] for ref_finder in ref_finders),
            )
            for ngram in cand_ngrams
        ),
        len(cand_ngrams),
    )


def modified_precision(
    references: Sequence[Sequence[str]], candidates: Sequence[Sequence[str]], *, n: int
) -> Fraction:
    if n == 1:
        ref_counters = list(map(Counter, references))
        cand_counters = list(map(Counter, candidates))
        return Fraction(
            sum(
                sum(
                    min(
                        cand_counter[word],
                        max(ref_counter.get(word, 0) for ref_counter in ref_counters),
                    )
                    for word in cand_counter
                )
                for cand_counter in cand_counters
            ),
            sum(map(len, candidates)),
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
    for candidate in candidates:
        cand_finder = finder_cls.from_words(candidate)
        cand_ngrams = list(ngrams(candidate, n))
        sum_clipped += sum(
            min(
                cand_finder.ngram_fd[ngram],
                max(ref_finder.ngram_fd[ngram] for ref_finder in ref_finders),
            )
            for ngram in cand_ngrams
        )

    return Fraction(sum_clipped, sum(map(len, candidates)))


def bleu_score(
    references: Sequence[Sequence[str]],
    candidates: Sequence[Sequence[str]],
    *,
    n: int,
    weights: Sequence[float] | None = None,
    smoothing_function: Callable[[Fraction], float] | None = None,
) -> float:
    logger.info(f"{references = }")
    logger.info(f"{candidates = }")

    if n < 1 or n > 4:
        raise ValueError("n must be in range [1; 4]")

    if weights and len(weights) != n:
        raise ValueError(f"weights array must be of length {n = }")

    if not weights:
        weights = [1 / n] * n

    logger.info(f"{weights = }")

    ref_lens = np.array(list(map(len, references)))
    cand_lens = np.array(list(map(len, candidates)))
    cand_lens = np.tile(cand_lens, (cand_lens.shape[0], ref_lens.shape[0]))
    r = ref_lens[np.absolute(np.subtract(cand_lens, ref_lens)).argmin(axis=1)].sum()
    c = sum(map(len, candidates))

    if c > r:
        brevity_penalty = 1
    else:
        brevity_penalty = np.e ** (1 - r / c)

    logger.info(f"{r = }, {c = }, {brevity_penalty = }")

    ps = [modified_precision(references, candidates, n=i) for i in range(1, n + 1)]

    logger.info("ps = [%s]" % ", ".join(map(str, ps)))

    if not all(ps):
        if not smoothing_function:
            raise ValueError(
                "modified precision score of zero detected with no smoothing function"
            )
        logging.warning("modified precision score of zero detected - smoothing enabled")
        ps = map(smoothing_function, ps)

    s = math.fsum(w * math.log(p) for w, p in zip(weights, ps))
    score = brevity_penalty * math.exp(s)

    logger.info(f"{score = }")

    return score
