import logging
from collections import Counter
from collections.abc import Sequence
from fractions import Fraction
import math
from typing import Callable

import numpy as np
from numpy.typing import NDArray
import nltk.collocations as colloc
from nltk.util import ngrams

from .parse import Sentence, Corpus, Corpora


logger = logging.getLogger(__name__)


def modified_ngram_precision_sentence(
    references: Corpus,
    candidate: Sentence,
    *,
    n: int,
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


def modified_ngram_precision(
    references: Corpus,
    candidate_corpus: Corpus,
    *,
    n: int,
) -> Fraction:
    if n == 1:
        ref_counters = list(map(Counter, references))
        cand_counters = list(map(Counter, candidate_corpus))
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
            sum(map(len, candidate_corpus)),
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
    for candidate in candidate_corpus:
        cand_finder = finder_cls.from_words(candidate)
        cand_ngrams = list(ngrams(candidate, n))
        sum_clipped += sum(
            min(
                cand_finder.ngram_fd[ngram],
                max(ref_finder.ngram_fd[ngram] for ref_finder in ref_finders),
            )
            for ngram in cand_ngrams
        )

    return Fraction(sum_clipped, sum(map(len, candidate_corpus)))


def find_best_matches_idx(
    sources: NDArray[np.int_], targets: NDArray[np.int_]
) -> NDArray[np.int_]:
    source_matrix = np.tile(sources[:, np.newaxis], (1, len(targets)))
    return np.absolute(source_matrix - targets.T).argmin(axis=1)


def bleu_score(
    references: Corpus,
    candidate: Corpus,
    *,
    n: int,
    weights: Sequence[float] | None = None,
    smoothing_function: Callable[[Fraction], float] | None = None,
) -> float:
    logger.info(f"{references = }")
    logger.info(f"{candidate = }")

    if n < 1 or n > 4:
        raise ValueError("n must be in range [1; 4]")

    if weights and len(weights) != n:
        raise ValueError(f"weights array must be of length {n = }")

    if not weights:
        weights = [1 / n] * n

    logger.info(f"{weights = }")

    ref_lens = np.array(list(map(len, references)))
    cand_lens = np.array(list(map(len, candidate)))
    best_len_matches_idx = find_best_matches_idx(cand_lens, ref_lens)
    r = ref_lens[best_len_matches_idx].sum()
    c = sum(map(len, candidate))

    brevity_penalty = 1 if c > r else np.e ** (1 - r / c)

    logger.info(f"{r = }, {c = }, {brevity_penalty = }")

    ps = [modified_ngram_precision(references, candidate, n=i) for i in range(1, n + 1)]

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
