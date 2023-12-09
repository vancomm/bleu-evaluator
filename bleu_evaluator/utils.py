from collections.abc import Iterable
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray


T = TypeVar("T")


def flattened(__iterable: Iterable[Iterable[T]]) -> Iterable[T]:
    return (item for sublist in __iterable for item in sublist)


def find_best_matches_idx(
    sources: NDArray[np.int_], targets: NDArray[np.int_]
) -> NDArray[np.int_]:
    """
    Find indices of values in `targets` array that are closest to values in `sources`
    array.

    :param sources: `np.array` of shape `(n,)`
    :param targets: `np.array` of shape `(m,)`
    :return: `np.array` of shape `(n,)` containing indices of `targets` array
    """
    if sources.ndim != 1 or targets.ndim != 1:
        raise ValueError("inputs must be one-dimensional arrays")

    source_matrix = np.tile(sources[:, np.newaxis], (1, len(targets)))
    return np.absolute(source_matrix - targets.T).argmin(axis=1)
