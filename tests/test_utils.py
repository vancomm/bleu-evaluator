import numpy as np
import pytest

from bleu_evaluator import utils


@pytest.mark.parametrize(
    "sources, targets, expected",
    [
        ([3], [4, 5, 6], [0]),
        ([3, 5], [4, 5, 6], [0, 1]),
        ([10, -4, 3], [0, 30, 100, -3], [0, 3, 0]),
    ],
)
def test_find_best_matches_idx(
    sources: list[int], targets: list[int], expected: list[int]
) -> None:
    actual = utils.find_best_matches_idx(np.array(sources), np.array(targets))
    assert np.array_equal(actual, expected)
