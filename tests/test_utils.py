import numpy as np
import pytest

from bleu_evaluator import utils


@pytest.mark.parametrize(
    "source, targets, expected",
    [
        (3, [4, 5, 6], 4),
        (7, [4, 5, 6], 6),
        (10, [0, 30, 100, -3], 0),
    ],
)
def test_get_closest_len(source: int, targets: list[int], expected: int) -> None:
    actual = utils.get_closest_len(source, targets)
    assert np.array_equal(actual, expected)
