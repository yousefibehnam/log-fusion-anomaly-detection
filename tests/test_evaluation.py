import pytest

from logfusion.evaluation import (
    detection,
    grouping_accuracy,
    mean_reciprocal_rank,
    rank_of,
    top_k_accuracy,
)


def test_detection_counts_and_scores():
    d = detection([1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0])
    assert (d.tp, d.fp, d.fn, d.tn) == (2, 1, 1, 3)
    assert d.precision == pytest.approx(2 / 3)
    assert d.recall == pytest.approx(2 / 3)
    assert d.f1 == pytest.approx(2 / 3)


def test_detection_without_positive_predictions():
    d = detection([1, 0], [0, 0])
    assert (d.precision, d.recall, d.f1) == (0.0, 0.0, 0.0)


def test_detection_shape_mismatch():
    with pytest.raises(ValueError):
        detection([1, 0], [1])


def test_ranking_measures():
    assert rank_of(["b", "a", "c"], "a") == 2
    assert rank_of(["b", "c"], "a") == 3
    ranks = [1, 2, 4]
    assert top_k_accuracy(ranks, 1) == pytest.approx(1 / 3)
    assert top_k_accuracy(ranks, 3) == pytest.approx(2 / 3)
    assert mean_reciprocal_rank(ranks) == pytest.approx((1 + 0.5 + 0.25) / 3)


def test_grouping_accuracy():
    # true groups {0,1} {2} {3}; predicted {0,1} {2,3}: only messages 0 and 1 are correct
    assert grouping_accuracy(["x", "x", "y", "y"], ["A", "A", "B", "C"]) == pytest.approx(0.5)
    assert grouping_accuracy(["p", "q"], ["A", "B"]) == 1.0
