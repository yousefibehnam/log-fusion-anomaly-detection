import pytest

from logfusion.detectors.rootcause import anomaly_walk, rank_by_score, rank_by_walk

CHAIN = [("frontend", "checkout"), ("checkout", "payment")]


def test_symptoms_upstream_mislead_score_ranking_but_not_the_walk():
    # payment is the root cause, but its callers look even more anomalous.
    scores = {"frontend": 3.0, "checkout": 2.5, "payment": 2.0}
    assert rank_by_score(scores)[0] == "frontend"
    assert rank_by_walk(CHAIN, scores)[0] == "payment"


def test_an_isolated_anomaly_is_ranked_first():
    scores = {"frontend": 0.2, "checkout": 0.1, "payment": 0.1, "cache": 5.0}
    assert rank_by_walk(CHAIN, scores)[0] == "cache"


def test_walk_is_a_probability_distribution():
    p = anomaly_walk(CHAIN, {"frontend": 1.0, "checkout": 2.0, "payment": 0.5})
    assert sum(p.values()) == pytest.approx(1.0)
    assert all(v >= 0 for v in p.values())


def test_zero_scores_fall_back_to_uniform_restart():
    p = anomaly_walk(CHAIN, {"frontend": 0.0, "checkout": 0.0, "payment": 0.0})
    assert p == pytest.approx({"frontend": 1 / 3, "checkout": 1 / 3, "payment": 1 / 3})


def test_ties_are_broken_by_name():
    assert rank_by_score({"b": 1.0, "a": 1.0}) == ["a", "b"]


def test_invalid_damping():
    with pytest.raises(ValueError):
        anomaly_walk(CHAIN, {"frontend": 1.0}, damping=1.0)
