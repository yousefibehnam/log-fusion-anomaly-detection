import numpy as np
import pytest

from logfusion.detectors.pca import PCADetector, q_statistic
from logfusion.detectors.robust import MAD_TO_SIGMA, RobustZ


def low_rank_data(rng, n=2000, noise=0.05):
    """Points near a 2-D plane inside 6-D space."""
    basis = rng.normal(size=(2, 6))
    return rng.normal(size=(n, 2)) @ basis + noise * rng.normal(size=(n, 6)), basis


def test_pca_flags_points_off_the_normal_subspace():
    rng = np.random.default_rng(1)
    x, basis = low_rank_data(rng)
    detector = PCADetector(variance=0.95, alpha=0.001, tfidf=False).fit(x)
    assert detector.n_components_ == 2

    on_plane = rng.normal(size=(500, 2)) @ basis + 0.05 * rng.normal(size=(500, 6))
    normal_direction = np.linalg.svd(basis)[2][-1]  # orthogonal to the plane
    off_plane = on_plane + 2.0 * normal_direction
    assert detector.predict(on_plane).mean() < 0.02
    assert detector.predict(off_plane).mean() == 1.0


def test_q_statistic_matches_monte_carlo_quantile():
    # SPE of Gaussian residuals with variances lam is sum(lam_i * z_i^2).
    lam = np.array([0.5, 0.3, 0.2, 0.1])
    rng = np.random.default_rng(7)
    spe = (lam * rng.normal(size=(400_000, 4)) ** 2).sum(axis=1)
    for alpha in (0.05, 0.01):
        assert q_statistic(lam, alpha) == pytest.approx(np.quantile(spe, 1 - alpha), rel=0.05)


def test_q_statistic_without_residual_variance_is_zero():
    assert q_statistic(np.array([]), 0.01) == 0.0
    assert q_statistic(np.zeros(3), 0.01) == 0.0


def test_tfidf_gives_zero_weight_to_columns_present_everywhere():
    x = np.array([[1, 0], [2, 1], [3, 0], [4, 0]], dtype=float)
    detector = PCADetector(tfidf=True).fit(x)
    np.testing.assert_allclose(detector.idf_, [0.0, np.log(4.0)])


def test_pca_needs_two_samples():
    with pytest.raises(ValueError):
        PCADetector().fit(np.ones((1, 3)))


def test_robust_z_hand_computed():
    # column 0: median 3, |dev| = 2 1 0 1 2 -> MAD 1 -> scale 1.4826
    # column 1: constant -> MAD 0 -> scale floored at 1.0
    x = np.array([[1, 5], [2, 5], [3, 5], [4, 5], [5, 5]], dtype=float)
    model = RobustZ(min_scale=1.0).fit(x)
    np.testing.assert_allclose(model.median_, [3, 5])
    np.testing.assert_allclose(model.scale_, [MAD_TO_SIGMA, 1.0])
    z = model.z(np.array([[3 + 2 * MAD_TO_SIGMA, 9.0]]))
    np.testing.assert_allclose(z, [[2.0, 4.0]])
    assert model.score(np.array([[3, 9.0]]))[0] == pytest.approx(4.0)
    assert model.predict(np.array([[3, 8.4], [3, 8.6]])).tolist() == [0, 1]


def test_robust_z_rejects_empty_input():
    with pytest.raises(ValueError):
        RobustZ().fit(np.empty((0, 2)))
