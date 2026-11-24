import numpy as np
import pandas as pd
import pytest

from bikecast.markov import (
    make_markov_fit_predict,
    markov_fit_predict,
    markov_forecast,
    prediction_interval,
    simulate_paths,
    size_params,
    transition_matrix,
)


def test_rows_sum_to_one():
    P = transition_matrix(np.array([0, 1, 0, 0, 2, 3, 0, 1]))
    assert np.allclose(P.sum(axis=1), 1)


def test_transition_counts_with_laplace_smoothing():
    # transitions: 0->1, 1->0, 0->0, 0->1, 1->1  => n00=1, n01=2, n10=1, n11=1
    P = transition_matrix(np.array([0, 1, 0, 0, 1, 1]))
    assert P[0, 1] == pytest.approx((2 + 1) / (1 + 2 + 2))
    assert P[1, 1] == pytest.approx((1 + 1) / (1 + 1 + 2))


def test_never_active_gives_low_probability():
    _, p = markov_forecast(np.zeros(100), h=3)
    assert (p < 0.05).all()


def test_always_active_forecast_close_to_mean():
    y = np.full(100, 4.0)
    y_hat, p = markov_forecast(y, h=1)
    assert p[0] > 0.95 and abs(y_hat[0] - 4) < 0.3


def test_probability_converges_to_stationary():
    y = np.tile([0, 0, 1], 60)
    _, p = markov_forecast(y, h=50, window=180)
    P = transition_matrix(y)
    stationary = P[0, 1] / (P[0, 1] + P[1, 0])
    assert p[-1] == pytest.approx(stationary, abs=1e-6)


def test_size_params_negbin_when_overdispersed():
    y = np.array([1, 1, 1, 2, 10, 1, 1, 15, 1, 2])
    s = size_params(y)
    assert s["dist"] == "negbin" and s["r"] > 0 and 0 < s["p"] < 1
    assert s["mean_size"] == pytest.approx(y.mean())


def test_size_params_poisson_when_not_overdispersed():
    s = size_params(np.array([3, 3, 3, 0, 3]))
    assert s["dist"] == "poisson" and s["mean_size"] == pytest.approx(3)


def test_window_limits_history():
    y = np.r_[np.full(300, 10.0), np.zeros(90)]
    y_short, _ = markov_forecast(y, h=1, window=90)
    y_long, _ = markov_forecast(y, h=1, window=365)
    assert y_short[0] < 0.05 and y_short[0] < y_long[0]


def test_empty_history_returns_zeros():
    y_hat, p = markov_forecast(np.array([]), h=4)
    assert y_hat.shape == (4,) and not y_hat.any() and not p.any()


def test_simulation_mean_matches_analytic_forecast():
    rng = np.random.default_rng(1)
    y = (rng.random(365) < 0.4) * rng.integers(1, 6, 365)
    y_hat, _ = markov_forecast(y, h=7, window=365)
    paths = simulate_paths(y, h=7, window=365, n_paths=20000, seed=3)
    assert np.allclose(paths.mean(axis=0), y_hat, rtol=0.08, atol=0.05)
    lo, hi = prediction_interval(y, h=7, window=365, n_paths=2000)
    assert (lo <= hi).all() and (lo >= 0).all()


def test_fit_predict_wrappers():
    train = pd.DataFrame({"y": [0, 1, 0, 2, 0, 1, 3, 0]})
    future = pd.DataFrame({"y": [0] * 5})
    assert markov_fit_predict(train, future).shape == (5,)
    y_hat, p = make_markov_fit_predict(90)(train, future)
    assert y_hat.shape == p.shape == (5,) and ((p >= 0) & (p <= 1)).all()
