import numpy as np
import pytest

from bikecast import metrics as M


def test_perfect_forecast():
    y = np.array([0, 2, 0, 5])
    assert M.mae(y, y) == 0 and M.bias_pct(y, y) == 0


def test_plus_one_bias_is_n_over_sum():
    y = np.array([0, 2, 0, 5, 1])
    assert M.bias_pct(y, y + 1) == pytest.approx(len(y) / y.sum())


def test_mae_value():
    assert M.mae([1, 2, 3], [2, 2, 1]) == pytest.approx(1.0)


def test_bias_sign_and_all_zero_actuals():
    assert M.bias_pct([1, 1], [0, 0]) == pytest.approx(-1)
    assert np.isfinite(M.bias_pct([0, 0], [1, 1]))


def test_mase_scale_is_seasonal_naive_in_sample():
    y_train = np.array([1, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 8, 9])  # lag-7 diffs all = 2
    assert M.mase([1, 1], [2, 3], y_train, m=7) == pytest.approx(1.5 / 2)


def test_mase_nan_when_scale_zero_or_short():
    assert np.isnan(M.mase([1], [1], np.full(20, 3.0)))
    assert np.isnan(M.mase([1], [1], np.array([1, 2, 3])))


def test_zero_day_accuracy():
    assert M.zero_day_accuracy([0, 3, 0, 1], [0.1, 0.9, 0.6, 0.4]) == pytest.approx(0.5)
