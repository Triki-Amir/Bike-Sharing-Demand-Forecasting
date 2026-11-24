import numpy as np
import pandas as pd

from bikecast.zip_model import FEATURES, zip_fit_predict, zip_fit_predict_flag, zip_fit_predict_info


def _frame(n: int, seed: int = 0, p_zero: float = 0.4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    is_weekend = (np.arange(n) % 7 >= 5).astype(int)
    temp = 15 + 10 * np.sin(np.arange(n) / 58)
    rain = rng.gamma(1, 2, n) * (rng.random(n) < 0.3)
    lam = np.exp(0.5 + 0.4 * is_weekend + 0.03 * temp - 0.05 * rain)
    y = np.where(rng.random(n) < p_zero, 0, rng.poisson(lam))
    return pd.DataFrame({"y": y.astype(float), "is_weekend": is_weekend, "temp_mean": temp,
                         "precip_mm": rain})


def test_zip_fits_and_predicts_plausible_values():
    df = _frame(400)
    train, future = df.iloc[:393], df.iloc[393:]
    pred, used_fallback = zip_fit_predict_info(train, future)
    assert not used_fallback
    assert pred.shape == (7,) and (pred >= 0).all()
    assert abs(pred.mean() - train.y.mean()) < 1.5


def test_zip_weekend_effect_is_learned():
    df = _frame(500, seed=3, p_zero=0.2)
    train = df.iloc[:490]
    future = pd.DataFrame({"is_weekend": [0, 1], "temp_mean": [15.0, 15.0], "precip_mm": [0.0, 0.0]})
    pred = zip_fit_predict(train, future)
    assert pred[1] > pred[0]


def test_zip_falls_back_when_too_few_active_days():
    train = pd.DataFrame({"y": [0.0] * 50 + [1.0] * 3, **{f: 0.0 for f in FEATURES}})
    future = pd.DataFrame({f: [0.0] * 7 for f in FEATURES})
    pred, used = zip_fit_predict_info(train, future)
    assert used and np.allclose(pred, train.y.iloc[-56:].mean())


def test_zip_falls_back_on_missing_features():
    df = _frame(200)
    df.loc[5, "temp_mean"] = np.nan
    pred, flag = zip_fit_predict_flag(df.iloc[:193], df.iloc[193:])
    assert (flag == 1).all() and np.isfinite(pred).all()
