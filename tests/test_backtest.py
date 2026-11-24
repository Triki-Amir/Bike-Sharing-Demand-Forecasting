import numpy as np
import pandas as pd
import pytest

from bikecast import backtest as B
from bikecast.markov import make_markov_fit_predict


def _df(n_stations: int = 3, start: str = "2026-01-01", end: str = "2026-06-30") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ds = pd.date_range(start, end)
    parts = []
    for i in range(n_stations):
        y = (rng.random(len(ds)) < 0.3 + 0.3 * i) * rng.integers(1, 5, len(ds))
        parts.append(pd.DataFrame({"unique_id": f"s{i}", "ds": ds, "y": y.astype(float),
                                   "segment": ["intermittent", "lumpy", "smooth"][i % 3]}))
    return pd.concat(parts, ignore_index=True)


def test_rolling_cutoffs_match_guide():
    cuts = B.rolling_cutoffs("2026-06-30", h=7, n_windows=4)
    assert [str(c.date()) for c in cuts] == ["2026-06-02", "2026-06-09", "2026-06-16", "2026-06-23"]


def test_backtest_never_uses_future_data():
    df = _df()
    seen = []

    def spy(train, future):
        seen.append((train.ds.max(), future.ds.min(), future.ds.max(), len(future)))
        return np.zeros(len(future))

    cuts = B.rolling_cutoffs("2026-06-30")
    out = B.backtest(df, spy, "Spy", cuts)
    assert len(out) == 3 * 4 * 7
    for train_max, fut_min, fut_max, n in seen:
        assert train_max < fut_min and n == 7 and (fut_max - fut_min).days == 6


def test_backtest_tuple_output_adds_extra_column():
    df = _df()
    out = B.backtest(df, make_markov_fit_predict(90), "Markov", B.rolling_cutoffs("2026-06-30"))
    assert {"Markov", "Markov_p_sale"} <= set(out.columns)
    assert out["Markov_p_sale"].between(0, 1).all()


def test_backtest_parallel_matches_serial():
    df = _df()
    cuts = B.rolling_cutoffs("2026-06-30")
    fp = make_markov_fit_predict(90)
    a = B.backtest(df, fp, "Markov", cuts, n_jobs=1)
    b = B.backtest(df, fp, "Markov", cuts, n_jobs=2)
    pd.testing.assert_frame_equal(a.sort_values(B.KEYS).reset_index(drop=True),
                                  b.sort_values(B.KEYS).reset_index(drop=True))


def test_statsforecast_cv_cutoffs_and_columns():
    df = _df()
    cv = B.statsforecast_cv(df)
    assert {"SeasonalNaive", "CrostonClassic", "TSB", "y", "cutoff"} <= set(cv.columns)
    assert sorted(str(c.date()) for c in cv.cutoff.unique()) == [
        "2026-06-02", "2026-06-09", "2026-06-16", "2026-06-23"]
    # Seasonal naive equals the value 7 days before (first day of each window).
    row = cv[(cv.unique_id == "s0") & (cv.ds == "2026-06-03")].iloc[0]
    lag = df[(df.unique_id == "s0") & (df.ds == "2026-05-27")].y.iloc[0]
    assert row.SeasonalNaive == pytest.approx(lag)


def test_tuning_only_sees_tuning_windows():
    df = _df()
    cuts = B.rolling_cutoffs("2026-06-30")[:2]
    (a, b), scores = B.tune_tsb(df, cuts, grid=(0.1, 0.3))
    assert len(scores) == 4 and (a, b) in {(0.1, 0.1), (0.1, 0.3), (0.3, 0.1), (0.3, 0.3)}
    w, mk = B.tune_markov_window(df, cuts, grid=(30, 90))
    assert w in (30, 90) and len(mk) == 2
    assert B._truncate_for_cutoffs(df, cuts, 7).ds.max() == pd.Timestamp("2026-06-16")


def test_score_summary_table_headline():
    df = _df()
    cuts = B.rolling_cutoffs("2026-06-30")
    cv = B.statsforecast_cv(df)
    zero = B.backtest(df, lambda tr, fu: np.zeros(len(fu)), "ZIP", cuts)
    mk = B.backtest(df, make_markov_fit_predict(90), "Markov", cuts)
    fc = B.combine(cv, zero, mk)
    assert len(fc) == 3 * 28
    res = B.score(fc, df, train_end="2026-06-02")
    assert len(res) == 3 * 5 and res.mae.ge(0).all()
    assert res.loc[res.model == "Markov", "zero_day_accuracy"].between(0, 1).all()
    summary = B.summarize(res)
    table = B.mase_table(summary)
    assert list(table.columns[:1]) == ["stations"] and "Markov" in table.columns
    long = B.to_long(fc)
    assert len(long) == len(fc) * 5 and long.loc[long.model == "TSB", "p_sale"].isna().all()
    head = B.headline(summary)
    for seg in head.values():
        assert seg["best_model"] != "SeasonalNaive"
