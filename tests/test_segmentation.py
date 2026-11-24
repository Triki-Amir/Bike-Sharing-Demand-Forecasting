import numpy as np
import pandas as pd
import pytest

from bikecast.segmentation import adi_cv2, classify, segment_stations


def test_every_day_constant_is_smooth():
    adi, cv2 = adi_cv2(np.array([3, 3, 3, 3]))
    assert adi == 1 and cv2 == 0 and classify(adi, cv2) == "smooth"


def test_rare_sales_is_intermittent():
    adi, cv2 = adi_cv2(np.array([0, 0, 2, 0, 0, 2, 0, 0]))
    assert adi == 4 and classify(adi, cv2) == "intermittent"


@pytest.mark.parametrize(
    "adi,cv2,expected",
    [(1.0, 0.1, "smooth"), (1.0, 0.9, "erratic"), (2.0, 0.1, "intermittent"), (2.0, 0.9, "lumpy"),
     (1.32, 0.1, "intermittent"), (1.0, 0.49, "erratic")],
)
def test_classify_quadrants_and_boundaries(adi, cv2, expected):
    assert classify(adi, cv2) == expected


def test_too_few_nonzero_returns_inf():
    adi, cv2 = adi_cv2(np.array([0, 0, 5, 0]))
    assert adi == float("inf") and np.isnan(cv2)


def test_cv2_uses_sample_std():
    y = np.array([1, 0, 3, 0, 5])
    _, cv2 = adi_cv2(y)
    nz = np.array([1, 3, 5])
    assert cv2 == pytest.approx((nz.std(ddof=1) / nz.mean()) ** 2)


def test_segment_stations_uses_training_period_only_and_min_days():
    dates = pd.date_range("2026-05-01", "2026-06-30")
    busy = pd.DataFrame({"station_id": "A", "date": dates, "trips": 5})
    # Station B only becomes active after the training end: must not be segmented.
    late = pd.DataFrame({"station_id": "B", "date": dates,
                         "trips": (dates > "2026-06-02").astype(int) * 3})
    out = segment_stations(pd.concat([busy, late]), train_end="2026-06-02", min_nonzero=10)
    assert list(out.station_id) == ["A"]
    row = out.iloc[0]
    assert row.n_days == len(dates[dates <= "2026-06-02"])
    assert row.segment == "smooth" and row.pct_zero_days == 0


def test_synthetic_segments_match_ground_truth(layers, manifest):
    seg = layers["segs"].set_index("station_id")["segment"].to_dict()
    assert seg == manifest["stations"]
