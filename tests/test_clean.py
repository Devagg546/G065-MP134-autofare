import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from autofare.clean import clean, clean_booking
from autofare.data_io import load_params


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("APP", "app"), (" app", "app"), ("app.", "app"),
     ("str.", "street"), ("sta.", "stand")],
)
def test_booking_spellings(raw, expected):
    assert clean_booking(pd.Series([raw])).iloc[0] == expected


def test_clean_corrects_units_typos_leaks_and_duplicates():
    cfg = load_params()["clean"]
    raw = pd.DataFrame({
        "booking": ["APP", "str.", "sta.", "APP"],
        "zone": ["Market", "Market", "Station", "Market"],
        "distance_km": [2500.0, 2.5, 150.0, 2500.0],
        "fare_rs": [1500.0, 150.0, 500.0, 1500.0],
        "fare_paid_receipt_rs": [1500.0, 150.0, 500.0, 1500.0],
    })
    original = raw.copy(deep=True)

    result = clean(raw, cfg, target="fare_rs")

    assert result["booking"].tolist() == ["app", "street", "stand"]
    assert result["distance_km"].tolist() == [2.5, 2.5, 150.0]
    assert result["fare_rs"].tolist() == [150.0, 150.0, 500.0]
    assert "fare_paid_receipt_rs" not in result.columns
    assert result.index.tolist() == [0, 1, 2]
    assert_frame_equal(raw, original)
    assert_frame_equal(clean(result, cfg, target="fare_rs"), result)


def test_clean_holdout_without_target_or_receipt():
    raw = pd.DataFrame({
        "booking": [" APP "], "zone": ["Market"], "distance_km": [3000.0],
    })

    result = clean(raw, load_params()["clean"])

    assert result["booking"].tolist() == ["app"]
    assert result["distance_km"].tolist() == [3.0]
    assert "fare_rs" not in result.columns
