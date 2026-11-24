import numpy as np
import pandas as pd
import pytest

from autofare.data_io import load_params
from autofare.model import build_pipeline, make_regressor


@pytest.mark.parametrize("candidate", load_params()["train"]["candidates"],
                         ids=lambda candidate: candidate["name"])
def test_pipeline_fits_and_predicts_unseen_categories(candidate):
    features = load_params()["features"]
    train = pd.DataFrame({
        "distance_km": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "wait_min": [0, 2, 1, 3, 0, 4],
        "passengers": [1, 2, 1, 3, 2, 1],
        "hour": [8, 9, 12, 14, 18, 20],
        "luggage_bags": [0, 1, 0, 2, 1, 0],
        "fuel_price_rs": [100.0] * 6,
        "booking": ["app", "street", "stand", "app", "street", "stand"],
        "zone": ["Market", "Station", "Market", "Station", "Market", "Station"],
    })
    pipeline = build_pipeline(candidate, features)
    pipeline.fit(train, [40.0, 60.0, 80.0, 100.0, 120.0, 140.0])
    holdout = train.iloc[:2].copy()
    holdout["booking"] = "new_booking"
    holdout["zone"] = "new_zone"

    predictions = pipeline.predict(holdout)

    assert predictions.shape == (2,)
    assert np.isfinite(predictions).all()


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="unknown model: unsupported"):
        make_regressor({"model": "unsupported"}, random_state=42)
