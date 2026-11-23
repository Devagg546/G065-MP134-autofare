"""Builds the scikit-learn pipeline: one-hot encode text columns, then a regressor."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def make_regressor(candidate, random_state):
    kind = candidate["model"]
    if kind == "linear":
        return LinearRegression()
    if kind == "random_forest":
        return RandomForestRegressor(
            n_estimators=candidate["n_estimators"],
            max_depth=candidate["max_depth"],
            random_state=random_state,
        )
    if kind == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=candidate["n_estimators"],
            learning_rate=candidate["learning_rate"],
            max_depth=candidate["max_depth"],
            random_state=random_state,
        )
    raise ValueError(f"unknown model: {kind}")


def build_pipeline(candidate, features, random_state=42):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), features["categorical"]),
        ("num", "passthrough", features["numeric"]),
    ])
    return Pipeline([("pre", pre), ("reg", make_regressor(candidate, random_state))])