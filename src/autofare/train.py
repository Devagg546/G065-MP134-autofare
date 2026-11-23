"""Train v1 on cleaned train.csv, track every run in MLflow, register the best.

Run from the repo root:
    PYTHONPATH=src python -m autofare.train
"""
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from autofare.clean import clean
from autofare.data_io import load_params, sha256_of
from autofare.model import build_pipeline


def feature_columns(features):
    return features["numeric"] + features["categorical"]


def main():
    params = load_params()
    data, cfg_clean = params["data"], params["clean"]
    features, cfg_train = params["features"], params["train"]
    target = data["target"]

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(cfg_train["experiment_name"])

    data_hash = sha256_of(data["train_path"])       # hash of the ORIGINAL file
    train = clean(pd.read_csv(data["train_path"]), cfg_clean, target)
    X, y = train[feature_columns(features)], train[target]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=cfg_train["test_size"],
        random_state=cfg_train["random_state"])

    results = []
    for cand in cfg_train["candidates"]:
        with mlflow.start_run(run_name=f"v1-{cand['name']}") as run:
            pipe = build_pipeline(cand, features, cfg_train["random_state"])
            pipe.fit(X_tr, y_tr)
            mae = mean_absolute_error(y_val, pipe.predict(X_val))

            mlflow.log_param("data_sha256", data_hash)
            mlflow.log_params({f"model_{k}": v for k, v in cand.items()})
            mlflow.log_param("rows_after_cleaning", len(train))
            mlflow.log_metric("mae_val", mae)
            info = mlflow.sklearn.log_model(
                pipe, name="model", input_example=X_tr.head(3),
                serialization_format="cloudpickle")
            print(f"{cand['name']:<18} mae_val={mae:.2f}  run_id={run.info.run_id}")
            results.append((mae, run.info.run_id, info.model_uri, pipe))

    mae, run_id, model_uri, best = min(results, key=lambda r: r[0])
    print(f"\nbest: run_id={run_id}  mae_val={mae:.2f}")

    # Register as version 1 and make it the champion (only the first time).
    name = cfg_train["registered_model_name"]
    client = MlflowClient()
    if client.search_model_versions(f"name='{name}'"):
        print(f"'{name}' already has versions; not registering again.")
    else:
        version = mlflow.register_model(model_uri, name).version
        client.set_registered_model_alias(name, "champion", version)
        print(f"registered {name} version {version} as @champion")

    Path(cfg_train["model_path"]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, cfg_train["model_path"])
    print(f"saved {cfg_train['model_path']}")


if __name__ == "__main__":
    main()