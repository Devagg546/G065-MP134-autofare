import hashlib
import pandas as pd
import yaml


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_train():
    params = load_params()
    return pd.read_csv(params["data"]["train_path"])


def sha256_of(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
