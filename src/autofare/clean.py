"""Cleaning for MP134. Fixes the data problems found by data_checks.py.

Every fix is safe to run on already-clean data (like later.csv): it only
changes rows that are actually wrong.
"""
import pandas as pd


def clean_booking(series):
    """'APP', ' app', 'app.' -> 'app'; 'str.' -> 'street'; 'sta.' -> 'stand'."""
    s = series.str.strip().str.lower().str.rstrip(".")
    return s.replace({"str": "street", "sta": "stand"})


def clean(df, cfg, target=None):
    """Return a cleaned copy of df. cfg is the 'clean' section of params.yaml.
    Pass target only when the target column is present (train / later).
    """
    df = df.copy()

    # spelling_variants: one spelling per booking type
    df["booking"] = clean_booking(df["booking"])

    # unit_mixup: Market recorded metres, convert back to km
    col, zone = cfg["unit_column"], cfg["unit_zone"]
    wrong = (df["zone"] == zone) & (df[col] > cfg["unit_fix_above"])
    df.loc[wrong, col] = df.loc[wrong, col] / cfg["unit_factor"]

    # target_typos: fares typed with an extra zero
    if target is not None and target in df.columns:
        typo = df[target] > cfg["target_typo_above"]
        df.loc[typo, target] = df.loc[typo, target] / cfg["target_typo_factor"]

    # leaking_column: never available at prediction time, so drop it
    df = df.drop(columns=[cfg["leak_column"]], errors="ignore")

    # duplicate_rows: none found, but dropping is harmless
    df = df.drop_duplicates()
    return df.reset_index(drop=True)