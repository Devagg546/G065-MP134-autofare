import pandas as pd


def check_leaking_column(train, holdout, target):
    """A column is suspicious if it is missing from the holdout file
    and is very close to the target."""
    missing_in_holdout = [c for c in train.columns
                          if c not in holdout.columns and c != target]
    results = {}
    for col in missing_in_holdout:
        if pd.api.types.is_numeric_dtype(train[col]):
            results[col] = train[col].corr(train[target])
    return results


def check_duplicate_rows(train):
    """Number of extra copies, i.e. rows you would delete."""
    return int(train.duplicated().sum())


# TODO (your homework): unit_mixup, sentinel_value, stuck_sensor,
# target_typos, spelling_variants
