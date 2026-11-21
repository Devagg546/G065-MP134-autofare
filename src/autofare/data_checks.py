"""The seven data checks for MP134 (Auto-Rickshaw Fare, Patna).

Run from the repo root:
    PYTHONPATH=src python -m autofare.data_checks
"""
import pandas as pd

from autofare.data_io import load_params

# Exact unit conversions the handbook allows for unit_mixup.
UNIT_FACTORS = [10, 12, 60, 100, 1000]
# Special numbers people type instead of leaving a blank.
SENTINEL_CANDIDATES = [-1, -9, -99, -999, -9999, 99, 999, 9999]


def numeric_columns(df, exclude=()):
    return [c for c in df.select_dtypes("number").columns if c not in exclude]


# 1. leaking_column ---------------------------------------------------------
def check_leaking_column(train, holdout, target):
    """Columns missing from the holdout file, and how close they are to the
    target. A leak is missing at prediction time AND almost equals the target.
    """
    results = {}
    for col in train.columns:
        if col in holdout.columns or col == target:
            continue
        if not pd.api.types.is_numeric_dtype(train[col]):
            continue
        diff = (train[col] - train[target]).abs()
        results[col] = {
            "correlation": round(float(train[col].corr(train[target])), 3),
            "median_abs_diff": float(diff.median()),
            "share_within_5_rs": round(float((diff <= 5).mean()), 3),
        }
    return results


# 2. unit_mixup -------------------------------------------------------------
def check_unit_mixup(train, group_col, exclude=()):
    """For each numeric column, compare each group's median with the typical
    group (the median of all the group medians). A group that is about
    10/12/60/100/1000 times off is a unit mix-up. Medians ignore odd rows.
    """
    findings = []
    for col in numeric_columns(train, exclude):
        medians = train.groupby(group_col)[col].median()
        typical = medians.median()
        if typical == 0:
            continue
        for group, value in medians.items():
            if value == 0:
                continue
            ratio = value / typical
            big = max(ratio, 1 / ratio)
            if big < 5:
                continue
            factor = min(UNIT_FACTORS, key=lambda f: abs(f - big))
            findings.append({"column": col, "group_value": group,
                             "ratio": round(float(ratio), 2), "factor": factor})
    return findings


# 3. sentinel_value ---------------------------------------------------------
def check_sentinel_value(train, exclude=(), min_count=5):
    """Look for a special number repeated many times. A real sentinel is
    impossible for the column (e.g. -1 minutes) AND repeated. A value that
    is normal for the column (a fare of 99 rupees) is not a sentinel.
    """
    hits = []
    for col in numeric_columns(train, exclude):
        for value in SENTINEL_CANDIDATES:
            n = int((train[col] == value).sum())
            if n == 0:
                continue
            others = train.loc[train[col] != value, col]
            inside_normal_range = others.min() <= value <= others.max()
            hits.append({"column": col, "value": value, "count": n,
                         "looks_like_sentinel": n >= min_count
                         and not inside_normal_range})
    return hits


# 4. duplicate_rows ---------------------------------------------------------
def check_duplicate_rows(train, id_col="record_id"):
    """Extra copies: exact copies, and copies that only differ in the id."""
    return {
        "exact": int(train.duplicated().sum()),
        "ignoring_id": int(train.drop(columns=id_col).duplicated().sum()),
    }


# 5. stuck_sensor -----------------------------------------------------------
def check_stuck_sensor(train, date_col="date", min_run=20, exclude=()):
    """Sort by date, then find the longest run of identical consecutive values
    in each numeric column. A long run (min_run rows or more) is suspicious.
    """
    df = train.sort_values([date_col, "record_id"]).reset_index(drop=True)
    findings = []
    for col in numeric_columns(df, exclude):
        # A new run starts every time the value changes.
        run_id = (df[col] != df[col].shift()).cumsum()
        lengths = run_id.value_counts()
        longest = lengths.idxmax()
        print(f"   longest run in {col}: {lengths[longest]} rows")
        if lengths[longest] < min_run:
            continue
        rows = df[run_id == longest]
        findings.append({
            "column": col,
            "value": rows[col].iloc[0],
            "rows": int(lengths[longest]),
            "start_date": rows[date_col].min(),
            "end_date": rows[date_col].max(),
        })
    return findings


# 6. target_typos -----------------------------------------------------------
def check_target_typos(train, target):
    """Sort the target and find the biggest jump between neighbours.
    Everything above that jump is a typo (typed 10 times too big).
    """
    s = train[target].sort_values().reset_index(drop=True)
    jump = s.shift(-1) / s          # how much bigger the next value is
    i = int(jump.idxmax())
    threshold = s[i + 1]
    return {
        "last_normal_value": int(s[i]),
        "first_typo_value": int(threshold),
        "jump_ratio": round(float(jump[i]), 2),
        "count": int((train[target] >= threshold).sum()),
    }


# 7. spelling_variants ------------------------------------------------------
def clean_text(series):
    """Normalise spelling: trim spaces, lower case, drop a trailing dot,
    and expand abbreviations."""
    s = series.str.strip().str.lower().str.rstrip(".")
    return s.replace({"str": "street", "sta": "stand"})


def check_spelling_variants(train, later):
    """Compare raw spellings in train with the clean spellings in later.csv.
    A column has spelling variants if train has more levels than later.
    """
    results = {}
    for col in train.select_dtypes(include=["object", "string"]).columns:
        if col in ("record_id", "date") or col not in later.columns:
            continue
        raw = int(train[col].nunique())
        clean_in_later = sorted(later[col].unique())
        results[col] = {
            "raw_levels": raw,
            "later_levels": clean_in_later,
            "has_variants": raw > len(clean_in_later),
            "train_after_cleaning": sorted(clean_text(train[col]).unique()),
            "true_levels": len(clean_in_later),
        }
    return results


# run everything ------------------------------------------------------------
def main():
    p = load_params()["data"]
    target = p["target"]
    train = pd.read_csv(p["train_path"])
    later = pd.read_csv(p["later_path"])
    holdout = pd.read_csv(p["holdout_path"])
    skip = ("record_id",)

    print("1. leaking_column:", check_leaking_column(train, holdout, target))
    print("2. unit_mixup:", check_unit_mixup(train, "zone", exclude=skip))
    print("3. sentinel_value:", check_sentinel_value(train, exclude=skip))
    print("4. duplicate_rows:", check_duplicate_rows(train))
    print("5. stuck_sensor:", check_stuck_sensor(train, exclude=skip))
    print("6. target_typos:", check_target_typos(train, target))
    print("7. spelling_variants:", check_spelling_variants(train, later))


if __name__ == "__main__":
    main()