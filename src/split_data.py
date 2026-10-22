"""Step 1: check the raw data, then make a stratified 60/20/20 train/validation/test split.

Run from the repo root:
    python -m src.split_data

Outputs
    data/processed/{train,val,test}.parquet   (gitignored)
    reports/tables/01_data_checks.json
    reports/tables/01_split_summary.csv
"""
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROCESSED, RAW, SEED, TABLES, TARGET


def data_checks(df: pd.DataFrame) -> dict:
    """Facts about the raw file, computed (never assumed)."""
    features = df.columns.drop(TARGET).tolist()
    dup_mask = df.duplicated()  # exact copy of an earlier row, label included
    # Same features but different labels: these can't be fixed by dropping duplicates
    n_conflicting = int(df.groupby(features)[TARGET].nunique().gt(1).sum())
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "frauds": int(df[TARGET].sum()),
        "fraud_rate_pct": round(100 * df[TARGET].mean(), 4),
        "missing_values": int(df.isna().sum().sum()),
        "exact_duplicate_rows": int(dup_mask.sum()),
        "exact_duplicate_frauds": int(df.loc[dup_mask, TARGET].sum()),
        "feature_groups_with_conflicting_labels": n_conflicting,
        "time_min_s": float(df["Time"].min()),
        "time_max_s": float(df["Time"].max()),
        "amount_min": float(df["Amount"].min()),
        "amount_median": float(df["Amount"].median()),
        "amount_max": float(df["Amount"].max()),
    }


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW)
    assert df[TARGET].isin([0, 1]).all(), "Class must be 0/1"

    checks = data_checks(df)
    for k, v in checks.items():
        print(f"{k:42s} {v}")

    # Decision: drop exact duplicates BEFORE splitting, so the same transaction
    # can't sit in both train and test (that would inflate test scores).
    df = df.drop_duplicates().reset_index(drop=True)
    checks["rows_after_dedup"] = len(df)
    checks["frauds_after_dedup"] = int(df[TARGET].sum())

    # 80/20 -> (train+val)/test, then 75/25 of the 80 -> 60/20/20 overall.
    # stratify keeps the fraud rate the same in every split.
    trval, test = train_test_split(
        df, test_size=0.20, stratify=df[TARGET], random_state=SEED
    )
    train, val = train_test_split(
        trval, test_size=0.25, stratify=trval[TARGET], random_state=SEED
    )

    # Leakage guards: no shared rows by index, none by content.
    assert not set(train.index) & set(val.index)
    assert not set(train.index) & set(test.index)
    assert not set(val.index) & set(test.index)
    feats = df.columns.drop(TARGET).tolist()
    overlap = pd.merge(train[feats], test[feats], how="inner").shape[0]
    assert overlap == 0, f"{overlap} identical feature rows in train and test"

    summary = pd.DataFrame(
        [
            {
                "split": name,
                "rows": len(part),
                "frauds": int(part[TARGET].sum()),
                "fraud_rate_pct": round(100 * part[TARGET].mean(), 4),
            }
            for name, part in [("train", train), ("val", val), ("test", test)]
        ]
    )
    print("\n", summary.to_string(index=False))

    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_parquet(PROCESSED / f"{name}.parquet", index=False)
    summary.to_csv(TABLES / "01_split_summary.csv", index=False)
    (TABLES / "01_data_checks.json").write_text(json.dumps(checks, indent=2))
    print("\nSaved splits to", PROCESSED)


if __name__ == "__main__":
    main()
