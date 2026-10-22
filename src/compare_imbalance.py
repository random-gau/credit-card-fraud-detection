"""Step 3: compare imbalance strategies with 5-fold stratified CV on TRAIN only.

Strategies (all logistic regression, same preprocessing):
  none            no reweighting (reference point)
  class_weight    class_weight='balanced'
  undersample     RandomUnderSampler to 1:1 (what the old script did)
  smote           SMOTE to 1:1

Why imblearn's Pipeline: a sampler inside it runs only during fit(). When
cross_validate scores a validation fold, the fold passes through preprocessing
and the model but is NEVER resampled. Resampling before the split would put
synthetic copies of validation frauds into training (leakage).

Every strategy uses the SAME 5 folds, so the comparison is paired.
Validation and test sets are not loaded.

Run: python -m src.compare_imbalance
Output: reports/tables/03_imbalance_cv.csv
"""
import time

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import SEED, TABLES
from src.evaluate import load_split
from src.features import make_preprocessor

SCORING = {
    "pr_auc": "average_precision",  # threshold-free: the metric we choose on
    "roc_auc": "roc_auc",
    "precision": "precision",  # these three use the 0.5 threshold
    "recall": "recall",
    "f1": "f1",
}


def logreg(class_weight=None):
    return LogisticRegression(class_weight=class_weight, max_iter=2000, random_state=SEED)


def strategies() -> dict:
    # Sampler goes AFTER preprocessing: SMOTE interpolates between neighbours,
    # so it needs scaled features to measure distance sensibly.
    return {
        "none": Pipeline([("pre", make_preprocessor()), ("clf", logreg())]),
        "class_weight": Pipeline([("pre", make_preprocessor()), ("clf", logreg("balanced"))]),
        "undersample": Pipeline(
            [
                ("pre", make_preprocessor()),
                ("sample", RandomUnderSampler(random_state=SEED)),
                ("clf", logreg()),
            ]
        ),
        "smote": Pipeline(
            [
                ("pre", make_preprocessor()),
                ("sample", SMOTE(random_state=SEED)),
                ("clf", logreg()),
            ]
        ),
    }


def main() -> None:
    X, y = load_split("train")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rows, per_fold = [], []
    for name, pipe in strategies().items():
        t0 = time.time()
        scores = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
        secs = time.time() - t0
        row = {"strategy": name, "seconds": round(secs, 1)}
        for m in SCORING:
            vals = scores[f"test_{m}"]
            row[f"{m}_mean"] = vals.mean()
            row[f"{m}_std"] = vals.std()
            per_fold += [{"strategy": name, "fold": i, "metric": m, "value": v} for i, v in enumerate(vals)]
        rows.append(row)
        print(f"done: {name:13s} PR-AUC {row['pr_auc_mean']:.4f} ± {row['pr_auc_std']:.4f}  ({secs:.0f}s)")

    res = pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False)
    cols = ["strategy"] + [f"{m}_{s}" for m in SCORING for s in ("mean", "std")] + ["seconds"]
    print("\n", res[cols].round(4).to_string(index=False))

    # Paired view: PR-AUC per fold, so you can see if one strategy wins on EVERY fold
    pf = pd.DataFrame(per_fold)
    print("\nPR-AUC per fold:\n", pf[pf.metric == "pr_auc"].pivot(index="fold", columns="strategy", values="value").round(4).to_string())

    TABLES.mkdir(parents=True, exist_ok=True)
    res[cols].to_csv(TABLES / "03_imbalance_cv.csv", index=False)
    pf.to_csv(TABLES / "03_imbalance_cv_folds.csv", index=False)
    print("\nSaved reports/tables/03_imbalance_cv.csv and 03_imbalance_cv_folds.csv")


if __name__ == "__main__":
    main()
