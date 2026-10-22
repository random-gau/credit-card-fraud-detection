"""Step 4: XGBoost with a small randomized hyperparameter search.

- Search runs on TRAIN only, with the SAME 5 stratified folds as step 3,
  scored on PR-AUC, so its CV score is directly (pairwise) comparable with
  the logistic-regression strategies in reports/tables/03_imbalance_cv_folds.csv.
- scale_pos_weight (XGBoost's class weight) is searched, not fixed, because
  step 3 showed weighting doesn't improve ranking for the linear model.
- Validation is scored once at the end for both models, as a sanity check.
  Selection is by CV; validation is reserved for threshold tuning (step 5).
- Test is not loaded.

Run: python -m src.tree_model
"""
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import ROOT, SEED, TABLES
from src.evaluate import evaluate, load_split
from src.features import make_preprocessor


def main() -> None:
    X, y = load_split("train")
    X_va, y_va = load_split("val")
    ratio = (y == 0).sum() / (y == 1).sum()  # negatives per fraud, from TRAIN only
    print(f"neg/pos ratio in train: {ratio:.1f}")

    pipe = Pipeline(
        [
            ("pre", make_preprocessor(scale=False)),  # trees don't need scaling
            (
                "clf",
                XGBClassifier(
                    tree_method="hist",
                    eval_metric="aucpr",
                    n_jobs=1,  # parallelism comes from the search instead
                    random_state=SEED,
                ),
            ),
        ]
    )
    grid = {
        "clf__n_estimators": [200, 400, 600],
        "clf__learning_rate": [0.03, 0.05, 0.1],
        "clf__max_depth": [3, 4, 5, 6],
        "clf__min_child_weight": [1, 3, 5],
        "clf__subsample": [0.7, 0.85, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__scale_pos_weight": [1.0, float(np.sqrt(ratio)), float(ratio)],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)  # same folds as step 3
    search = RandomizedSearchCV(
        pipe,
        grid,
        n_iter=25,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        random_state=SEED,
        return_train_score=True,
        verbose=1,
    )
    t0 = time.time()
    search.fit(X, y)
    print(f"search took {time.time() - t0:.0f}s")

    # Top configurations
    cvr = pd.DataFrame(search.cv_results_)
    keep = ["rank_test_score", "mean_test_score", "std_test_score", "mean_train_score"] + [
        c for c in cvr.columns if c.startswith("param_")
    ]
    top = cvr[keep].sort_values("rank_test_score").head(10)
    top.columns = [c.replace("param_clf__", "") for c in top.columns]
    print("\nTop 10 configs (CV PR-AUC on train):\n", top.round(4).to_string(index=False))

    best_i = search.best_index_
    print("\nBest params:", {k.replace("clf__", ""): v for k, v in search.best_params_.items()})
    print(
        f"Best CV PR-AUC: {cvr.loc[best_i, 'mean_test_score']:.4f} "
        f"± {cvr.loc[best_i, 'std_test_score']:.4f} "
        f"(train-fold PR-AUC {cvr.loc[best_i, 'mean_train_score']:.4f})"
    )

    # Paired comparison against step 3, fold by fold
    folds = pd.read_csv(TABLES / "03_imbalance_cv_folds.csv")
    lr = folds[(folds.metric == "pr_auc") & (folds.strategy == "class_weight")].set_index("fold")["value"]
    xgb_folds = pd.Series([cvr.loc[best_i, f"split{i}_test_score"] for i in range(5)])
    paired = pd.DataFrame({"logreg_balanced": lr.values, "xgb_best": xgb_folds.values})
    paired["diff"] = paired.xgb_best - paired.logreg_balanced
    print("\nPR-AUC per fold (same folds):\n", paired.round(4).to_string())
    print(f"XGBoost wins {int((paired['diff'] > 0).sum())}/5 folds, mean diff {paired['diff'].mean():+.4f}")

    # Sanity check on validation (selection already done by CV)
    best = search.best_estimator_  # refit on all of train
    base = joblib.load(ROOT / "models" / "baseline_logreg.joblib")
    rows = []
    for name, model in [("logreg_balanced", base), ("xgb_best", best)]:
        proba = model.predict_proba(X_va)[:, 1]
        rows.append({"model": name, "split": "val", **evaluate(y_va, proba)})
    rows.append(
        {"model": "xgb_best", "split": "train",
         **evaluate(y, best.predict_proba(X)[:, 1])}
    )
    res = pd.DataFrame(rows)
    print("\nValidation (threshold 0.5, tuned in step 5):\n", res.round(4).to_string(index=False))

    TABLES.mkdir(parents=True, exist_ok=True)
    top.to_csv(TABLES / "04_xgb_search_top10.csv", index=False)
    paired.to_csv(TABLES / "04_paired_folds.csv", index_label="fold")
    res.to_csv(TABLES / "04_model_comparison_val.csv", index=False)
    joblib.dump(best, ROOT / "models" / "xgb_best.joblib")
    print("\nSaved reports/tables/04_*.csv and models/xgb_best.joblib")


if __name__ == "__main__":
    main()
