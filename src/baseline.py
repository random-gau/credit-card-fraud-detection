"""Step 2: baseline logistic regression with class_weight='balanced'.

Trained on train, scored on validation. The test set is NOT loaded here.

Run: python -m src.baseline
Outputs: reports/tables/02_baseline.csv, models/baseline_logreg.joblib
"""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import ROOT, SEED, TABLES
from src.evaluate import evaluate, load_split
from src.features import make_preprocessor


def make_baseline() -> Pipeline:
    # Scaling lives inside the pipeline, so it is fitted on train only.
    return Pipeline(
        [
            ("pre", make_preprocessor(scale=True)),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",  # weight each class by 1 / its frequency
                    max_iter=2000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def main() -> None:
    X_tr, y_tr = load_split("train")
    X_va, y_va = load_split("val")

    model = make_baseline().fit(X_tr, y_tr)

    rows = []
    for name, X, y in [("train", X_tr, y_tr), ("val", X_va, y_va)]:
        proba = model.predict_proba(X)[:, 1]
        rows.append({"model": "logreg_balanced", "split": name, **evaluate(y, proba)})
    res = pd.DataFrame(rows)
    print(res.round(4).to_string(index=False))
    print(f"\nNo-skill PR-AUC on val (= fraud rate): {y_va.mean():.4f}")

    # Descriptive only: which scaled features push the score up or down.
    coefs = pd.Series(
        model.named_steps["clf"].coef_[0],
        index=model.named_steps["pre"].get_feature_names_out(),
    ).sort_values()
    print("\nMost negative coefficients:\n", coefs.head(5).round(3).to_string())
    print("\nMost positive coefficients:\n", coefs.tail(5).round(3).to_string())

    TABLES.mkdir(parents=True, exist_ok=True)
    res.to_csv(TABLES / "02_baseline.csv", index=False)
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models" / "baseline_logreg.joblib")
    print("\nSaved reports/tables/02_baseline.csv and models/baseline_logreg.joblib")


if __name__ == "__main__":
    main()
