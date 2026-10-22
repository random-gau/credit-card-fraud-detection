"""Step 6: the ONE run on the test set.

Everything is frozen before this script: both models (fit on train), their
thresholds (tuned on validation, from reports/tables/05_thresholds.json).
Nothing here is fitted or tuned on test.

Reports, for logistic regression (baseline) and XGBoost (best):
  - PR-AUC, ROC-AUC, precision, recall, F1 at the frozen threshold, confusion matrix
  - 95% bootstrap CIs (2,000 resamples, stratified by class so each has 95 frauds)
  - paired bootstrap of the XGBoost - baseline difference (same resample for both)

Guard: refuses to run twice unless --rerun is passed, to keep the test set honest.

Run: python -m src.final_test
"""
import json
import sys
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from src.config import FIGURES, ROOT, SEED, TABLES
from src.evaluate import evaluate, load_split

MODELS = {"logreg_balanced": "baseline_logreg.joblib", "xgb_best": "xgb_best.joblib"}
N_BOOT = 2000
MARKER = TABLES / "06_TEST_RUN_DONE.txt"


def boot_metrics(y, p, t, idx):
    yb, pb = y[idx], p[idx]
    yhat = (pb >= t).astype(int)
    return (
        average_precision_score(yb, pb),
        precision_score(yb, yhat, zero_division=0),
        recall_score(yb, yhat, zero_division=0),
        f1_score(yb, yhat, zero_division=0),
    )


def main() -> None:
    if MARKER.exists() and "--rerun" not in sys.argv:
        print(f"Test set already used ({MARKER.read_text().strip()}). Pass --rerun to override.")
        return

    thresholds = json.loads((TABLES / "05_thresholds.json").read_text())["thresholds"]
    X_te, y_te = load_split("test")
    y = y_te.to_numpy()
    probas = {n: joblib.load(ROOT / "models" / f).predict_proba(X_te)[:, 1] for n, f in MODELS.items()}

    # Point estimates
    rows = []
    for n, p in probas.items():
        rows.append({"model": n, **evaluate(y, p, thresholds[n])})
        rows.append({"model": n + " (t=0.5, reference)", **evaluate(y, p, 0.5)})
    point = pd.DataFrame(rows)

    # Stratified bootstrap: resample frauds and non-frauds separately
    rng = np.random.default_rng(SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    names = ["pr_auc", "precision", "recall", "f1"]
    boots = {n: [] for n in probas}
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for n, p in probas.items():
            boots[n].append(boot_metrics(y, p, thresholds[n], idx))
    boots = {n: np.array(v) for n, v in boots.items()}

    ci_rows = []
    for n, b in boots.items():
        for j, m in enumerate(names):
            lo, hi = np.percentile(b[:, j], [2.5, 97.5])
            ci_rows.append({"model": n, "metric": m, "ci_low": lo, "ci_high": hi})
    ci = pd.DataFrame(ci_rows)

    diff = boots["xgb_best"] - boots["logreg_balanced"]  # paired: same resample
    diff_rows = []
    for j, m in enumerate(names):
        d = diff[:, j]
        lo, hi = np.percentile(d, [2.5, 97.5])
        diff_rows.append({"metric": m, "mean_diff": d.mean(), "ci_low": lo, "ci_high": hi,
                          "share_xgb_better": (d > 0).mean()})
    diffs = pd.DataFrame(diff_rows)

    cols = ["model", "threshold", "pr_auc", "roc_auc", "precision", "recall", "f1", "tp", "fp", "fn", "tn"]
    print(f"TEST SET: {len(y)} transactions, {int(y.sum())} frauds\n")
    print(point[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n95% bootstrap CIs (frozen thresholds):\n", ci.round(4).to_string(index=False))
    print("\nPaired difference, XGBoost - baseline:\n", diffs.round(4).to_string(index=False))

    # Confusion matrices at frozen thresholds
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), dpi=150)
    for ax, n in zip(axes, MODELS):
        r = point[point.model == n].iloc[0]
        cm = np.array([[r.tn, r.fp], [r.fn, r.tp]])
        ax.imshow(np.log1p(cm), cmap="Blues", vmin=0)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{int(v):,}", ha="center", va="center", fontsize=11,
                    color="white" if np.log1p(v) > np.log1p(cm.max()) * 0.6 else "#0b0b0b")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["genuine", "fraud"],
               yticklabels=["genuine", "fraud"], xlabel="Predicted", ylabel="Actual",
               title=f"{n}\nthreshold {r.threshold:.3g}")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Test set confusion matrices (colour = log count)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "06_confusion_matrices_test.png")

    TABLES.mkdir(parents=True, exist_ok=True)
    point[cols].to_csv(TABLES / "06_test_results.csv", index=False)
    ci.to_csv(TABLES / "06_test_bootstrap_ci.csv", index=False)
    diffs.to_csv(TABLES / "06_test_paired_diff.csv", index=False)
    MARKER.write_text(f"test run at {datetime.now().isoformat(timespec='seconds')}\n")
    print("\nSaved reports/tables/06_*.csv and reports/figures/06_confusion_matrices_test.png")


if __name__ == "__main__":
    main()
