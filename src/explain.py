"""Step 7: explain the XGBoost model with SHAP.

SHAP values come from XGBoost's own TreeSHAP (pred_contribs=True): exact for
tree ensembles, and immune to shap/xgboost version mismatches. The shap library
is used only for plotting. Values are in log-odds units: positive pushes towards
fraud. We check additivity (base + sum of contributions = model margin).

Explanations use the test set. Nothing is fitted or tuned here, so this does not
leak; the test numbers from step 6 stay final.

Features are the model's inputs after preprocessing (scale=False), so 'Amount'
is log1p(Amount), shown as log_amount.

Run: python -m src.explain
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from src.config import FIGURES, ROOT, SEED, TABLES
from src.evaluate import load_split


def save(path):
    plt.gcf().set_dpi(150)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close("all")


def main() -> None:
    model = joblib.load(ROOT / "models" / "xgb_best.joblib")
    t = json.loads((TABLES / "05_thresholds.json").read_text())["thresholds"]["xgb_best"]
    pre, clf = model.named_steps["pre"], model.named_steps["clf"]

    X_te, y_te = load_split("test")
    y = y_te.to_numpy()
    Xt = pre.transform(X_te)
    names = ["log_amount" if n == "Amount" else n for n in pre.get_feature_names_out()]

    booster = clf.get_booster()
    dm = xgb.DMatrix(Xt, feature_names=booster.feature_names)
    contrib = booster.predict(dm, pred_contribs=True)  # (n, features + 1); last column = base
    sv, base = contrib[:, :-1], contrib[:, -1]
    margin = booster.predict(dm, output_margin=True)
    print("additivity check (base + sum SHAP == margin):", np.allclose(sv.sum(1) + base, margin, atol=1e-3))

    proba = model.predict_proba(X_te)[:, 1]

    # Global importance: all test rows, and frauds only
    imp = pd.DataFrame({
        "feature": names,
        "mean_abs_shap_all": np.abs(sv).mean(0),
        "mean_abs_shap_frauds": np.abs(sv[y == 1]).mean(0),
        "mean_shap_frauds": sv[y == 1].mean(0),  # sign: direction on frauds
    }).sort_values("mean_abs_shap_all", ascending=False)
    print("\nTop 10 features by mean |SHAP| (log-odds), test set:\n",
          imp.head(10).round(3).to_string(index=False))

    expl = shap.Explanation(values=sv, base_values=base, data=Xt, feature_names=names)

    # Beeswarm on all frauds + 5,000 random genuine (plotting all 56k is slow)
    rng = np.random.default_rng(SEED)
    rows = np.concatenate([np.where(y == 1)[0], rng.choice(np.where(y == 0)[0], 5000, replace=False)])
    FIGURES.mkdir(parents=True, exist_ok=True)
    shap.plots.beeswarm(expl[rows], max_display=12, show=False)
    plt.title("SHAP values: 95 test frauds + 5,000 genuine", fontsize=10)
    save(FIGURES / "07_shap_beeswarm.png")

    shap.plots.bar(expl, max_display=12, show=False)
    plt.title("Mean |SHAP| over the test set", fontsize=10)
    save(FIGURES / "07_shap_bar.png")

    # Two example predictions: the most confident catch, and the most confidently missed fraud
    fraud = np.where(y == 1)[0]
    tp_i = fraud[np.argmax(proba[fraud])]
    fn_i = fraud[np.argmin(proba[fraud])]
    for label, i in [("caught", tp_i), ("missed", fn_i)]:
        flagged = "flagged" if proba[i] >= t else "NOT flagged"
        print(f"\n{label} fraud (test row {i}): p={proba[i]:.4g} -> {flagged} at t={t:.3g}")
        top = pd.Series(sv[i], index=names).sort_values(key=np.abs, ascending=False).head(5)
        print("  top contributions (log-odds):", top.round(2).to_dict())
        shap.plots.waterfall(expl[i], max_display=10, show=False)
        plt.title(f"{label.capitalize()} fraud: p = {proba[i]:.3g} (threshold {t:.3g})", fontsize=10)
        save(FIGURES / f"07_shap_waterfall_{label}.png")

    missed = fraud[proba[fraud] < t]
    print(f"\nMissed frauds: {len(missed)}; their median predicted p = {np.median(proba[missed]):.4g}")

    TABLES.mkdir(parents=True, exist_ok=True)
    imp.to_csv(TABLES / "07_shap_importance.csv", index=False)
    print("\nSaved reports/tables/07_shap_importance.csv and reports/figures/07_*.png")


if __name__ == "__main__":
    main()
