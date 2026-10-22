"""Step 5: choose decision thresholds on VALIDATION, for both models.

Rules reported (no cost model exists, so the precision target is a stated
assumption, not a business fact):
  primary   highest recall with precision >= TARGET_PRECISION (0.80, i.e. at
            most 1 in 5 alerts is a false alarm); ties -> highest threshold
  others    max F1, and recall at precision >= 0.5 / 0.7 / 0.9 (trade-off table)

Both models get their own tuned threshold, so step 6 compares like with like.
Thresholds are saved to reports/tables/05_thresholds.json for step 6.
Test is not loaded.

Run: python -m src.threshold
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.config import FIGURES, ROOT, TABLES
from src.evaluate import evaluate, load_split

TARGET_PRECISION = 0.80
OTHER_TARGETS = [0.5, 0.7, 0.9]
MODELS = {"logreg_balanced": "baseline_logreg.joblib", "xgb_best": "xgb_best.joblib"}
COLORS = {"logreg_balanced": "#2a78d6", "xgb_best": "#eb6834"}  # fixed categorical order


def curve(y, proba) -> pd.DataFrame:
    prec, rec, thr = precision_recall_curve(y, proba)
    # last (precision=1, recall=0) point has no threshold
    return pd.DataFrame({"threshold": thr, "precision": prec[:-1], "recall": rec[:-1]})


def best_recall_at_precision(c: pd.DataFrame, p_min: float):
    ok = c[c.precision >= p_min]
    if ok.empty:
        return None
    return float(ok[ok.recall == ok.recall.max()].threshold.max())


def max_f1_threshold(c: pd.DataFrame) -> float:
    f1 = 2 * c.precision * c.recall / (c.precision + c.recall).replace(0, np.nan)
    return float(c.loc[f1.idxmax(), "threshold"])


def main() -> None:
    X_va, y_va = load_split("val")
    rows, chosen, curves = [], {}, {}

    for name, fname in MODELS.items():
        proba = joblib.load(ROOT / "models" / fname).predict_proba(X_va)[:, 1]
        c = curve(y_va, proba)
        curves[name] = c

        rules = {"default_0.5": 0.5, "max_f1": max_f1_threshold(c)}
        for p in sorted(OTHER_TARGETS + [TARGET_PRECISION]):
            rules[f"recall@P>={p}"] = best_recall_at_precision(c, p)

        for rule, t in rules.items():
            if t is None:
                rows.append({"model": name, "rule": rule, "threshold": np.nan})
                continue
            m = evaluate(y_va, proba, t)
            m["alerts"] = m["tp"] + m["fp"]
            rows.append({"model": name, "rule": rule, **m})
        chosen[name] = rules[f"recall@P>={TARGET_PRECISION}"]

    res = pd.DataFrame(rows)
    cols = ["model", "rule", "threshold", "precision", "recall", "f1", "tp", "fp", "fn", "alerts"]
    print("Operating points on VALIDATION (94 frauds):\n")
    print(res[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nChosen (recall @ precision >= {TARGET_PRECISION}):", chosen)

    # How fragile is the chosen XGBoost threshold? Neighbouring thresholds:
    proba_x = joblib.load(ROOT / "models" / MODELS["xgb_best"]).predict_proba(X_va)[:, 1]
    t0 = chosen["xgb_best"]
    if t0 is not None:
        print("\nXGBoost sensitivity around chosen threshold:")
        for t in sorted({t0 / 4, t0 / 2, t0, min(t0 * 2, 0.999), min(t0 * 4, 0.999)}):
            m = evaluate(y_va, proba_x, t)
            print(f"  t={t:.4g}  precision={m['precision']:.3f}  recall={m['recall']:.3f}  tp={m['tp']} fp={m['fp']}")

    # Figure 1: PR curves with chosen points
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=150)
    for name, c in curves.items():
        ax.plot(c.recall, c.precision, lw=2, color=COLORS[name], label=name)
        t = chosen[name]
        if t is not None:
            r = res[(res.model == name) & (res.rule == f"recall@P>={TARGET_PRECISION}")].iloc[0]
            ax.plot(r.recall, r.precision, "o", ms=8, color=COLORS[name],
                    markeredgecolor="white", markeredgewidth=2, zorder=5)
            ax.annotate(f"t={t:.3g}\nP={r.precision:.2f}, R={r.recall:.2f}",
                        (r.recall, r.precision), textcoords="offset points",
                        xytext=(-10, -32), fontsize=8, color="#3d3d3a", ha="right")
    ax.axhline(TARGET_PRECISION, color="#b4b2a9", lw=1, ls="--")
    ax.text(0.01, TARGET_PRECISION + 0.015, f"precision target {TARGET_PRECISION}", fontsize=8, color="#73726c")
    ax.axhline(y_va.mean(), color="#b4b2a9", lw=1, ls=":")
    ax.text(0.01, y_va.mean() + 0.015, "no-skill", fontsize=8, color="#73726c")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1.02), ylim=(0, 1.05),
           title="Precision–recall on validation (94 frauds)")
    ax.grid(color="#e8e6dc", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "05_pr_curve_val.png")

    # Figure 2: XGBoost precision and recall vs threshold (same 0-1 scale, one axis)
    cx = curves["xgb_best"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    ax.plot(cx.threshold, cx.precision, lw=2, color="#2a78d6", label="precision")
    ax.plot(cx.threshold, cx.recall, lw=2, color="#eb6834", label="recall")
    if t0 is not None:
        ax.axvline(t0, color="#b4b2a9", lw=1, ls="--")
        ax.text(t0, 0.03, f" chosen t={t0:.3g}", fontsize=8, color="#73726c")
    ax.set(xscale="log", xlabel="Threshold (log scale)", ylabel="Value", ylim=(0, 1.05),
           title="XGBoost: precision and recall vs threshold (validation)")
    ax.grid(color="#e8e6dc", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "05_xgb_threshold_tradeoff.png")

    TABLES.mkdir(parents=True, exist_ok=True)
    res[cols].to_csv(TABLES / "05_operating_points_val.csv", index=False)
    (TABLES / "05_thresholds.json").write_text(json.dumps(
        {"rule": f"max recall with precision >= {TARGET_PRECISION} on validation",
         "thresholds": chosen}, indent=2))
    print("\nSaved reports/tables/05_*.csv/json and reports/figures/05_*.png")


if __name__ == "__main__":
    main()
