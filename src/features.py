"""Preprocessing used by every model.

Nothing here is fitted at import time. make_preprocessor() returns an UNFITTED
ColumnTransformer that goes inside each model's Pipeline, so scaling statistics
are learned from the training data (or training fold) only.

Self-check: python -m src.features
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

V_COLS = [f"V{i}" for i in range(1, 29)]
SECONDS_PER_DAY = 86_400


def _time_to_cycle(X):
    """Time (seconds since the first transaction) -> position in a 24 h cycle.

    Row-wise, so no leakage. Raw Time just counts up over the 2 days of data;
    sin/cos keeps 23:59 next to 00:00. Caveat: we don't know what clock time
    Time = 0 corresponds to, so this is 'time since start', not true hour of day.
    """
    t = np.asarray(X, dtype=float).reshape(-1, 1) % SECONDS_PER_DAY
    angle = 2 * np.pi * t / SECONDS_PER_DAY
    return np.hstack([np.sin(angle), np.cos(angle)])


def _cycle_names(transformer, input_features):
    return np.array(["time_sin", "time_cos"])


def make_preprocessor(scale: bool = True) -> ColumnTransformer:
    """scale=True for linear models; scale=False for tree models (they don't need it)."""

    def scaler():
        return StandardScaler() if scale else "passthrough"

    amount = Pipeline(
        [
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", scaler()),
        ]
    )
    time_cycle = FunctionTransformer(_time_to_cycle, feature_names_out=_cycle_names)

    return ColumnTransformer(
        [
            ("v", scaler(), V_COLS),
            ("amount", amount, ["Amount"]),
            ("time", time_cycle, ["Time"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


if __name__ == "__main__":
    import pandas as pd

    from src.config import PROCESSED, TARGET

    train = pd.read_parquet(PROCESSED / "train.parquet")
    X = train.drop(columns=TARGET)
    pre = make_preprocessor(scale=True).fit(X)  # fitted on TRAIN only
    Xt = pd.DataFrame(pre.transform(X), columns=pre.get_feature_names_out())
    print("features:", Xt.shape[1], list(Xt.columns))
    print(Xt[["V1", "V14", "Amount", "time_sin"]].describe().round(3))
