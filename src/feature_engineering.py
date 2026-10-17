"""Stateless feature engineering.

Every transform here is a *deterministic function of a single row* (no statistics are
learned from data), so applying it before the train/test split cannot leak information.
It is nevertheless wrapped inside the sklearn Pipeline so that training, cross-validation
and prediction all run exactly the same code path.
"""
import numpy as np
import pandas as pd

from src.config import ADDON_SERVICES, CATEGORICAL_FEATURES, NUMERIC_FEATURES

TENURE_BINS = [-0.1, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-12m", "13-24m", "25-48m", "49m+"]

ENGINEERED_NUMERIC = ["num_addon_services", "is_autopay"]
ENGINEERED_CATEGORICAL = ["tenure_group"]

FEATURE_SETS = {
    "base": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
    "engineered": {
        "numeric": NUMERIC_FEATURES + ENGINEERED_NUMERIC,
        "categorical": CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL,
    },
}


def get_feature_columns(feature_set: str = "base"):
    """Return (numeric_columns, categorical_columns) for the requested feature set."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set '{feature_set}'. Choose from {list(FEATURE_SETS)}.")
    fs = FEATURE_SETS[feature_set]
    return list(fs["numeric"]), list(fs["categorical"])


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add three hypothesis-driven features (returns a copy).

    num_addon_services : count of add-on services ("Yes" only) - captures how embedded a
                         customer is in the product ecosystem.
    is_autopay         : 1 if the payment method is automatic - a proxy for friction at
                         billing time.
    tenure_group       : coarse tenure buckets - lets a *linear* model represent the
                         non-linear tenure/churn relationship (trees can learn this alone).
    """
    out = df.copy()
    out["num_addon_services"] = (out[ADDON_SERVICES] == "Yes").sum(axis=1).astype(float)
    out["is_autopay"] = (
        out["PaymentMethod"].astype("object").str.contains("automatic", case=False, na=False).astype(float)
    )
    out["tenure_group"] = pd.cut(out["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS).astype("object")
    return out
