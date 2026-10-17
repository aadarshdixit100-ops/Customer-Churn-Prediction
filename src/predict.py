"""Reusable prediction API: validate -> (saved) pipeline -> probability -> threshold.

The saved pipeline already contains the *fitted* preprocessing and model, so nothing is
re-fitted here. A churn probability is a model ESTIMATE of risk under the training data's
patterns - it is not a guarantee that a customer will (or will not) leave.

Examples
    python -m src.predict --json '{"tenure": 3, "MonthlyCharges": 80, ...}'
    python -m src.predict --csv new_customers.csv --output scored.csv
"""
import argparse
import json
import logging
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)
OPTIONAL_FIELDS = {"TotalCharges"}  # can be approximated (with a warning) if omitted


@lru_cache(maxsize=2)
def load_artifacts(model_path: str = str(config.MODEL_PATH), metadata_path: str = str(config.METADATA_PATH)):
    """Load the fitted pipeline and its metadata (threshold, schema, ...)."""
    for p in (model_path, metadata_path):
        if not Path(p).exists():
            raise FileNotFoundError(f"{p} not found. Train the model first: `python -m src.train`.")
    return joblib.load(model_path), json.loads(Path(metadata_path).read_text())


def _to_frame(customers) -> pd.DataFrame:
    if isinstance(customers, pd.DataFrame):
        return customers.copy().reset_index(drop=True)
    if isinstance(customers, dict):
        return pd.DataFrame([customers])
    if isinstance(customers, (list, tuple)) and all(isinstance(c, dict) for c in customers) and customers:
        return pd.DataFrame(list(customers))
    raise TypeError("customers must be a dict, a list of dicts, or a pandas DataFrame.")


def validate_customers(customers, metadata: dict):
    """Check required fields, allowed categories and numeric ranges.

    Raises ValueError for unusable input; returns (clean_dataframe, list_of_warnings) otherwise.
    """
    df = _to_frame(customers)
    schema, warnings = metadata["schema"], []

    required = [c for c in config.RAW_FEATURES if c not in OPTIONAL_FIELDS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required field(s): {missing}")

    # SeniorCitizen may arrive as 0/1/bool
    df["SeniorCitizen"] = df["SeniorCitizen"].map(
        lambda v: {0: "No", 1: "Yes", False: "No", True: "Yes", "0": "No", "1": "Yes"}.get(v, v))

    for col, allowed in schema["categorical"].items():
        df[col] = df[col].astype("object").map(lambda v: v.strip() if isinstance(v, str) else v)
        bad = sorted({str(v) for v in df[col] if v not in allowed})
        if bad:
            raise ValueError(f"Invalid value(s) {bad} for '{col}'. Allowed: {allowed}")

    for col in ["tenure", "MonthlyCharges"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any() or (df[col] < 0).any():
            raise ValueError(f"'{col}' must be a non-negative number.")
    if "TotalCharges" not in df.columns or df["TotalCharges"].isna().any():
        df["TotalCharges"] = pd.to_numeric(df.get("TotalCharges"), errors="coerce")
        fill = df["tenure"] * df["MonthlyCharges"]
        df["TotalCharges"] = df["TotalCharges"].fillna(fill)
        warnings.append("TotalCharges not provided: approximated as tenure x MonthlyCharges.")
    else:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        if df["TotalCharges"].isna().any() or (df["TotalCharges"] < 0).any():
            raise ValueError("'TotalCharges' must be a non-negative number.")

    for col, rng in schema["numeric"].items():
        if ((df[col] < rng["min"]) | (df[col] > rng["max"])).any():
            warnings.append(f"'{col}' is outside the range seen in training "
                            f"[{rng['min']:.1f}, {rng['max']:.1f}]; the prediction is an extrapolation.")

    # Logical consistency (the training data follows these rules; breaking them = unusual profile)
    no_net = df["InternetService"] == "No"
    addon_cols = [c for c in config.ADDON_SERVICES]
    if (no_net & (df[addon_cols] != "No internet service").any(axis=1)).any():
        warnings.append("InternetService is 'No' but an add-on service is not 'No internet service' "
                        "- an unusual profile for this dataset.")
    if ((df["PhoneService"] == "No") & (df["MultipleLines"] != "No phone service")).any():
        warnings.append("PhoneService is 'No' but MultipleLines is not 'No phone service' - unusual profile.")
    return df, warnings


def predict_churn(customers, threshold: float | None = None) -> pd.DataFrame:
    """Score one or many customers.

    Returns a DataFrame with churn_probability, predicted_churn (probability >= threshold)
    and the threshold used; validation warnings are stored in `result.attrs['warnings']`.
    """
    model, metadata = load_artifacts()
    thr = metadata["threshold"] if threshold is None else float(threshold)
    if not 0.0 <= thr <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    df, warnings = validate_customers(customers, metadata)
    prob = model.predict_proba(df[config.RAW_FEATURES])[:, 1]  # saved pipeline: no re-fitting
    out = pd.DataFrame({"churn_probability": prob, "predicted_churn": prob >= thr, "threshold": thr})
    out.attrs["warnings"] = warnings
    return out


def predict_one(customer: dict, threshold: float | None = None) -> dict:
    res = predict_churn(customer, threshold)
    row = res.iloc[0]
    return {"churn_probability": float(row.churn_probability), "predicted_churn": bool(row.predicted_churn),
            "threshold": float(row.threshold), "warnings": res.attrs["warnings"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score customers with the saved churn model.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--json", help="one customer as a JSON string")
    src.add_argument("--csv", help="CSV file with one customer per row")
    parser.add_argument("--output", help="where to write scored CSV (batch mode)")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.json:
        print(json.dumps(predict_one(json.loads(args.json), args.threshold), indent=2))
    else:
        df = pd.read_csv(args.csv)
        res = predict_churn(df, args.threshold)
        scored = pd.concat([df.reset_index(drop=True), res], axis=1)
        for w in res.attrs["warnings"]:
            logger.warning("WARNING: %s", w)
        if args.output:
            scored.to_csv(args.output, index=False)
            logger.info("Wrote %d scored customers to %s", len(scored), args.output)
        else:
            print(scored[["churn_probability", "predicted_churn"]].head(20).to_string())


if __name__ == "__main__":
    main()
