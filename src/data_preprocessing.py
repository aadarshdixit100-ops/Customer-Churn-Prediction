"""Loading, validating, cleaning, splitting and the sklearn preprocessing pipeline.

Order of operations (important for avoiding leakage):
  1. load_raw_data -> validate_schema -> clean_data   (deterministic, row-wise fixes only)
  2. split_train_test                                   (stratified, seeded)
  3. build_preprocessor / make_pipeline                 (FITTED ONLY on training folds)

Run as a script to create the train/test CSVs:   python -m src.data_preprocessing
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config
from src.feature_engineering import add_engineered_features, get_feature_columns

logger = logging.getLogger(__name__)

REQUIRED_RAW_COLUMNS = [config.ID_COL, config.TARGET] + config.RAW_FEATURES


# ----------------------------------------------------------------------------- loading
def load_raw_data(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV, with a helpful error if it has not been downloaded yet."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}.\n"
            f"Download it manually from:\n  {config.DOWNLOAD_URL}\n"
            f"and save it as:\n  {config.RAW_DATA_PATH}\n"
            "or run:  python -m src.data_preprocessing --download"
        )
    return pd.read_csv(path)


def download_raw_data(url: str = config.DOWNLOAD_URL, path: Path = config.RAW_DATA_PATH) -> Path:
    """Optional convenience: fetch the public IBM CSV. Fails loudly if offline."""
    import urllib.request
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as exc:  # network errors vary widely; re-raise with guidance
        raise RuntimeError(
            f"Could not download the dataset ({exc}). Download it manually from {url} "
            f"and place it at {path}."
        ) from exc
    return path


def validate_schema(df: pd.DataFrame) -> None:
    """Fail early and clearly if expected columns are missing."""
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df[config.ID_COL].duplicated().any():
        raise ValueError("customerID is expected to be unique but contains duplicates.")


# ---------------------------------------------------------------------------- cleaning
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Correct types and apply *row-wise, rule-based* fixes (no statistics learned).

    * Strip whitespace in text columns.
    * TotalCharges arrives as text with blank strings -> numeric. In this dataset every
      blank belongs to a customer with tenure == 0 (nothing billed yet), so the
      domain-correct value is 0, not a mean/median imputed from other customers.
      Any *other* unparsable value stays NaN and is handled by the in-pipeline imputer.
    * SeniorCitizen (0/1) -> "No"/"Yes" so it is treated as a categorical feature.
    * Churn "Yes"/"No" -> 1/0.
    """
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_string_dtype(out[col]) or out[col].dtype == object:
            out[col] = out[col].astype("object").str.strip()

    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    new_customer = (out["tenure"] == 0) & out["TotalCharges"].isna()
    out.loc[new_customer, "TotalCharges"] = 0.0

    if out["SeniorCitizen"].dtype != object:
        out["SeniorCitizen"] = out["SeniorCitizen"].map({0: "No", 1: "Yes"})

    out[config.TARGET] = out[config.TARGET].map({"No": 0, "Yes": 1})
    if out[config.TARGET].isna().any():
        raise ValueError("Unexpected values in the Churn column (expected only 'Yes'/'No').")
    out[config.TARGET] = out[config.TARGET].astype(int)
    return out


def summarize_dataset(raw: pd.DataFrame, clean: pd.DataFrame) -> dict:
    """Facts about the dataset, computed (never typed in by hand)."""
    total_charges_raw = pd.to_numeric(raw["TotalCharges"].astype("object").str.strip(), errors="coerce")
    feature_cols = [c for c in raw.columns if c != config.ID_COL]
    return {
        "n_rows": int(len(raw)),
        "n_columns": int(raw.shape[1]),
        "n_unique_customers": int(raw[config.ID_COL].nunique()),
        "churn_rate": float(clean[config.TARGET].mean()),
        "n_churned": int(clean[config.TARGET].sum()),
        "n_retained": int((clean[config.TARGET] == 0).sum()),
        "duplicate_customer_ids": int(raw[config.ID_COL].duplicated().sum()),
        "duplicate_rows_ignoring_id": int(raw.duplicated(subset=feature_cols).sum()),
        "unparsable_total_charges": int(total_charges_raw.isna().sum()),
        "unparsable_total_charges_all_tenure_zero": bool(
            (raw.loc[total_charges_raw.isna(), "tenure"] == 0).all()
        ),
        "missing_values_after_cleaning": int(clean.isna().sum().sum()),
    }


# ----------------------------------------------------------------------------- splitting
def split_train_test(df: pd.DataFrame, test_size: float = config.TEST_SIZE,
                     random_state: int = config.RANDOM_STATE):
    """Stratified hold-out split. The test set is not touched until final evaluation.

    Caveat: a random split assumes customers are independent and exchangeable. The dataset
    has no timestamps, so we cannot do a (more realistic) out-of-time split.
    """
    return train_test_split(df, test_size=test_size, stratify=df[config.TARGET],
                            random_state=random_state)


def load_split(which: str) -> pd.DataFrame:
    """Load the saved development ('train') or held-out ('test') split."""
    path = {"train": config.TRAIN_PATH, "test": config.TEST_PATH}[which]
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.data_preprocessing` first.")
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame):
    """Separate features (raw feature columns only) from the target."""
    return df[config.RAW_FEATURES].copy(), df[config.TARGET].to_numpy()


# --------------------------------------------------------------------------- pipeline
def build_preprocessor(feature_set: str = "base", scale_numeric: bool = True) -> ColumnTransformer:
    """ColumnTransformer: impute -> (scale) numeric, impute -> one-hot categorical.

    scale_numeric=True  for Logistic Regression (regularised, gradient-based: features on
                        comparable scales make the penalty fair and optimisation stable).
    scale_numeric=False for tree models (splits are invariant to monotone rescaling).
    handle_unknown='ignore' means an unseen category at prediction time becomes all-zeros
    instead of crashing.
    """
    numeric_cols, categorical_cols = get_feature_columns(feature_set)
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    cat_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    return ColumnTransformer(
        [("num", Pipeline(num_steps), numeric_cols), ("cat", Pipeline(cat_steps), categorical_cols)],
        remainder="drop",            # drops customerID or any other extra column
        verbose_feature_names_out=True,
    )


def make_pipeline(classifier, feature_set: str = "base", scale_numeric: bool = True) -> Pipeline:
    """Full model = feature engineering -> preprocessing -> classifier, as ONE object.

    Passing this single object to cross_validate / GridSearchCV guarantees every
    transformer is re-fitted on the training folds only.
    """
    return Pipeline([
        ("features", FunctionTransformer(add_engineered_features, validate=False)),
        ("preprocess", build_preprocessor(feature_set, scale_numeric)),
        ("model", classifier),
    ])


# ------------------------------------------------------------------------------ script
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Clean the raw data and create train/test splits.")
    parser.add_argument("--download", action="store_true", help="download the raw CSV first")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.download and not config.RAW_DATA_PATH.exists():
        download_raw_data()
    raw = load_raw_data()
    validate_schema(raw)
    clean = clean_data(raw)

    summary = summarize_dataset(raw, clean)
    train_df, test_df = split_train_test(clean)
    summary.update({
        "n_train": int(len(train_df)), "n_test": int(len(test_df)),
        "train_churn_rate": float(train_df[config.TARGET].mean()),
        "test_churn_rate": float(test_df[config.TARGET].mean()),
        "test_size": config.TEST_SIZE, "random_state": config.RANDOM_STATE,
    })

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(config.TRAIN_PATH, index=False)
    test_df.to_csv(config.TEST_PATH, index=False)
    config.DATA_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    for key, value in summary.items():
        logger.info("%-42s %s", key, value)
    logger.info("Saved %s and %s", config.TRAIN_PATH.name, config.TEST_PATH.name)


if __name__ == "__main__":
    main()
