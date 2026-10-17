"""Central configuration: paths, seeds, column groups and *illustrative* cost assumptions.

Keeping these in one place means notebooks, training, evaluation and the dashboard
can never silently disagree about, e.g., which random seed or which columns are used.
"""
from dataclasses import dataclass, asdict
from pathlib import Path

# --------------------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA_PATH = RAW_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
TRAIN_PATH = PROCESSED_DIR / "train.csv"          # development set (CV + tuning + threshold)
TEST_PATH = PROCESSED_DIR / "test.csv"            # held-out; touched once for final evaluation
OOF_PATH = PROCESSED_DIR / "oof_predictions.csv"  # out-of-fold probabilities on the dev set
DATA_SUMMARY_PATH = REPORTS_DIR / "data_summary.json"

MODEL_PATH = MODELS_DIR / "churn_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
CV_DEFAULT_PATH = REPORTS_DIR / "cv_results_default.csv"
CV_NESTED_PATH = REPORTS_DIR / "cv_results_tuned_nested.csv"
CV_FOLDS_PATH = REPORTS_DIR / "cv_fold_scores_tuned.csv"
TEST_METRICS_PATH = REPORTS_DIR / "test_metrics.json"
RISK_SCORES_PATH = REPORTS_DIR / "customer_risk_scores.csv"
EVAL_REPORT_PATH = REPORTS_DIR / "model_evaluation.md"

DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

# --------------------------------------------------------------------- reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5          # stratified CV folds on the development set
INNER_SPLITS = 3      # inner folds used for hyper-parameter search inside nested CV
OOF_RANDOM_STATE = 7  # different fold assignment than the one used for tuning

# ------------------------------------------------------------------------- columns
TARGET = "Churn"
ID_COL = "customerID"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
RAW_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
ADDON_SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies",
]
# Columns an analyst might hesitate to use for retention decisions (fairness review).
SENSITIVE_COLUMNS = ["gender", "SeniorCitizen"]


# --------------------------------------------------- illustrative business assumptions
@dataclass(frozen=True)
class CostAssumptions:
    """ILLUSTRATIVE numbers - NOT measured from the dataset.

    customer_value        : value (e.g. margin over the planning horizon) of keeping a
                            customer who would otherwise leave.
    intervention_cost     : cost of one retention action (call, discount, ...), paid for
                            every customer we flag, whether or not they would churn.
    intervention_success  : probability that the action actually retains a would-be churner.
    """
    customer_value: float = 400.0
    intervention_cost: float = 25.0
    intervention_success: float = 0.30

    def as_dict(self):
        return asdict(self)


DEFAULT_COSTS = CostAssumptions()
