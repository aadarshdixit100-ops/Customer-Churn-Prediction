"""SHAP explanations for the final model (global + local).

Explainer choice
  * Logistic Regression -> shap.LinearExplainer   (exact for linear models; units = log-odds)
  * LightGBM / XGBoost  -> shap.TreeExplainer     (exact, fast; units = log-odds)
  * Random Forest       -> shap.TreeExplainer     (units = probability)

Run:  python -m src.explain     (needs `python -m src.train` first)
"""
import json
import logging
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

from src import config
from src.data_preprocessing import load_split, split_xy

logger = logging.getLogger(__name__)


def load_model_parts(model_path: Path = config.MODEL_PATH):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"{model_path} not found. Run `python -m src.train` first.")
    pipe = joblib.load(model_path)
    return pipe, pipe.named_steps["preprocess"], pipe.named_steps["model"]


def transform_features(pipe, X: pd.DataFrame) -> pd.DataFrame:
    """Apply the FITTED feature-engineering + preprocessing steps; return a named DataFrame."""
    Xt = pipe[:-1].transform(X)
    names = [n.split("__", 1)[1] for n in pipe.named_steps["preprocess"].get_feature_names_out()]
    return pd.DataFrame(Xt, columns=names, index=X.index)


def source_feature(encoded_name: str, categorical_cols) -> str:
    """Map an encoded column ('Contract_One year') back to its original feature ('Contract')."""
    for col in sorted(categorical_cols, key=len, reverse=True):
        if encoded_name.startswith(col + "_"):
            return col
    return encoded_name


def build_explainer(clf, background: pd.DataFrame):
    if isinstance(clf, LogisticRegression):
        return shap.LinearExplainer(clf, background), "log-odds"
    units = "probability" if clf.__class__.__name__ == "RandomForestClassifier" else "log-odds"
    return shap.TreeExplainer(clf), units


def _positive_class(values):
    """Normalise SHAP output shapes across versions/models to (n_samples, n_features)."""
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    return values[:, :, 1] if values.ndim == 3 else values


def compute_shap(pipe, X: pd.DataFrame, X_background: pd.DataFrame):
    """Return (Explanation, transformed X, units)."""
    _, _, clf = load_model_parts_from(pipe)
    Xt, Xb = transform_features(pipe, X), transform_features(pipe, X_background)
    explainer, units = build_explainer(clf, Xb)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="LightGBM binary classifier")  # benign: output-format notice
        values = _positive_class(explainer.shap_values(Xt))
    base = explainer.expected_value
    base = float(np.ravel(base)[-1]) if np.ndim(base) else float(base)
    expl = shap.Explanation(values=values, base_values=np.full(len(Xt), base),
                            data=Xt.values, feature_names=list(Xt.columns))
    return expl, Xt, units


def load_model_parts_from(pipe):
    return pipe, pipe.named_steps["preprocess"], pipe.named_steps["model"]


def importance_tables(expl, categorical_cols):
    """Mean |SHAP| per encoded column, and per ORIGINAL feature (dummies summed per customer first)."""
    names = np.array(expl.feature_names)
    per_col = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(expl.values).mean(axis=0)}
                           ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    sources = np.array([source_feature(n, categorical_cols) for n in names])
    agg = {s: np.abs(expl.values[:, sources == s].sum(axis=1)).mean() for s in dict.fromkeys(sources)}
    per_feature = pd.Series(agg, name="mean_abs_shap").sort_values(ascending=False).rename_axis("feature").reset_index()
    return per_col, per_feature


def plot_summary(expl, Xt, filename="shap_summary_beeswarm.png", max_display=15):
    plt.figure()
    shap.summary_plot(expl.values, Xt, max_display=max_display, show=False)
    plt.title("SHAP summary: direction and size of each feature's effect on predicted churn risk")
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    return fig


def plot_importance_bar(per_feature: pd.DataFrame, units: str, filename="shap_importance_bar.png", top=12):
    top_df = per_feature.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(top_df["feature"], top_df["mean_abs_shap"], color="#3b6ea5")
    ax.set(xlabel=f"Mean |SHAP value| ({units}) - dummies of one feature summed",
           title="Global importance by original feature")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    return fig


def plot_waterfall(expl, i: int, title: str, filename: str, max_display: int = 12):
    plt.figure()
    shap.plots.waterfall(expl[i], max_display=max_display, show=False)
    plt.title(title, fontsize=10)
    fig = plt.gcf()
    fig.savefig(config.FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    return fig


def pick_local_examples(y_true, prob, threshold: float) -> dict:
    """Pick instructive customers (by position): highest risk, lowest risk, nearest the threshold,
    a missed churner (false negative) and a false alarm (false positive)."""
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    picks = {"highest_risk": int(np.argmax(prob)), "lowest_risk": int(np.argmin(prob)),
             "near_threshold": int(np.argmin(np.abs(prob - threshold)))}
    fn = np.where((y_true == 1) & (prob < threshold))[0]
    if len(fn):
        picks["missed_churner_FN"] = int(fn[np.argsort(prob[fn])[len(fn) // 2]])  # median-probability FN
    fp = np.where((y_true == 0) & (prob >= threshold))[0]
    if len(fp):
        picks["false_alarm_FP"] = int(fp[np.argmax(prob[fp])])
    return picks


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pipe, _, clf = load_model_parts()
    meta = json.loads(Path(config.METADATA_PATH).read_text())
    train_df, test_df = load_split("train"), load_split("test")
    X_train, _ = split_xy(train_df)
    X_test, y_test = split_xy(test_df)

    # Explain predictions on UNSEEN customers (test set); background = a sample of training data.
    background = X_train.sample(n=min(500, len(X_train)), random_state=config.RANDOM_STATE)
    expl, Xt, units = compute_shap(pipe, X_test, background)
    cats = list(config.CATEGORICAL_FEATURES)
    if meta["feature_set"] == "engineered":
        cats.append("tenure_group")
    per_col, per_feature = importance_tables(expl, cats)
    per_col.to_csv(config.REPORTS_DIR / "shap_importance_by_column.csv", index=False)
    per_feature.to_csv(config.REPORTS_DIR / "shap_importance_by_feature.csv", index=False)
    plot_summary(expl, Xt)
    plot_importance_bar(per_feature, units)

    prob = pipe.predict_proba(X_test)[:, 1]
    picks = pick_local_examples(y_test, prob, meta["threshold"])
    rows = []
    for kind, i in picks.items():
        plot_waterfall(expl, i, f"{kind}: predicted churn probability {prob[i]:.2f} "
                                f"(actual: {'churned' if y_test[i] else 'stayed'})", f"shap_waterfall_{kind}.png")
        rows.append({"example": kind, "test_position": i, config.ID_COL: test_df.iloc[i][config.ID_COL],
                     "predicted_probability": prob[i], "actual_churn": int(y_test[i]),
                     **{c: test_df.iloc[i][c] for c in config.RAW_FEATURES}})
    pd.DataFrame(rows).to_csv(config.REPORTS_DIR / "shap_local_examples.csv", index=False)
    logger.info("SHAP units: %s | top features: %s", units, list(per_feature["feature"].head(6)))
    plt.close("all")


if __name__ == "__main__":
    main()
