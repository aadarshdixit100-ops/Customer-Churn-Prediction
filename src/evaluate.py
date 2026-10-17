"""Evaluation: metrics, threshold analysis, business-cost framework, calibration, plots.

Two ways to use this module
  * as a library (notebooks, train.py import the functions), and
  * as a script:  python -m src.evaluate
    -> evaluates the FINAL saved model on the held-out test set exactly once and
       writes reports/test_metrics.json, reports/model_evaluation.md and figures.

Rule of this project: model choice and decision threshold are fixed using
cross-validation / out-of-fold predictions on the development set BEFORE this script runs.
"""
import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from src import config
from src.config import CostAssumptions
from src.data_preprocessing import load_split, split_xy

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", context="notebook")
DEFAULT_GRID = np.round(np.arange(0.02, 0.99, 0.01), 2)


# ================================================================== metrics
def confusion_counts(y_true, y_prob, threshold: float) -> dict:
    """TP/FP/TN/FN when we flag every customer with probability >= threshold."""
    y_true = np.asarray(y_true)
    flagged = np.asarray(y_prob) >= threshold
    return {
        "tp": int(np.sum(flagged & (y_true == 1))), "fp": int(np.sum(flagged & (y_true == 0))),
        "tn": int(np.sum(~flagged & (y_true == 0))), "fn": int(np.sum(~flagged & (y_true == 1))),
    }


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Threshold-free (ranking / probability) metrics plus threshold-dependent ones."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_prob)) > 1 else 0.5,
        "average_precision": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "prevalence": float(y_true.mean()),
        "flagged_fraction": float(y_pred.mean()),
    }
    out.update(confusion_counts(y_true, y_prob, threshold))
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def threshold_table(y_true, y_prob, thresholds=DEFAULT_GRID) -> pd.DataFrame:
    """Precision / recall / F1 / confusion counts for each candidate threshold."""
    rows = []
    for t in thresholds:
        c = confusion_counts(y_true, y_prob, t)
        prec = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else np.nan
        rec = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else np.nan
        f1 = 2 * prec * rec / (prec + rec) if (prec and rec and not np.isnan(prec)) else 0.0
        rows.append({"threshold": float(t), "flagged_pct": 100 * (c["tp"] + c["fp"]) / len(y_true),
                     "precision": prec, "recall": rec, "f1": f1, **c})
    return pd.DataFrame(rows)


# ================================================================== business cost framework
def baseline_costs(y_true, a: CostAssumptions) -> dict:
    """Expected total cost of two trivial policies, under the stated assumptions."""
    y_true = np.asarray(y_true)
    n, pos = len(y_true), int(y_true.sum())
    return {
        "no_action": pos * a.customer_value,
        "flag_everyone": n * a.intervention_cost + pos * (1 - a.intervention_success) * a.customer_value,
    }


def theoretical_threshold(a: CostAssumptions) -> float:
    """If probabilities were perfectly calibrated, flag a customer iff
        p * success * value  >  intervention_cost   <=>   p > cost / (success * value).
    Returns a value possibly > 1 (meaning: never worth intervening)."""
    return a.intervention_cost / (a.intervention_success * a.customer_value)


def cost_table(y_true, y_prob, a: CostAssumptions = config.DEFAULT_COSTS,
               thresholds=DEFAULT_GRID) -> pd.DataFrame:
    """Expected cost per threshold.

    Expected total cost = FN * V                       (missed churners are lost)
                       + TP * (C + (1 - s) * V)        (we pay C; with prob. 1-s the churner still leaves)
                       + FP * C                        (wasted intervention)
    Net savings vs doing nothing = s * V * TP - C * (TP + FP).
    All of V, C, s are ASSUMPTIONS (see CostAssumptions).
    """
    tab = threshold_table(y_true, y_prob, thresholds)
    V, C, s = a.customer_value, a.intervention_cost, a.intervention_success
    tab["expected_total_cost"] = tab["fn"] * V + tab["tp"] * (C + (1 - s) * V) + tab["fp"] * C
    tab["net_savings_vs_no_action"] = s * V * tab["tp"] - C * (tab["tp"] + tab["fp"])
    tab["net_savings_per_1000"] = 1000 * tab["net_savings_vs_no_action"] / len(y_true)
    return tab


def choose_threshold_by_cost(y_true, y_prob, a: CostAssumptions = config.DEFAULT_COSTS,
                             thresholds=DEFAULT_GRID, max_flag_fraction=None):
    """Threshold minimising expected cost, optionally subject to a capacity limit
    (e.g. the retention team can only contact `max_flag_fraction` of customers)."""
    tab = cost_table(y_true, y_prob, a, thresholds)
    feasible = tab if max_flag_fraction is None else tab[tab["flagged_pct"] <= 100 * max_flag_fraction]
    if feasible.empty:
        raise ValueError("No threshold satisfies the capacity constraint.")
    best = feasible.loc[feasible["expected_total_cost"].idxmin()]
    return float(best["threshold"]), tab


def cost_sensitivity(y_true, y_prob, intervention_costs, success_rates,
                     customer_value: float = config.DEFAULT_COSTS.customer_value,
                     thresholds=DEFAULT_GRID):
    """How the cost-optimal threshold and savings move when assumptions change."""
    best_t = pd.DataFrame(index=intervention_costs, columns=success_rates, dtype=float)
    savings = best_t.copy()
    for c in intervention_costs:
        for s in success_rates:
            a = CostAssumptions(customer_value, c, s)
            t, tab = choose_threshold_by_cost(y_true, y_prob, a, thresholds)
            best_t.loc[c, s] = t
            savings.loc[c, s] = tab.loc[np.isclose(tab["threshold"], t), "net_savings_per_1000"].iloc[0]
    for df in (best_t, savings):
        df.index.name, df.columns.name = "intervention_cost", "success_rate"
    return best_t, savings


# ================================================================== uncertainty & fairness
def bootstrap_ci(y_true, y_prob, threshold, a: CostAssumptions = config.DEFAULT_COSTS,
                 n_boot: int = 1000, seed: int = config.RANDOM_STATE) -> dict:
    """Percentile bootstrap 95% CIs on the test set (resampling customers).
    Quantifies sampling uncertainty from a finite test set; it does NOT capture
    uncertainty from the model fit or from distribution shift."""
    rng = np.random.default_rng(seed)
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    n = len(y_true)
    draws = {k: [] for k in ["roc_auc", "average_precision", "precision", "recall", "f1",
                             "brier", "net_savings_per_1000"]}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if yt.min() == yt.max():
            continue
        m = compute_metrics(yt, yp, threshold)
        for k in ["roc_auc", "average_precision", "precision", "recall", "f1", "brier"]:
            draws[k].append(m[k])
        draws["net_savings_per_1000"].append(
            1000 * (a.intervention_success * a.customer_value * m["tp"]
                    - a.intervention_cost * (m["tp"] + m["fp"])) / n)
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in draws.items()}


def subgroup_metrics(df: pd.DataFrame, y_true, y_prob, threshold: float, column: str) -> pd.DataFrame:
    """Descriptive performance by subgroup (a basic fairness / robustness check).
    Small groups give noisy estimates - read as a screening tool, not as proof."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    rows = []
    for g, idx in df.reset_index(drop=True).groupby(column).groups.items():
        idx = np.asarray(list(idx))
        yt, yp = y_true[idx], y_prob[idx]
        m = compute_metrics(yt, yp, threshold)
        rows.append({column: g, "n": len(idx), "churn_rate": yt.mean(), "flagged_rate": m["flagged_fraction"],
                     "precision": m["precision"], "recall": m["recall"], "mean_predicted_prob": yp.mean(),
                     "roc_auc": roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else np.nan})
    return pd.DataFrame(rows)


def calibration_table(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted_probability": mean_pred, "observed_churn_rate": frac_pos})


# ================================================================== plots
def _save(fig, filename: str | None):
    if filename:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    return fig


def plot_pr_roc(y_true, y_prob, title: str, filename: str | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    axes[0].plot(rec, prec, lw=2, label=f"model (AP = {ap:.3f})")
    axes[0].axhline(np.mean(y_true), ls="--", c="grey", label=f"no-skill = prevalence ({np.mean(y_true):.3f})")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve", ylim=(0, 1.02))
    axes[0].legend(loc="upper right")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    axes[1].plot(fpr, tpr, lw=2, label=f"model (AUC = {auc:.3f})")
    axes[1].plot([0, 1], [0, 1], "--", c="grey", label="no-skill")
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[1].legend(loc="lower right")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return _save(fig, filename)


def plot_confusion_matrix(y_true, y_prob, threshold: float, title: str, filename: str | None = None):
    c = confusion_counts(y_true, y_prob, threshold)
    cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Predicted: stays", "Predicted: churns"],
                yticklabels=["Actually stayed", "Actually churned"])
    ax.set_title(f"{title}\n(threshold = {threshold:.2f})")
    fig.tight_layout()
    return _save(fig, filename)


def plot_calibration(sets: dict, filename: str | None = None, n_bins: int = 10):
    """sets = {label: (y_true, y_prob)}. Reliability diagram + score histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].plot([0, 1], [0, 1], "--", c="grey", label="perfectly calibrated")
    for label, (y, p) in sets.items():
        tab = calibration_table(y, p, n_bins)
        axes[0].plot(tab.mean_predicted_probability, tab.observed_churn_rate, "o-",
                     label=f"{label} (Brier {brier_score_loss(y, p):.3f})")
        axes[1].hist(p, bins=30, alpha=0.55, label=label)
    axes[0].set(xlabel="Mean predicted churn probability (bin)", ylabel="Observed churn rate (bin)",
                title="Calibration (reliability) curve")
    axes[0].legend()
    axes[1].set(xlabel="Predicted churn probability", ylabel="Customers", title="Distribution of risk scores")
    axes[1].legend()
    fig.tight_layout()
    return _save(fig, filename)


def plot_threshold_pr(tab: pd.DataFrame, marks: dict | None = None, filename: str | None = None):
    """Precision, recall, F1 and share of customers flagged versus threshold."""
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(tab.threshold, tab.precision, label="Precision", lw=2)
    ax.plot(tab.threshold, tab.recall, label="Recall", lw=2)
    ax.plot(tab.threshold, tab.f1, label="F1", lw=2, ls=":")
    ax.plot(tab.threshold, tab.flagged_pct / 100, label="Share of customers flagged", lw=1.5, ls="--", c="grey")
    for label, t in (marks or {}).items():
        ax.axvline(t, c="k", alpha=0.35, ls="-.")
        ax.text(t, 1.015, label, rotation=0, ha="center", fontsize=8)
    ax.set(xlabel="Decision threshold", ylabel="Value", ylim=(0, 1.08),
           title="Precision / recall trade-off as the threshold moves")
    ax.legend(loc="center right")
    fig.tight_layout()
    return _save(fig, filename)


def plot_cost_curve(tab: pd.DataFrame, a: CostAssumptions, chosen: float, filename: str | None = None):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(tab.threshold, tab.net_savings_per_1000, lw=2)
    ax.axhline(0, c="grey", lw=1)
    ax.axvline(chosen, c="C3", ls="--", label=f"cost-optimal threshold = {chosen:.2f}")
    t_star = theoretical_threshold(a)
    if t_star < 1:
        ax.axvline(t_star, c="C2", ls=":", label=f"calibrated-theory threshold C/(sV) = {t_star:.2f}")
    ax.set(xlabel="Decision threshold", ylabel="Net savings per 1,000 customers ($)",
           title="ILLUSTRATIVE expected net savings vs. doing nothing\n"
                 f"(V=${a.customer_value:.0f}, C=${a.intervention_cost:.0f}, success={a.intervention_success:.0%})")
    ax.legend()
    fig.tight_layout()
    return _save(fig, filename)


def plot_sensitivity(best_t: pd.DataFrame, savings: pd.DataFrame, filename: str | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    sns.heatmap(best_t.astype(float), annot=True, fmt=".2f", cmap="viridis", ax=axes[0], cbar=False)
    axes[0].set_title("Cost-optimal threshold")
    sns.heatmap(savings.astype(float), annot=True, fmt=".0f", cmap="RdYlGn", center=0, ax=axes[1], cbar=False)
    axes[1].set_title("Net savings per 1,000 customers ($)")
    for ax in axes:
        ax.set_xlabel("Intervention success rate")
        ax.set_ylabel("Intervention cost ($)")
    fig.suptitle("Sensitivity of the threshold decision to ILLUSTRATIVE assumptions", y=1.03)
    fig.tight_layout()
    return _save(fig, filename)


# ================================================================== report helpers
def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".3f", index: bool = False) -> str:
    """Tiny markdown table writer (avoids an extra 'tabulate' dependency)."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in d.itertuples(index=False):   # itertuples keeps each column's own dtype
        cells = [format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _fmt_ci(value: float, ci: tuple, pct: bool = False, money: bool = False) -> str:
    if pct:
        return f"{value:.1%} (95% CI {ci[0]:.1%} to {ci[1]:.1%})"
    if money:
        return f"${value:,.0f} (95% CI ${ci[0]:,.0f} to ${ci[1]:,.0f})"
    return f"{value:.3f} (95% CI {ci[0]:.3f} to {ci[1]:.3f})"


# ================================================================== final evaluation script
def evaluate_final_model() -> dict:
    """Evaluate the saved final model ONCE on the held-out test set."""
    from sklearn.dummy import DummyClassifier
    from src.data_preprocessing import make_pipeline

    for path in (config.MODEL_PATH, config.METADATA_PATH, config.OOF_PATH):
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} not found. Run `python -m src.train` first.")
    model = joblib.load(config.MODEL_PATH)
    meta = json.loads(Path(config.METADATA_PATH).read_text())
    costs = CostAssumptions(**meta["cost_assumptions"])
    threshold = meta["threshold"]  # fixed BEFORE looking at test data

    train_df, test_df = load_split("train"), load_split("test")
    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)
    p_test = model.predict_proba(X_test)[:, 1]

    # trivial baseline, trained on the same development data
    dummy = make_pipeline(DummyClassifier(strategy="prior"), meta["feature_set"], False).fit(X_train, y_train)
    p_dummy = dummy.predict_proba(X_test)[:, 1]

    oof = pd.read_csv(config.OOF_PATH)
    results = {
        "model_name": meta["model_name"], "threshold": threshold,
        "threshold_method": meta["threshold_method"], "cost_assumptions": costs.as_dict(),
        "n_test": int(len(y_test)), "test_prevalence": float(y_test.mean()),
        "final_at_chosen_threshold": compute_metrics(y_test, p_test, threshold),
        "final_at_0.5": compute_metrics(y_test, p_test, 0.5),
        "dummy_at_0.5": compute_metrics(y_test, p_dummy, 0.5),
        "oof_at_chosen_threshold": compute_metrics(oof["churn"], oof["oof_probability"], threshold),
        "bootstrap_95ci": bootstrap_ci(y_test, p_test, threshold, costs),
        "baseline_costs_test": baseline_costs(y_test, costs),
    }
    tab_test = cost_table(y_test, p_test, costs)
    row = tab_test.loc[np.isclose(tab_test.threshold, threshold)].iloc[0]
    results["cost_at_chosen_threshold_test"] = {
        "expected_total_cost": float(row.expected_total_cost),
        "net_savings_vs_no_action": float(row.net_savings_vs_no_action),
        "net_savings_per_1000": float(row.net_savings_per_1000)}
    row5 = tab_test.loc[np.isclose(tab_test.threshold, 0.5)].iloc[0]
    results["cost_at_0.5_test"] = {"net_savings_per_1000": float(row5.net_savings_per_1000)}
    results["flag_everyone_net_savings_per_1000"] = 1000 * (
        results["baseline_costs_test"]["no_action"] - results["baseline_costs_test"]["flag_everyone"]) / len(y_test)

    # ---- figures (test set; threshold already fixed)
    plot_pr_roc(y_test, p_test, "Held-out test set", "test_pr_roc_curves.png")
    plot_confusion_matrix(y_test, p_test, threshold, "Test confusion matrix (chosen threshold)",
                          "test_confusion_matrix_chosen.png")
    plot_confusion_matrix(y_test, p_test, 0.5, "Test confusion matrix (default 0.5)",
                          "test_confusion_matrix_default.png")
    plot_calibration({"out-of-fold (dev)": (oof["churn"], oof["oof_probability"]),
                      "test": (y_test, p_test)}, "calibration_curve.png")
    plot_threshold_pr(threshold_table(oof["churn"], oof["oof_probability"]),
                      {"cost-optimal": threshold, "default 0.5": 0.5}, "threshold_precision_recall_oof.png")
    best_t_oof, tab_oof = choose_threshold_by_cost(oof["churn"], oof["oof_probability"], costs)
    plot_cost_curve(tab_oof, costs, threshold, "threshold_cost_curve_oof.png")

    # ---- diagnostics for the report (descriptive, not used to change anything)
    sub = {c: subgroup_metrics(test_df, y_test, p_test, threshold, c) for c in config.SENSITIVE_COLUMNS}
    test_thresholds = tab_test[tab_test.threshold.isin([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])]

    # ---- customer-level risk scores: every customer gets an OUT-OF-SAMPLE score
    dev_scores = pd.DataFrame({
        config.ID_COL: train_df[config.ID_COL].values, "churn_probability": oof["oof_probability"].values,
        "score_source": "out-of-fold (development set)", "actual_churn": oof["churn"].values})
    test_scores = pd.DataFrame({
        config.ID_COL: test_df[config.ID_COL].values, "churn_probability": p_test,
        "score_source": "final model (held-out test set)", "actual_churn": y_test})
    scores = pd.concat([dev_scores, test_scores], ignore_index=True)
    scores["flagged_at_threshold"] = scores["churn_probability"] >= threshold
    scores["risk_rank_percentile"] = scores["churn_probability"].rank(pct=True)
    scores.sort_values("churn_probability", ascending=False).to_csv(config.RISK_SCORES_PATH, index=False)

    config.TEST_METRICS_PATH.write_text(json.dumps(results, indent=2))
    _write_report(results, meta, test_thresholds, sub, tab_oof, costs)
    return results


def _write_report(res, meta, test_thresholds, sub, tab_oof, costs) -> None:
    ci, f, f5, d = res["bootstrap_95ci"], res["final_at_chosen_threshold"], res["final_at_0.5"], res["dummy_at_0.5"]
    oof = res["oof_at_chosen_threshold"]
    cv_default = pd.read_csv(config.CV_DEFAULT_PATH)
    cv_nested = pd.read_csv(config.CV_NESTED_PATH)
    cols = ["model", "average_precision_mean", "average_precision_std", "roc_auc_mean", "neg_brier_mean"]
    cv_nested_show = cv_nested[cols].copy()
    cv_nested_show["brier"] = -cv_nested_show.pop("neg_brier_mean")
    cv_default_show = cv_default[cols].copy()
    cv_default_show["brier"] = -cv_default_show.pop("neg_brier_mean")
    cost = res["cost_at_chosen_threshold_test"]
    near_best = tab_oof[tab_oof["net_savings_per_1000"] >= 0.98 * tab_oof["net_savings_per_1000"].max()]
    flat_note = (f"On the development (out-of-fold) data, every threshold from {near_best.threshold.min():.2f} to "
                 f"{near_best.threshold.max():.2f} gives expected savings within 2% of the best - the cost curve is flat near its "
                 f"optimum, so the exact threshold matters much less than choosing a sensible region and getting the costs right.")
    within = ci["average_precision"][0] <= oof["average_precision"] <= ci["average_precision"][1]

    md = f"""# Model evaluation report

*Generated by `python -m src.evaluate`. Every number below is computed from the saved model and data -
nothing is typed in by hand. Re-running with different library versions may change values slightly.*

## 1. Setup
- Dataset: IBM Telco Customer Churn sample dataset; development set n = {meta['n_train']}, held-out test n = {res['n_test']}.
- Test prevalence of churn: {res['test_prevalence']:.1%}.
- **Final model:** `{res['model_name']}` with feature set `{meta['feature_set']}`.
  Selection rule: {meta['selection_rule']}
- **Decision threshold:** {res['threshold']:.2f}, chosen by the rule "{res['threshold_method']}" - i.e. on out-of-fold development predictions, minimising expected
  cost under *illustrative* assumptions. It was fixed before the test set was scored.

## 2. Cross-validation on the development set (model comparison)
Stratified {config.N_SPLITS}-fold CV. Threshold-free metrics are used for model selection.

**Default hyper-parameters**

{df_to_markdown(cv_default_show)}

**Tuned (nested CV: {config.INNER_SPLITS}-fold random search inside each of {config.N_SPLITS} outer folds - an approximately unbiased estimate of the *tuning procedure*)**

{df_to_markdown(cv_nested_show)}

## 3. Held-out test results (scored once)
| Metric | Final model @ chosen threshold | Final model @ 0.50 | Dummy baseline @ 0.50 |
|---|---|---|---|
| Accuracy | {f['accuracy']:.3f} | {f5['accuracy']:.3f} | {d['accuracy']:.3f} |
| Precision | {f['precision']:.3f} | {f5['precision']:.3f} | {d['precision']:.3f} |
| Recall | {f['recall']:.3f} | {f5['recall']:.3f} | {d['recall']:.3f} |
| F1 | {f['f1']:.3f} | {f5['f1']:.3f} | {d['f1']:.3f} |
| Share of customers flagged | {f['flagged_fraction']:.1%} | {f5['flagged_fraction']:.1%} | {d['flagged_fraction']:.1%} |
| ROC-AUC (threshold-free) | {f['roc_auc']:.3f} | {f5['roc_auc']:.3f} | {d['roc_auc']:.3f} |
| PR-AUC / Average precision (threshold-free) | {f['average_precision']:.3f} | {f5['average_precision']:.3f} | {d['average_precision']:.3f} |
| Brier score (lower is better) | {f['brier']:.3f} | {f5['brier']:.3f} | {d['brier']:.3f} |

Confusion matrix at the chosen threshold: TP = {f['tp']}, FP = {f['fp']}, TN = {f['tn']}, FN = {f['fn']}.

**Uncertainty (bootstrap over the {res['n_test']} test customers):**
- ROC-AUC: {_fmt_ci(f['roc_auc'], ci['roc_auc'])}
- PR-AUC: {_fmt_ci(f['average_precision'], ci['average_precision'])}
- Precision @ threshold: {_fmt_ci(f['precision'], ci['precision'], pct=True)}
- Recall @ threshold: {_fmt_ci(f['recall'], ci['recall'], pct=True)}
- Brier: {_fmt_ci(f['brier'], ci['brier'])}

Consistency check: the out-of-fold development PR-AUC ({oof['average_precision']:.3f}) {'lies inside' if within else 'lies OUTSIDE'} the test-set 95% interval, i.e. the test result {'is consistent with' if within else 'differs from'} the cross-validation estimate.

Reminder: the dummy baseline's accuracy of {d['accuracy']:.1%} comes from always predicting "stays" - this is why accuracy alone is misleading here.

## 4. Threshold analysis (ILLUSTRATIVE costs - assumptions, not measurements)
Assumed: customer value V = ${costs.customer_value:,.0f}; intervention cost C = ${costs.intervention_cost:,.0f} per flagged customer;
intervention success probability s = {costs.intervention_success:.0%}.
If probabilities were perfectly calibrated the break-even rule is *flag if p > C/(sV)* = {theoretical_threshold(costs):.2f}.

On the held-out test set, at the pre-chosen threshold {res['threshold']:.2f}, the framework gives an expected net saving of
**${cost['net_savings_vs_no_action']:,.0f}** (${cost['net_savings_per_1000']:,.0f} per 1,000 customers) versus doing nothing,
compared with ${res['cost_at_0.5_test']['net_savings_per_1000']:,.0f} per 1,000 at the default 0.50 threshold and
${res['flag_everyone_net_savings_per_1000']:,.0f} per 1,000 for flagging everyone.
These are model-based expectations under assumed costs; they are **not** measured retention gains.

{flat_note}

Test-set metrics across thresholds (descriptive only - not used for selection):

{df_to_markdown(test_thresholds[['threshold','flagged_pct','precision','recall','f1','tp','fp','tn','fn','net_savings_per_1000']])}

Threshold choice changes *decisions* (who is flagged); it does not change the model's ranking of customers, so ROC-AUC and PR-AUC are identical for every threshold.

## 5. Subgroup check (descriptive; test set at the chosen threshold)
""" 
    for col, tab in sub.items():
        md += f"\n**{col}**\n\n{df_to_markdown(tab)}\n"
    md += """
Groups differ in base churn rate, so differing flag rates are expected; compare precision/recall and look for large gaps.
Small groups give noisy estimates. Using attributes such as gender or age to decide who receives retention offers raises fairness and
legal questions that should be reviewed before any real deployment.

## 6. Limitations
- Single random split of one snapshot dataset: no temporal validation, no check for drift.
- Dataset is an IBM *sample* (fictional telecom) - results do not transfer to a real operator.
- Churn labels are observational; associations (here and in SHAP) are not causal effects of interventions.
- Cost figures are illustrative; real value, cost and success rates must be measured (ideally with an A/B test).
- Test metrics have sampling uncertainty (see confidence intervals); small differences between models are not meaningful.
"""
    config.EVAL_REPORT_PATH.write_text(md)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = evaluate_final_model()
    f = res["final_at_chosen_threshold"]
    logger.info("Test: AP=%.3f ROC-AUC=%.3f precision=%.3f recall=%.3f (threshold=%.2f)",
                f["average_precision"], f["roc_auc"], f["precision"], f["recall"], res["threshold"])
    logger.info("Wrote %s", config.EVAL_REPORT_PATH)
    plt.close("all")


if __name__ == "__main__":
    main()
