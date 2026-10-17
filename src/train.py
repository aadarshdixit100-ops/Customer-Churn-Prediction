"""Model comparison, tuning, selection, threshold choice and final fit.

THIS SCRIPT NEVER READS THE TEST SET. Everything here uses the development set only.

Pipeline of decisions
  A. Default-config stratified CV for every candidate (incl. Dummy baseline).
  B. Limited random-search tuning, evaluated with NESTED CV (inner loop tunes, outer loop
     scores) so that tuned scores are not inflated by the search.
  C. Choose the final model with a stated rule: highest mean CV PR-AUC, but if several
     models are within one standard error of the best, prefer the simplest.
  D. Re-tune the winner on the full development set, produce out-of-fold probabilities
     (fold assignment different from the tuning folds).
  E. Choose the decision threshold on those out-of-fold probabilities by minimising expected
     cost under ILLUSTRATIVE cost assumptions.
  F. Save the fitted pipeline + metadata.

Run:  python -m src.train            (full; a few minutes)
      python -m src.train --quick    (smaller search; for smoke tests)
"""
import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_validate)
from xgboost import XGBClassifier

from src import config
from src.data_preprocessing import load_split, make_pipeline, split_xy
from src.evaluate import choose_threshold_by_cost, theoretical_threshold

logger = logging.getLogger(__name__)
DEFAULT_FEATURE_SET = "base"  # chosen by parsimony: the CV ablation showed no gain beyond noise from engineered features

SCORING = {
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
    "neg_brier": "neg_brier_score",
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
}


# --------------------------------------------------------------------- candidate models
def get_model_specs(seed: int = config.RANDOM_STATE) -> dict:
    """name -> estimator factory, scaling flag, search space, complexity rank.

    complexity is used ONLY as a tie-breaker (prefer simpler when scores are statistically
    indistinguishable). Estimators use n_jobs=1; parallelism is applied at the CV level.
    """
    return {
        "Dummy (prior)": dict(
            make=lambda: DummyClassifier(strategy="prior"), scale=False, complexity=0,
            params=None, n_iter=0),
        "LogisticRegression": dict(
            make=lambda: LogisticRegression(max_iter=3000, random_state=seed), scale=True, complexity=1,
            params={"model__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]}, n_iter=8),
        "RandomForest": dict(
            make=lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=1,
                                                random_state=seed),
            scale=False, complexity=2,
            params={"model__max_depth": [4, 6, 8, 12, None], "model__min_samples_leaf": [3, 5, 10, 20, 40],
                    "model__max_features": ["sqrt", 0.3, 0.5]}, n_iter=8),
        "LightGBM": dict(
            make=lambda: LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                        n_jobs=1, random_state=seed, verbose=-1),
            scale=False, complexity=3,
            params={"model__num_leaves": [4, 7, 15, 31], "model__learning_rate": [0.02, 0.05, 0.1],
                    "model__n_estimators": [100, 200, 300, 500], "model__min_child_samples": [10, 20, 40, 80],
                    "model__colsample_bytree": [0.5, 0.7, 0.9], "model__reg_lambda": [0, 1, 5, 10]}, n_iter=12),
        "XGBoost": dict(
            make=lambda: XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
                                       colsample_bytree=0.8, eval_metric="logloss", tree_method="hist",
                                       n_jobs=1, random_state=seed),
            scale=False, complexity=3,
            params={"model__max_depth": [2, 3, 4, 5], "model__learning_rate": [0.02, 0.05, 0.1],
                    "model__n_estimators": [100, 200, 300, 500], "model__min_child_weight": [1, 5, 10],
                    "model__subsample": [0.7, 0.85, 1.0], "model__colsample_bytree": [0.5, 0.7, 0.9],
                    "model__reg_lambda": [1, 5, 10]}, n_iter=12),
    }


def build_pipeline(name: str, feature_set: str, spec: dict | None = None):
    spec = spec or get_model_specs()[name]
    return make_pipeline(spec["make"](), feature_set, spec["scale"])


def _summarise(name: str, cv_out: dict, extra: dict | None = None) -> dict:
    row = {"model": name, **(extra or {})}
    for key, values in cv_out.items():
        if key.startswith(("test_", "train_")):
            row[f"{key.replace('test_', '')}_mean"] = float(np.mean(values))
            row[f"{key.replace('test_', '')}_std"] = float(np.std(values, ddof=1))
    row["fit_time_s"] = float(np.mean(cv_out["fit_time"]))
    return row


# ------------------------------------------------------------------------------ stage A
def run_default_cv(X, y, feature_set: str, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Default hyper-parameters, stratified 5-fold CV. Also compares class_weight='balanced'
    (an imbalance-handling option) with the standard unweighted model."""
    specs = get_model_specs(seed)
    specs["LogisticRegression (balanced)"] = dict(
        make=lambda: LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
        scale=True, complexity=1)
    cv = StratifiedKFold(config.N_SPLITS, shuffle=True, random_state=seed)
    rows = []
    for name, spec in specs.items():
        t0 = time.time()
        out = cross_validate(build_pipeline(name, feature_set, spec), X, y, cv=cv, scoring=SCORING,
                             return_train_score=True, n_jobs=-1)
        rows.append(_summarise(name, out))
        logger.info("[default CV] %-32s AP=%.4f AUC=%.4f (%.0fs)", name,
                    rows[-1]["average_precision_mean"], rows[-1]["roc_auc_mean"], time.time() - t0)
    return pd.DataFrame(rows)


def feature_set_ablation(X, y, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Do the engineered features help? Let CV (not intuition, not the test set) decide."""
    specs = get_model_specs(seed)
    cv = StratifiedKFold(config.N_SPLITS, shuffle=True, random_state=seed)
    rows = []
    for name in ["LogisticRegression", "RandomForest", "LightGBM", "XGBoost"]:
        for fs in ["base", "engineered"]:
            out = cross_validate(build_pipeline(name, fs, specs[name]), X, y, cv=cv, scoring=SCORING, n_jobs=-1)
            rows.append(_summarise(name, out, {"feature_set": fs}))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ stage B
def run_nested_cv(X, y, feature_set: str, seed: int = config.RANDOM_STATE, quick: bool = False):
    """Outer 5-fold CV scores a *procedure*: 'tune with inner 3-fold random search, then predict'.
    Returns (summary table, per-fold scores)."""
    specs = get_model_specs(seed)
    outer = StratifiedKFold(config.N_SPLITS, shuffle=True, random_state=seed)
    inner = StratifiedKFold(config.INNER_SPLITS, shuffle=True, random_state=seed + 1)
    rows, fold_rows = [], []
    for name, spec in specs.items():
        t0 = time.time()
        pipe = build_pipeline(name, feature_set, spec)
        if spec["params"] is None:
            estimator = pipe
        else:
            n_iter = 2 if quick else spec["n_iter"]
            estimator = RandomizedSearchCV(pipe, spec["params"], n_iter=n_iter, scoring="average_precision",
                                           cv=inner, random_state=seed, n_jobs=1, refit=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # "search space smaller than n_iter"
            out = cross_validate(estimator, X, y, cv=outer, scoring=SCORING, n_jobs=-1)
        rows.append(_summarise(name, out, {"complexity": spec["complexity"]}))
        for k in range(config.N_SPLITS):
            fold_rows.append({"model": name, "fold": k, **{m: out[f"test_{m}"][k] for m in SCORING}})
        logger.info("[nested CV]  %-32s AP=%.4f +/- %.4f AUC=%.4f (%.0fs)", name,
                    rows[-1]["average_precision_mean"], rows[-1]["average_precision_std"],
                    rows[-1]["roc_auc_mean"], time.time() - t0)
    return pd.DataFrame(rows), pd.DataFrame(fold_rows)


# ------------------------------------------------------------------------------ stage C
def select_model(results: pd.DataFrame, n_folds: int = config.N_SPLITS):
    """Stated selection rule (document it, then follow it):

    1. Rank non-baseline models by mean cross-validated PR-AUC (average precision).
    2. Let SE = std / sqrt(k) of the best model. Every model whose mean is within one SE of
       the best is considered statistically indistinguishable (the 'one-standard-error rule').
    3. Among those, choose the simplest (lowest complexity rank); break remaining ties by
       higher mean PR-AUC.
    Folds share data so SE is only a rough guide; it guards against picking the winner of a noisy race.
    """
    cand = results[results["model"] != "Dummy (prior)"].copy()
    best = cand.loc[cand["average_precision_mean"].idxmax()]
    se = best["average_precision_std"] / np.sqrt(n_folds)
    within = cand[cand["average_precision_mean"] >= best["average_precision_mean"] - se]
    chosen = within.sort_values(["complexity", "average_precision_mean"], ascending=[True, False]).iloc[0]
    rule = (f"highest CV PR-AUC ({best['model']}, {best['average_precision_mean']:.4f}); "
            f"within 1 SE ({se:.4f}) -> {list(within['model'])}; simplest chosen -> {chosen['model']}")
    return str(chosen["model"]), rule


# --------------------------------------------------------------------------------- main
def fit_final(X, y, name: str, feature_set: str, seed: int, quick: bool):
    """Re-tune the chosen model on the whole development set and refit."""
    spec = get_model_specs(seed)[name]
    pipe = build_pipeline(name, feature_set, spec)
    cv = StratifiedKFold(config.N_SPLITS, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(pipe, spec["params"], n_iter=2 if quick else spec["n_iter"],
                                scoring="average_precision", cv=cv, random_state=seed,
                                n_jobs=-1, refit=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        search.fit(X, y)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def build_schema(X: pd.DataFrame) -> dict:
    """What valid inputs look like, learned from the development data (used by predict.py + the app)."""
    return {
        "categorical": {c: sorted(map(str, X[c].dropna().unique())) for c in config.CATEGORICAL_FEATURES},
        "numeric": {c: {"min": float(X[c].min()), "max": float(X[c].max())} for c in config.NUMERIC_FEATURES},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--feature-set", default=DEFAULT_FEATURE_SET, choices=["base", "engineered"])
    parser.add_argument("--quick", action="store_true", help="tiny search budget (smoke test only)")
    parser.add_argument("--max-flag-fraction", type=float, default=None,
                        help="optional capacity limit: max share of customers the team can contact")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    train_df = load_split("train")                       # development set only
    X, y = split_xy(train_df)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    config.MODELS_DIR.mkdir(exist_ok=True)
    logger.info("Development set: %d customers, churn rate %.3f, feature set '%s'",
                len(y), y.mean(), args.feature_set)

    default_cv = run_default_cv(X, y, args.feature_set)
    default_cv.to_csv(config.CV_DEFAULT_PATH, index=False)

    nested, folds = run_nested_cv(X, y, args.feature_set, quick=args.quick)
    nested.to_csv(config.CV_NESTED_PATH, index=False)
    folds.to_csv(config.CV_FOLDS_PATH, index=False)

    chosen, rule = select_model(nested)
    logger.info("SELECTED MODEL: %s | %s", chosen, rule)

    final_model, best_params, search_score = fit_final(X, y, chosen, args.feature_set,
                                                       config.RANDOM_STATE, args.quick)
    logger.info("Best params: %s (inner-CV AP %.4f)", best_params, search_score)

    # out-of-fold probabilities for the development set, using the tuned configuration
    oof_cv = StratifiedKFold(config.N_SPLITS, shuffle=True, random_state=config.OOF_RANDOM_STATE)
    oof_prob = cross_val_predict(clone(final_model), X, y, cv=oof_cv, method="predict_proba", n_jobs=-1)[:, 1]
    pd.DataFrame({config.ID_COL: train_df[config.ID_COL], "churn": y, "oof_probability": oof_prob}
                 ).to_csv(config.OOF_PATH, index=False)

    costs = config.DEFAULT_COSTS
    threshold, _ = choose_threshold_by_cost(y, oof_prob, costs, max_flag_fraction=args.max_flag_fraction)
    logger.info("Cost-optimal threshold on out-of-fold data: %.2f (calibrated-theory C/(sV)=%.2f)",
                threshold, theoretical_threshold(costs))

    joblib.dump(final_model, config.MODEL_PATH)
    metadata = {
        "model_name": chosen, "feature_set": args.feature_set, "best_params": {k: (v.item() if hasattr(v, "item") else v) for k, v in best_params.items()},
        "selection_rule": rule, "threshold": threshold,
        "threshold_method": "minimise expected cost on out-of-fold development predictions"
                            + (f", flagged share <= {args.max_flag_fraction:.0%}" if args.max_flag_fraction else ""),
        "cost_assumptions": costs.as_dict(), "cost_assumptions_note": "ILLUSTRATIVE - not measured from data",
        "n_train": int(len(y)), "train_churn_rate": float(y.mean()),
        "random_state": config.RANDOM_STATE, "schema": build_schema(X),
        "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    Path(config.METADATA_PATH).write_text(json.dumps(metadata, indent=2))
    logger.info("Saved %s and %s", config.MODEL_PATH.name, config.METADATA_PATH.name)


if __name__ == "__main__":
    main()
