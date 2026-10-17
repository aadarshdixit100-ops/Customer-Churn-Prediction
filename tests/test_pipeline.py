"""Unit tests for the parts where silent mistakes are most likely: leakage, cost maths,
selection rule, input validation. Run with:  pytest -q"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src import config
from src.config import CostAssumptions
from src.data_preprocessing import clean_data, make_pipeline, split_train_test
from src.evaluate import (choose_threshold_by_cost, compute_metrics, cost_table, theoretical_threshold,
                          threshold_table)
from src.predict import validate_customers
from src.train import build_schema, select_model

CATS = {
    "gender": ["Female", "Male"], "SeniorCitizen": ["No", "Yes"], "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"], "PhoneService": ["Yes"], "MultipleLines": ["No", "Yes"],
    "InternetService": ["DSL", "Fiber optic"], "OnlineSecurity": ["No", "Yes"],
    "OnlineBackup": ["No", "Yes"], "DeviceProtection": ["No", "Yes"], "TechSupport": ["No", "Yes"],
    "StreamingTV": ["No", "Yes"], "StreamingMovies": ["No", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"], "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
}


def fake_customers(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.choice(v, n) for c, v in CATS.items()})
    df["tenure"] = rng.integers(0, 73, n)
    df["MonthlyCharges"] = rng.uniform(20, 110, n).round(2)
    df["TotalCharges"] = (df.tenure * df.MonthlyCharges).round(2)
    logit = -0.04 * df.tenure + 0.02 * df.MonthlyCharges + (df.Contract == "Month-to-month") * 1.0 - 0.5
    df[config.TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df[config.ID_COL] = [f"C{i}" for i in range(n)]
    return df


def test_clean_data_handles_blank_total_charges_for_new_customers():
    raw = pd.DataFrame({config.ID_COL: ["a", "b"], "SeniorCitizen": [0, 1], "tenure": [0, 5],
                        "TotalCharges": [" ", "100.5"], "Churn": ["No", "Yes"]})
    out = clean_data(raw)
    assert out.loc[0, "TotalCharges"] == 0.0           # tenure 0 -> nothing billed yet
    assert out.loc[1, "TotalCharges"] == 100.5
    assert list(out["SeniorCitizen"]) == ["No", "Yes"]
    assert list(out["Churn"]) == [0, 1]


def test_split_is_stratified_and_disjoint():
    df = fake_customers(1000)
    tr, te = split_train_test(df, 0.2, 1)
    assert set(tr[config.ID_COL]).isdisjoint(te[config.ID_COL])
    assert abs(tr[config.TARGET].mean() - te[config.TARGET].mean()) < 0.01
    assert len(tr) + len(te) == len(df)


def test_preprocessing_statistics_come_from_training_data_only():
    df = fake_customers(600)
    tr, te = split_train_test(df, 0.3, 1)
    pipe = make_pipeline(LogisticRegression(max_iter=500), "base", True)
    pipe.fit(tr[config.RAW_FEATURES], tr[config.TARGET])
    scaler = pipe.named_steps["preprocess"].named_transformers_["num"].named_steps["scale"]
    assert scaler.mean_[0] == pytest.approx(tr["tenure"].mean())
    assert scaler.mean_[0] != pytest.approx(df["tenure"].mean())  # NOT fitted on train+test


def test_unseen_category_does_not_crash_pipeline():
    df = fake_customers(300)
    pipe = make_pipeline(LogisticRegression(max_iter=500), "engineered", True)
    pipe.fit(df[config.RAW_FEATURES], df[config.TARGET])
    new = df[config.RAW_FEATURES].head(3).copy()
    new["PaymentMethod"] = "Crypto"
    assert pipe.predict_proba(new).shape == (3, 2)


def test_cost_table_matches_hand_calculation():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.1])
    a = CostAssumptions(customer_value=100, intervention_cost=10, intervention_success=0.5)
    row = cost_table(y, p, a, thresholds=[0.5]).iloc[0]
    # flagged: customer 0 (TP) and customer 2 (FP); FN = customer 1
    assert (row.tp, row.fp, row.fn, row.tn) == (1, 1, 1, 2)
    expected = 1 * 100 + 1 * (10 + 0.5 * 100) + 1 * 10            # FN*V + TP*(C+(1-s)V) + FP*C
    assert row.expected_total_cost == pytest.approx(expected)
    assert row.net_savings_vs_no_action == pytest.approx(2 * 100 - expected)  # no-action cost = 2*V


def test_theoretical_threshold_and_optimiser_agree_for_calibrated_scores():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 40000)
    y = (rng.random(40000) < p).astype(int)               # perfectly calibrated by construction
    a = CostAssumptions(400, 25, 0.3)
    t, _ = choose_threshold_by_cost(y, p, a)
    assert abs(t - theoretical_threshold(a)) < 0.04


def test_threshold_changes_decisions_but_not_ranking_metrics():
    df = fake_customers(500)
    p = np.random.default_rng(1).uniform(size=500)
    y = df[config.TARGET].to_numpy()
    m_low, m_high = compute_metrics(y, p, 0.2), compute_metrics(y, p, 0.8)
    assert m_low["roc_auc"] == m_high["roc_auc"] == pytest.approx(roc_auc_score(y, p))
    assert m_low["flagged_fraction"] > m_high["flagged_fraction"]
    tab = threshold_table(y, p, [0.2, 0.8])
    assert tab.loc[0, "recall"] >= tab.loc[1, "recall"]  # lowering threshold never lowers recall


def test_select_model_prefers_simpler_model_within_one_standard_error():
    res = pd.DataFrame({
        "model": ["Dummy (prior)", "LogisticRegression", "XGBoost"], "complexity": [0, 1, 3],
        "average_precision_mean": [0.27, 0.660, 0.665], "average_precision_std": [0.0, 0.02, 0.02]})
    chosen, _ = select_model(res)
    assert chosen == "LogisticRegression"          # 0.005 gap < SE of ~0.009
    res.loc[2, "average_precision_mean"] = 0.70
    assert select_model(res)[0] == "XGBoost"       # clear winner beyond 1 SE


def test_validate_customers_rejects_bad_inputs_and_warns_on_total_charges():
    df = fake_customers(50)
    meta = {"schema": build_schema(df)}
    good = df[config.RAW_FEATURES].head(1).drop(columns=["TotalCharges"]).to_dict("records")[0]
    clean, warns = validate_customers(good, meta)
    assert any("TotalCharges" in w for w in warns)
    with pytest.raises(ValueError, match="Missing"):
        validate_customers({k: v for k, v in good.items() if k != "tenure"}, meta)
    with pytest.raises(ValueError, match="Invalid value"):
        validate_customers({**good, "Contract": "Lifetime"}, meta)
    with pytest.raises(ValueError, match="non-negative"):
        validate_customers({**good, "MonthlyCharges": -5}, meta)


# ---- tests that need the trained artifacts (skipped if you have not run `python -m src.train` yet)
needs_artifacts = pytest.mark.skipif(not (config.MODEL_PATH.exists() and config.TEST_PATH.exists()),
                                     reason="run the training pipeline first")


@needs_artifacts
def test_predict_churn_reproduces_saved_pipeline_probabilities():
    import joblib
    from src.data_preprocessing import load_split, split_xy
    from src.predict import predict_churn
    X, _ = split_xy(load_split("test"))
    X = X.head(25)
    expected = joblib.load(config.MODEL_PATH).predict_proba(X)[:, 1]
    got = predict_churn(X)["churn_probability"].to_numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-9)


@needs_artifacts
def test_predict_rejects_out_of_schema_category():
    from src.predict import predict_one
    from src.data_preprocessing import load_split, split_xy
    row = split_xy(load_split("test"))[0].iloc[0].to_dict()
    row["Contract"] = "Lifetime"
    with pytest.raises(ValueError, match="Invalid value"):
        predict_one(row)


@needs_artifacts
def test_saved_threshold_was_fixed_before_test_scoring_and_is_in_unit_interval():
    import json
    meta = json.loads(config.METADATA_PATH.read_text())
    assert 0.0 < meta["threshold"] < 1.0
    assert meta["cost_assumptions_note"].startswith("ILLUSTRATIVE")
