"""Streamlit dashboard. Run:  streamlit run app.py

Everything shown comes from the artifacts produced by the pipeline (saved model, metrics JSON,
figures, processed data). Nothing here is simulated: predictions call src.predict.predict_one,
which uses the saved, fitted pipeline.
"""
import json

import pandas as pd
import streamlit as st

from src import config
from src.evaluate import threshold_table, cost_table
from src.predict import load_artifacts, predict_one

st.set_page_config(page_title="Customer churn risk", layout="wide")


@st.cache_data
def load_json(path):
    return json.loads(path.read_text())


def require(paths):
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        st.error("Missing files: " + ", ".join(missing) +
                 ". Run `python -m src.data_preprocessing`, `python -m src.train`, "
                 "`python -m src.evaluate` and `python -m src.explain` first.")
        st.stop()


def show_figure(name, caption=None):
    path = config.FIGURES_DIR / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"{name} not found - run the corresponding notebook/script.")


require([config.MODEL_PATH, config.METADATA_PATH, config.TEST_METRICS_PATH, config.TRAIN_PATH,
         config.TEST_PATH, config.DATA_SUMMARY_PATH, config.OOF_PATH])
meta, metrics, summary = load_json(config.METADATA_PATH), load_json(config.TEST_METRICS_PATH), load_json(config.DATA_SUMMARY_PATH)

st.title("Customer churn risk")
st.caption("IBM Telco sample dataset (fictional company). A learning project: probabilities are model estimates, "
           "not guarantees, and the cost figures are illustrative assumptions.")

tab_data, tab_model, tab_thr, tab_shap, tab_pred = st.tabs(
    ["Data", "Model performance", "Threshold and costs", "Explanations (SHAP)", "Score a customer"])

# ------------------------------------------------------------------ data
with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{summary['n_rows']:,}")
    c2.metric("Columns", summary["n_columns"])
    c3.metric("Churn rate", f"{summary['churn_rate']:.1%}")
    c4.metric("Train / test", f"{summary['n_train']:,} / {summary['n_test']:,}")
    train = pd.read_csv(config.TRAIN_PATH)
    st.subheader("Churn rate by feature (development set)")
    feature = st.selectbox("Feature", ["Contract", "InternetService", "PaymentMethod", "TechSupport",
                                       "OnlineSecurity", "SeniorCitizen", "PaperlessBilling", "Partner", "Dependents"])
    st.bar_chart(train.groupby(feature)["Churn"].mean().rename("churn rate"))
    col_a, col_b = st.columns(2)
    with col_a:
        show_figure("eda_churn_by_tenure.png", "Churn by tenure")
    with col_b:
        show_figure("eda_confounding_checks.png", "Confounding checks")

# ------------------------------------------------------------------ model
with tab_model:
    f, f5, d = metrics["final_at_chosen_threshold"], metrics["final_at_0.5"], metrics["dummy_at_0.5"]
    ci = metrics["bootstrap_95ci"]
    st.write(f"Final model: **{metrics['model_name']}** - evaluated once on {metrics['n_test']:,} held-out customers.")
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC", f"{f['roc_auc']:.3f}", f"95% CI {ci['roc_auc'][0]:.3f}-{ci['roc_auc'][1]:.3f}", delta_color="off")
    c2.metric("PR-AUC", f"{f['average_precision']:.3f}", f"no-skill {f['prevalence']:.3f}", delta_color="off")
    c3.metric("Brier score", f"{f['brier']:.3f}", f"baseline {d['brier']:.3f}", delta_color="off")
    table = pd.DataFrame({f"Chosen threshold ({metrics['threshold']:.2f})": f, "Default 0.50": f5, "Dummy baseline": d}).T
    st.dataframe(table[["accuracy", "precision", "recall", "f1", "flagged_fraction", "tp", "fp", "tn", "fn"]]
                 .style.format("{:.3f}", subset=["accuracy", "precision", "recall", "f1", "flagged_fraction"]))
    col_a, col_b = st.columns(2)
    with col_a:
        show_figure("test_confusion_matrix_chosen.png", "Confusion matrix, chosen threshold")
    with col_b:
        show_figure("test_confusion_matrix_default.png", "Confusion matrix, 0.50 threshold")
    show_figure("test_pr_roc_curves.png")
    show_figure("calibration_curve.png")
    st.subheader("Cross-validation comparison (nested CV, development set)")
    st.dataframe(pd.read_csv(config.CV_NESTED_PATH)[["model", "average_precision_mean", "average_precision_std",
                                                      "roc_auc_mean", "neg_brier_mean"]].round(3))

# ------------------------------------------------------------------ thresholds
with tab_thr:
    st.write("Move the sliders to change the **illustrative** cost assumptions. The table is computed live from the "
             "out-of-fold development predictions; the test set is not used to pick thresholds.")
    c1, c2, c3 = st.columns(3)
    value = c1.slider("Value of retaining a would-be churner ($)", 100, 1500, int(meta["cost_assumptions"]["customer_value"]), 50)
    cost = c2.slider("Cost of one retention action ($)", 5, 200, int(meta["cost_assumptions"]["intervention_cost"]), 5)
    success = c3.slider("Action success probability", 0.05, 0.9, float(meta["cost_assumptions"]["intervention_success"]), 0.05)
    from src.evaluate import choose_threshold_by_cost, theoretical_threshold
    from src.config import CostAssumptions
    oof = pd.read_csv(config.OOF_PATH)
    a = CostAssumptions(value, cost, success)
    best_t, tab = choose_threshold_by_cost(oof["churn"], oof["oof_probability"], a)
    st.metric("Cost-optimal threshold under these assumptions", f"{best_t:.2f}",
              f"calibrated-theory C/(sV) = {theoretical_threshold(a):.2f}", delta_color="off")
    st.line_chart(tab.set_index("threshold")[["precision", "recall", "f1"]])
    st.line_chart(tab.set_index("threshold")[["net_savings_per_1000"]])
    st.dataframe(tab[tab.threshold.round(2).isin([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])]
                 [["threshold", "flagged_pct", "precision", "recall", "f1", "tp", "fp", "tn", "fn", "net_savings_per_1000"]]
                 .round(3))
    st.caption("The model was shipped with the threshold chosen under the original assumptions "
               f"({meta['threshold']:.2f}); changing the sliders only shows how that decision would move.")

# ------------------------------------------------------------------ SHAP
with tab_shap:
    st.write("SHAP describes what the model relies on (predictive associations), not what would *cause* a customer to stay.")
    col_a, col_b = st.columns(2)
    with col_a:
        show_figure("shap_importance_bar.png")
    with col_b:
        show_figure("shap_summary_beeswarm.png")
    example = st.selectbox("Example customer", ["highest_risk", "lowest_risk", "near_threshold",
                                                "missed_churner_FN", "false_alarm_FP"])
    show_figure(f"shap_waterfall_{example}.png")

# ------------------------------------------------------------------ predict
with tab_pred:
    _, m = load_artifacts()
    cats, nums = m["schema"]["categorical"], m["schema"]["numeric"]
    st.write("Enter a customer profile. The saved pipeline (preprocessing + model) is applied; nothing is re-fitted.")
    with st.form("customer"):
        c1, c2, c3 = st.columns(3)
        inputs = {}
        with c1:
            inputs["tenure"] = st.number_input("Tenure (months)", 0, 120, 12)
            inputs["MonthlyCharges"] = st.number_input("Monthly charges ($)", 0.0, 300.0, 70.0, 0.5)
            inputs["Contract"] = st.selectbox("Contract", cats["Contract"])
            inputs["PaymentMethod"] = st.selectbox("Payment method", cats["PaymentMethod"])
            inputs["PaperlessBilling"] = st.selectbox("Paperless billing", cats["PaperlessBilling"])
            inputs["gender"] = st.selectbox("Gender", cats["gender"])
        with c2:
            inputs["InternetService"] = st.selectbox("Internet service", cats["InternetService"])
            for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]:
                inputs[col] = st.selectbox(col, cats[col])
        with c3:
            for col in ["StreamingTV", "StreamingMovies", "PhoneService", "MultipleLines", "SeniorCitizen", "Partner", "Dependents"]:
                inputs[col] = st.selectbox(col, cats[col])
        submitted = st.form_submit_button("Score customer")
    if submitted:
        try:
            result = predict_one(inputs)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.metric("Estimated churn probability", f"{result['churn_probability']:.1%}")
            if result["predicted_churn"]:
                st.warning(f"Flagged as at-risk (probability >= threshold {result['threshold']:.2f}).")
            else:
                st.success(f"Not flagged (probability < threshold {result['threshold']:.2f}).")
            for w in result["warnings"]:
                st.info(w)
            st.caption("A probability is a model estimate based on patterns in past customers, not a guarantee. "
                       "Total charges were approximated as tenure x monthly charges.")
