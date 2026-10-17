# Resume summary, defensible claims and GitHub checklist

## Resume entry (pick 3-4 bullets; keep numbers exactly as measured)
**Customer Churn Prediction and Risk Analysis** | Python, scikit-learn, LightGBM, XGBoost, SHAP, Streamlit | *GitHub link*
- Built a leakage-safe churn modelling pipeline on the IBM Telco dataset (7,043 customers, 26.5% churn): stratified train/test split, scikit-learn `Pipeline` + `ColumnTransformer`, and nested cross-validation to compare Logistic Regression, Random Forest, XGBoost and LightGBM against a baseline.
- Selected LightGBM by a pre-defined PR-AUC / one-standard-error rule; on a held-out test set it reached ROC-AUC 0.85 (95% CI 0.83-0.87) and PR-AUC 0.67 vs 0.27 for a no-skill baseline, with probabilities calibrated (Brier 0.135 vs 0.195).
- Framed threshold selection as a cost-based decision: derived the expected-cost function, chose a threshold (0.22) from out-of-fold data, showed sensitivity to assumptions, and demonstrated that thresholding changes decisions, not ranking quality.
- Used SHAP (global + local) to interpret the model - contract type, tenure, internet service and add-ons were most influential, consistent with statistical EDA (chi-square/Cramer's V, Mann-Whitney/Cliff's delta, confounding checks) - and explicitly separated associations from causal claims.
- Delivered reusable validated prediction code, 12 unit tests, a Streamlit dashboard and documentation.

## Claims you CAN make (all measured in this repo)
* The dataset has 7,043 customers and 21 columns; churn rate 26.5%.
* Models were compared with stratified and nested cross-validation; the test set was scored once after model and threshold were fixed.
* Test ROC-AUC 0.849, PR-AUC 0.665, Brier 0.135; at the chosen threshold precision 0.49 and recall 0.83 (flagging 45% of customers); at 0.5, precision 0.66 and recall 0.52.
* Feature engineering (3 features) did not meaningfully improve CV performance, so it was not adopted.
* Logistic Regression and the tree ensembles were close; the selected LightGBM won by a small, consistent margin (about +0.012 PR-AUC in CV).
* SHAP additivity was verified; top drivers are contract, tenure, internet service, online security.
* Under *assumed, illustrative* costs the cost-optimal threshold was 0.22 and matched the theoretical break-even.

## Claims you must NOT make
* "Reduced churn" / "improved retention" / "saved $X" - nothing was deployed or A/B tested; dollar figures are assumptions.
* "95% accuracy" or any figure not in `reports/model_evaluation.md`.
* "SHAP shows what causes churn" - it shows what the model uses.
* "Production-ready" or "real telecom data" - it is an IBM sample dataset for a fictional company.
* "Model generalises to future customers" - there is no temporal validation.
* "Tuned thoroughly" - the search was deliberately small.

## Good interview framing
"I deliberately did not chase accuracy. The best I could honestly get was ROC-AUC 0.85, and I can explain why that is a realistic ceiling,
how I avoided leakage, why LightGBM and logistic regression are statistically close, and how the threshold depends on business costs I had to assume."

## GitHub publishing checklist
- [ ] Re-run `python -m src.data_preprocessing && python -m src.train && python -m src.evaluate && python -m src.explain && pytest -q` in a fresh virtual environment; confirm numbers still match the README (small last-digit differences are fine, update the README if they are not).
- [ ] Re-run notebooks 01-06 in order (`jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb`) so outputs are visible on GitHub; check no cell errors.
- [ ] Confirm `.gitignore` excludes `data/raw/*.csv` and `data/processed/*.csv`; commit `models/`, `reports/` (small) so the dashboard/README images work.
- [ ] Add a LICENSE (e.g. MIT) and keep the dataset attribution (IBM).
- [ ] Replace `<your-repo-url>` in the README; add your name/contact; confirm images render.
- [ ] Skim every notebook once and make sure you can explain each cell (use `docs/INTERVIEW_GUIDE.md`).
- [ ] Do not commit secrets or absolute paths (`git grep -n "/home"`).
- [ ] Optional: add a GitHub Actions workflow running `pytest`, and deploy the dashboard (it needs the generated artifacts committed or built in CI).
- [ ] Pin the repo description, topics (`churn-prediction`, `shap`, `lightgbm`, `scikit-learn`), and add the best figure (SHAP summary or cost curve) to the README top.
