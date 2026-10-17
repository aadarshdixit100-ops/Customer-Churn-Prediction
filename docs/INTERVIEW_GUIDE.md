# Interview guide: 30 questions on this project

Answers are written so you can say them aloud. Numbers refer to *this* project's actual results
(`reports/model_evaluation.md`); re-check them if you re-run with different library versions.

---
## A. EDA and the data

**1. What is one row, what is the target, and how imbalanced is it?**
One row is one telecom customer (7,043 in total). The target `Churn` is 1 if the customer left in the observation window.
26.5% churn, so about 1 in 4: moderate imbalance, not extreme.

**2. You found a numeric column stored as text. What did you do?**
`TotalCharges` had 11 blank strings. I inspected them before coercing: all had `tenure == 0` (new customers with nothing billed), so
0 is the domain-correct value. That is a row-wise rule, not a statistic learned from other customers, so it cannot leak. Dropping them or
median-imputing would have been wrong (median would invent a ~$1,400 bill).

**3. Why did you not drop the 22 rows that are identical apart from `customerID`?**
Different IDs mean different customers. With ~16 mostly-categorical columns, coincidences are expected. Deleting them would remove real customers.

**4. With 5,600 customers, nearly everything is "significant". How did you handle that?**
I reported effect sizes next to p-values (Cramér's V for categorical features, Cliff's delta for numeric), used confidence intervals,
and corrected for multiple comparisons (Benjamini-Hochberg). E.g. contract has V = 0.41 (large) while MultipleLines is significant but V = 0.04 (negligible).

**5. What is Simpson's paradox and did you see it?**
An association that changes or reverses once you condition on another variable. Churners pay more per month overall, but *within* each internet-service type,
higher bills go with *lower* churn: the overall pattern comes from the fiber group being both expensive and high-churn. So "charges drive churn" would be a wrong conclusion.

---
## B. Preprocessing and leakage

**6. What is data leakage? Give an example from this project.**
Information from outside the training data (validation/test rows or the future) influencing training. Examples I guarded against: fitting a scaler or
encoder on all data; the snapshot issue where churners' `tenure` stops at the churn date. Notebook 03 has a demonstration: selecting 20 features from pure noise
*outside* cross-validation yields ROC-AUC 0.81; inside the pipeline it is 0.50.

**7. How does your pipeline prevent leakage?**
Split first; then one scikit-learn `Pipeline` (feature engineering -> `ColumnTransformer` -> model) is passed to `cross_validate`, which refits every step on the
training folds only. The test set is touched once, after model and threshold are frozen.

**8. Why scale for Logistic Regression but not for tree models?**
L2-regularised linear models are scale-sensitive (the penalty depends on coefficient size, and solvers converge better on comparable scales). Trees split on thresholds, so monotone rescaling changes nothing.

**9. One-hot vs label encoding?**
Label encoding invents an order (DSL < Fiber < None) which linear models take literally. One-hot has no implied order. I also use `handle_unknown="ignore"` so new categories do not crash prediction.

**10. Why is a random split a limitation here?**
It assumes customers are exchangeable. With timestamps I would split by time (train on the past, test on later months) to detect drift. This dataset has none, so my test estimate is likely somewhat optimistic about future performance.

---
## C. Models

**11. Explain Logistic Regression in two sentences.**
It models the log-odds of churn as a weighted sum of features and converts that to a probability with the sigmoid; weights are fitted by maximum likelihood with a penalty.
Each coefficient is the change in log-odds per unit change in the feature, holding others fixed.

**12. Bagging vs boosting?**
Bagging (Random Forest) averages many deep trees trained on bootstrap samples to reduce variance. Boosting (XGBoost/LightGBM) adds shallow trees sequentially, each fitting the errors of the ensemble so far, to reduce bias.

**13. Your best model is LightGBM, but Logistic Regression is close. Defend the choice.**
By my pre-stated rule (highest nested-CV PR-AUC; prefer simpler among models within one standard error) LightGBM won: 0.673 vs 0.660. It beat Logistic Regression in all 5 outer folds, but only by about +0.012 PR-AUC, which is smaller than the fold-to-fold standard deviation (~0.02). I would say plainly that it is a small edge and that Logistic Regression is a defensible, more transparent alternative; SHAP rankings from both models broadly agree (Spearman 0.87).

**14. The default LightGBM scored 0.653 and the tuned one 0.673. What happened?**
The default overfit (training PR-AUC about 0.85 vs validation 0.65). Tuning chose a heavily regularised model (4 leaves, min 80 samples per leaf, column subsampling, L2).

**15. Why did you use nested cross-validation?**
If you tune with CV and report the best CV score, it is optimistically biased (maximum of many noisy estimates). Nested CV tunes on inner folds and scores on outer folds the tuning procedure never saw.

---
## D. Cross-validation and imbalance

**16. How does stratified k-fold work, and why stratify?**
Split into k parts, each preserving the churn ratio; train on k-1, validate on the remaining part, rotate, average. Stratification keeps every fold representative, which matters for imbalanced targets.

**17. Did you use SMOTE? Why/why not?**
No. 26.5% is moderate imbalance; SMOTE must be applied only inside training folds; synthetic one-hot customers are unrealistic; and resampling distorts predicted probabilities, which I need for cost-based decisions. I changed the decision threshold instead. `class_weight="balanced"` kept the ranking (PR-AUC 0.657 vs 0.658) but pushed the mean predicted probability to 0.41 vs a true rate of 0.265 and worsened Brier (0.165 vs 0.135).

---
## E. Metrics

**18. Why is accuracy misleading?**
Always predicting "stays" gives 73.5% accuracy with zero recall. In my results a lower threshold reduces accuracy (0.80 -> 0.72) while improving the business outcome.

**19. What do false positives and false negatives mean for churn?**
FP: we spend retention budget on someone who would have stayed. FN: a churner we never contacted, lost revenue. Which is worse depends on the cost of an intervention vs the value of a retained customer.

**20. ROC-AUC vs PR-AUC: when is each preferable?**
ROC-AUC = probability a random churner is scored above a random stayer; threshold-free, but the false-positive rate hides false alarms when negatives are plentiful. PR-AUC focuses on the positive class and has a no-skill baseline equal to prevalence (0.265), so it is more informative for imbalanced problems where we care about the churner class. I selected on PR-AUC and reported both.

**21. Why does lowering the threshold raise recall and lower precision?**
More customers are flagged, so more true churners are caught (recall up), but the extra flagged customers have lower scores and are more often stayers (precision down).

---
## F. Thresholds and cost

**22. How did you choose the threshold?**
By minimising expected cost on out-of-fold development predictions under illustrative assumptions (value $400, action cost $25, 30% success). Expected cost = FN·V + TP·(C + (1−s)V) + FP·C. The result, 0.22, matches the theoretical break-even C/(sV) = 0.21 that holds when probabilities are calibrated.

**23. Are your dollar figures real?**
No. They are labelled illustrative; the dataset contains no information on action costs or success rates. I show a sensitivity analysis (optimal threshold ranges from 0.05 to 0.93 across plausible values) and I stress that real numbers need an A/B test.

**24. Does threshold tuning improve your model?**
No. ROC-AUC and PR-AUC are identical at every threshold because the ranking does not change. The threshold only moves the operating point and the business outcome.

**25. What if the retention team can only call 30% of customers?**
Add a capacity constraint. The best feasible threshold is then ~0.38 (flagging 29.9%, precision 0.59, recall 0.67) regardless of the break-even.

---
## G. Calibration, SHAP, limitations

**26. What is calibration and why does it matter for you?**
Among customers scored ~0.3, about 30% should churn. It matters because I multiply probabilities by dollar values; miscalibrated scores give wrong expected costs. My model is close to the diagonal on out-of-fold and test data, and Brier is 0.135 vs 0.195 for the baseline.

**27. How are SHAP values computed, and what is the difference between global and local explanations?**
SHAP values are Shapley values: a feature's average marginal contribution over all feature orderings, so that base value + sum of contributions = the prediction (I verified this to 1e-15). TreeExplainer computes them exactly for trees. Global = average |SHAP| / beeswarm (what matters overall). Local = waterfall for one customer.

**28. Does SHAP show what would reduce churn?**
No. It describes the model's use of features, a predictive association. Contract type may be a symptom of loyalty rather than a cause; `tenure` is highly important but cannot be changed. Causal claims need experiments.

**29. Your ROC-AUC is 0.85, not 0.95. Is that a problem?**
It is a realistic ceiling for this data: churn depends on things not recorded (competitor offers, service problems). Roughly half of the customers flagged at my threshold do not churn (precision 0.49). I would not trust an 0.95 on this dataset without checking for leakage.

**30. What are the main limitations and what would you do next?**
Sample dataset from a fictional company; one snapshot, random split, no temporal validation; labels are observational; illustrative costs; fairness not assessed beyond a descriptive subgroup check. Next: time-based validation, survival modelling (time-to-churn), uplift modelling or an A/B test to measure who is *persuadable*, calibration monitoring, and a fairness review of attributes like age and gender.
