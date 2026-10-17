
# Customer Churn Prediction

This project uses machine learning to predict which telecom customers are likely to leave the service. I compared different models, optimized the prediction threshold, and used SHAP to understand the main factors affecting customer churn.

### Key Features

* Analyzed data from 7,043 customer accounts.
* Compared Logistic Regression, Random Forest, XGBoost, and LightGBM.
* Used cross-validation for model evaluation.
* Optimized the classification threshold to improve churn detection.
* Used SHAP to identify important factors such as contract type, tenure, and internet service.

### Results

* **Recall:** 83.2%
* **Precision:** 48.7%
* **ROC-AUC:** 0.85
* **PR-AUC:** 0.67

### Tools Used

Python, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, SHAP
