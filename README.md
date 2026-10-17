# Customer Churn Prediction

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-machine%20learning-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-modeling-189FDD)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-modeling-2E8B57)](https://lightgbm.readthedocs.io/)

This project uses machine learning to predict which telecom customers are likely to leave the service. I compared different models, adjusted the prediction threshold, and used SHAP to understand which factors were most related to churn.

## Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost, LightGBM
- **Model Explainability:** SHAP

## Features

- Analysed data from 7,043 customer accounts.
- Compared Logistic Regression, Random Forest, XGBoost, and LightGBM.
- Used cross-validation to evaluate model performance.
- Adjusted the classification threshold to improve churn detection.
- Used SHAP to identify influential features such as contract type, tenure, and internet service.

## How It Works

The model learns patterns from customer data and predicts which customers are more likely to leave the service. I compared multiple classification models and adjusted the decision threshold to balance recall and precision. SHAP was used to better understand which features influenced the predictions.

```text
Customer data
     |
Data cleaning and preparation
     |
Train and evaluate classification models
     |
Choose a classification threshold
     |
Predict customers likely to churn
     |
Use SHAP to understand predictions
```

## Results

| Metric | Score |
|---|---:|
| Recall | 83.2% |
| Precision | 48.7% |
| ROC-AUC | 0.85 |
| PR-AUC | 0.67 |

Recall measures how many customers who churned were correctly identified. Precision measures how many customers predicted to churn actually churned. The best threshold depends on how the business balances missed churners against false alarms.

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the project's training or evaluation scripts using the commands provided in the repository.

## Data

The project uses data from 7,043 telecom customer accounts. Check the repository for the dataset and the scripts used to prepare the data.

## Current Limitations

- The results depend on the dataset and the way the classification threshold is selected.
- Precision is lower than recall, so some customers predicted to churn may not actually leave.
- Model performance should be checked again before using it on new customer data.

## Possible Improvements

- Test the models on additional datasets.
- Tune the classification threshold based on business costs.
- Monitor performance when new customer data becomes available.
- Explore ways to improve precision while keeping recall useful.

## What I Learned

Through this project, I practised comparing classification models, evaluating them with different metrics, adjusting a prediction threshold, and using SHAP to understand model predictions.

## Author

**Aadarsh Dixit**  
[GitHub Profile](https://github.com/aadarshdixit100-ops)
