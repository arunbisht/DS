# Customer Churn Prediction

This project completes the provided Data Science Assignment using the IBM Telco Customer Churn dataset.

## 1. What are we trying to do?

The business wants to identify telecom customers who may leave (churn) so the retention team can contact them proactively.

- Target: `Churn`
- `Yes` = customer churned
- `No` = customer did not churn
- Train/test split: 70% / 30%
- Random seed: 42
- Final model: Decision Tree Classifier

## 2. Project structure

```text
customer_churn_project/
├── data/
│   ├── TelcoCustomerChurn.csv
│   └── TelcoCustomerChurn_Data_Dictionary.csv
├── notebook/
│   └── churn_analysis.ipynb
├── model/
│   ├── churn_model.pkl
│   ├── feature_importance.csv
│   └── model_comparison.csv
├── src/
│   └── feature_engineering.py
├── app.py
├── requirements.txt
├── sample_request.json
└── README.md
```

## 3. Beginner setup

Open a terminal in this folder.

### Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Mac/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install packages

```bash
pip install -r requirements.txt
```

### Run the notebook

```bash
jupyter notebook
```

Open:
`notebook/churn_analysis.ipynb`

Run the cells from top to bottom.

## 4. Run the API

From the project root:

```bash
uvicorn app:app --reload
```

Open the interactive API page:

`http://127.0.0.1:8000/docs`

Choose `POST /predict`, click **Try it out**, paste the JSON from `sample_request.json`, and click **Execute**.

Expected response shape:

```json
{
  "prediction": "Yes",
  "churn_probability": 0.82
}
```

The exact probability depends on the trained model.

## 5. Important concepts in simple language

### Feature

A column used by the model to make a prediction.

Example: `tenure`, `Contract`, `MonthlyCharges`.

### Target

The thing we want to predict.

Here the target is `Churn`.

### Training data

The 70% of records used to learn patterns.

### Test data

The 30% kept separate so we can check how the model performs on unseen customers.

### Encoding

Machine-learning models need numbers. One-hot encoding converts categories such as `Month-to-month`, `One year`, and `Two year` into numeric columns.

### Imputation

When a value is missing, we replace it using a rule. Here numeric missing values use the median and categorical missing values use the most frequent category.

### Pipeline

A pipeline connects feature creation + preprocessing + model into one reusable object. This is important because the same transformations must be applied to new API customers.

### Recall

Of all customers who really churned, how many did we successfully identify?

For a retention team, recall is particularly important because missing a genuine churner can mean losing the customer.

## 6. Dataset observations

The uploaded dataset contains 7,043 customers and 21 columns. There are no duplicate rows.

`TotalCharges` is stored as text in the CSV. Converting it to numeric reveals 11 blank/non-numeric values; these are treated as missing and handled by the preprocessing pipeline.

The target distribution is approximately:

- No churn: 73.5%
- Churn: 26.5%

## 7. Feature engineering

Two required features were created:

1. `AvgMonthlySpend` = `TotalCharges / tenure`
   - For zero-tenure customers, the feature falls back to `MonthlyCharges`.
   - It represents the customer's average spend per month.

2. `TotalServices`
   - Counts how many service columns contain `Yes`.
   - It gives a simple measure of how many services the customer uses.

An additional interpretability feature, `TenureGroup`, groups tenure into:
`0-6`, `7-12`, `13-24`, `25-48`, `49-72` months.

## 8. Model experiments

Three Decision Tree configurations are included in the notebook:

- Model A: max_depth=5, min_samples_split=20
- Model B: max_depth=8, min_samples_leaf=10, class_weight='balanced'
- Model C: max_depth=6, min_samples_leaf=5, class_weight='balanced

Model C is used as the final model because it provides a stronger balance of recall and F1 while explicitly addressing the imbalance between churn and non-churn customers.

## 9. Final test results

Using the fixed 70:30 test split:

- Accuracy: 0.7425
- Precision: 0.5102
- Recall: 0.7594
- F1: 0.6103
- Confusion matrix: [[1143, 409], [135, 426]]

Interpretation:

- The model catches about 75.9% of actual churners.
- Precision means that about 51.0% of customers predicted as churners actually churned.
- Because the retention use case is proactive outreach, recall is an important business metric: a false negative means the company failed to flag a customer who actually churned.

## 10. Important leakage decision

`customerID` is removed because it is an identifier, not a meaningful behavioural feature.

The target `Churn` is never passed into the feature preprocessing/model.

The preprocessing and feature engineering are stored inside the saved pipeline, so the API uses exactly the same transformations as training.

## 11. GIT Path

https://github.com/arunbisht/DS
