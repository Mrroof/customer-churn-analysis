# Customer Churn and Retention Analysis

> **Portfolio case study · Python · pandas · scikit-learn · Matplotlib**

## Executive summary
This project examines customer churn patterns in a synthetic subscription dataset and builds a baseline classification model to identify records associated with churn.

## Business questions
- What proportion of the demo customer base churned?
- How does churn differ across contract types?
- How well does a logistic regression model distinguish churn from retention?

## Demonstration findings
- Dataset size: **1,000 synthetic customer records**.
- Overall demo churn rate: **48%**.
- Month-to-month contracts had a **58.5%** churn rate, compared with **39.0%** for one-year and **26.6%** for two-year contracts.
- Held-out model accuracy: **72%**; ROC AUC: **0.778**.
- Churn-class precision: **72%**; recall: **70%**.

<image src="docs/churn-by-contract.svg" alt="Synthetic churn rate by contract type" />

These are patterns in the generated dataset, not evidence that contract length causes churn.

## Method
1. Inspect data quality and churn distribution.
2. Explore churn by contract type and selected customer attributes.
3. Train a logistic regression classifier.
4. Evaluate using a held-out sample, class-level precision/recall and ROC AUC.

## Run locally
```bash
pip install -r requirements.txt
python src/analyze_churn.py
```

Outputs are saved in `outputs/`.

## Business interpretation
The synthetic contract pattern suggests a useful retention question: should month-to-month customers receive earlier engagement or clearer renewal options? This is a hypothesis to test with genuine customer data, not a recommendation to deploy directly.

## Limitations
The data is synthetic and does not contain real customer records. Model metrics are illustrative. A production churn model requires representative data, leakage checks, probability calibration, time-based validation, fairness review, cost-sensitive evaluation and business validation.
