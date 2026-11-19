# Customer Churn and Retention Analysis

## Objective
Explore customer attributes associated with churn and demonstrate a basic classification workflow.

## Data notice
The included customer records are **synthetic demonstration data**, not real customer information. Do not describe results as findings from a real telecom or bank.

## Questions
- What is the demo churn rate?
- How does churn differ by contract type?
- How are tenure, charges and support contacts associated with churn?
- How does logistic regression perform on a held-out sample?

## Run
```bash
pip install -r requirements.txt
python src/analyze_churn.py
```
Outputs are saved in `outputs/`.

## Limitations
A production retention model requires representative data, leakage checks, calibration, fairness assessment, cost-sensitive evaluation and business validation.
