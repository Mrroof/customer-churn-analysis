from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/customer_churn_demo.csv")
out = ROOT / "outputs"
out.mkdir(exist_ok=True)

df["churn_flag"] = (df["churn"].astype(str).str.lower() == "yes").astype(int)
print("Rows:", len(df), "Demo churn rate:", round(df.churn_flag.mean(), 3))
print(df.groupby("contract_type").churn_flag.agg(["count", "mean"]))

ax = df.groupby("contract_type").churn_flag.mean().plot(
    kind="bar", figsize=(7, 4), title="Demo churn rate by contract type"
)
ax.set_ylabel("Churn rate")
ax.set_xlabel("Contract type")
plt.tight_layout()
plt.savefig(out / "churn_by_contract.png", dpi=160)
plt.close()

features = [
    "tenure_months", "monthly_charges", "support_tickets_90d",
    "contract_type", "internet_service"
]
prep = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), ["tenure_months", "monthly_charges", "support_tickets_90d"]),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), ["contract_type", "internet_service"])
])
pipe = Pipeline([
    ("prep", prep),
    ("model", LogisticRegression(max_iter=1000))
])

Xtr, Xte, ytr, yte = train_test_split(
    df[features], df.churn_flag, test_size=0.25,
    random_state=7, stratify=df.churn_flag
)
pipe.fit(Xtr, ytr)
pred = pipe.predict(Xte)
prob = pipe.predict_proba(Xte)[:, 1]
report = classification_report(yte, pred, zero_division=0)
auc = roc_auc_score(yte, prob)

(out / "model_metrics.txt").write_text(
    "SYNTHETIC DATA ONLY\n" + report + f"\nROC AUC: {auc:.3f}\n",
    encoding="utf-8"
)
print(report, "ROC AUC:", round(auc, 3))
