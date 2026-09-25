"""
Train the loan default model that the FastAPI app will serve.

This reuses the same made-up loan applicant data and the same LightGBM
setup as Day 8 and Day 11, so this repo's story stays consistent: same
model family, but now wrapped as something a stranger can call over
HTTP instead of only living inside a notebook.

Run:
    python scripts/train_model.py

Writes:
    model/loan_default_model.joblib   (trained LightGBM pipeline + metadata)
"""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "loan_default_model.joblib"


def make_data(n: int = 8000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Build a pretend town of loan applicants with known, planted rules."""
    rng = np.random.default_rng(random_state)

    age = np.clip(rng.normal(38, 11, n), 21, 70)
    annual_income = np.clip(rng.lognormal(mean=np.log(55000), sigma=0.45, size=n), 15000, 300000)
    employment_years = np.clip(rng.uniform(0, 1, n) * (age - 20), 0, 40)
    credit_score = np.clip(rng.normal(660, 70, n) + 0.5 * employment_years, 300, 850)
    loan_to_income = rng.uniform(0.05, 0.8, n)
    loan_amount = annual_income * loan_to_income
    debt_to_income = np.clip(rng.normal(0.30, 0.10, n), 0.02, 0.8)
    late_rate = 0.5 + np.clip((700 - credit_score) / 100, 0, None) * 1.2
    num_late_payments = np.clip(rng.poisson(late_rate), 0, 12)
    has_cosigner = rng.binomial(1, 0.2, n)

    logit = (
        -2.6
        + 3.0 * loan_to_income
        + 4.0 * (debt_to_income - 0.30)
        - 0.012 * (credit_score - 660)
        + 0.35 * num_late_payments
        - 0.04 * employment_years
        - 0.8 * has_cosigner
    )
    prob_default = 1 / (1 + np.exp(-logit))
    defaulted = rng.binomial(1, prob_default)

    return pd.DataFrame(
        {
            "age": age.round(0),
            "annual_income": annual_income.round(0),
            "employment_years": employment_years.round(1),
            "credit_score": credit_score.round(0),
            "loan_amount": loan_amount.round(0),
            "debt_to_income": debt_to_income.round(3),
            "num_late_payments": num_late_payments,
            "has_cosigner": has_cosigner,
            "defaulted": defaulted,
        }
    )


def main() -> None:
    df = make_data()
    features = [c for c in df.columns if c != "defaulted"]
    X, y = df[features], df["defaulted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "f1_at_0.5": round(float(f1_score(y_test, (proba >= 0.5).astype(int))), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    print("Test metrics:", json.dumps(metrics, indent=2))

    bundle = {
        "model": model,
        "feature_order": features,
        "metrics": metrics,
        "random_state": RANDOM_STATE,
        "model_version": "1.0.0",
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved model bundle to {MODEL_PATH}")


if __name__ == "__main__":
    main()
