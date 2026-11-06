"""Analysis pipeline for the supplied European card transactions dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "Class"
NON_FEATURES = {TARGET, "Time"}


def load_data(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"Time", "Amount", TARGET, *[f"V{i}" for i in range(1, 29)]}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Required dataset columns contain missing values; inspect and resolve them before analysis.")
    if not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Class must contain binary labels 0 (legitimate) and 1 (fraud).")
    return frame.sort_values("Time", kind="stable").reset_index(drop=True)


def fit_evaluate(frame: pd.DataFrame) -> dict:
    features = [c for c in frame.columns if c not in NON_FEATURES]
    cut = int(len(frame) * .80)
    train, test = frame.iloc[:cut].copy(), frame.iloc[cut:].copy()
    if train[TARGET].nunique() < 2 or test[TARGET].nunique() < 2:
        raise ValueError("Both chronological partitions must contain fraud and legitimate transactions.")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, C=.7, solver="liblinear", random_state=42),
    )
    model.fit(train[features], train[TARGET])
    test["risk_score"] = model.predict_proba(test[features])[:, 1]
    # A simple ranking comparator that uses only transaction amount, not labels.
    test["amount_only_score"] = test["Amount"].rank(pct=True, method="average")
    metrics = {
        "pr_auc": average_precision_score(test[TARGET], test["risk_score"]),
        "roc_auc": roc_auc_score(test[TARGET], test["risk_score"]),
        "fraud_rate": float(test[TARGET].mean()),
        "fraud_count": int(frame[TARGET].sum()),
        "total_rows": len(frame),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_time_end": float(train.Time.max()),
        "test_time_start": float(test.Time.min()),
    }
    return {"train": train, "test": test, "model": model, "features": features, "metrics": metrics}


def review_metrics(frame: pd.DataFrame, score_col: str, review_share: float) -> dict:
    if not 0 < review_share <= 1:
        raise ValueError("review_share must be greater than 0 and no more than 1.")
    count = max(1, int(np.ceil(len(frame) * review_share)))
    queue = frame.nlargest(count, score_col).copy()
    fraud = queue[TARGET].astype(bool)
    true_positives = int(fraud.sum())
    total_fraud = int(frame[TARGET].sum())
    false_positives = int(count - true_positives)
    captured = float(queue.loc[fraud, "Amount"].sum())
    total_fraud_value = float(frame.loc[frame[TARGET].eq(1), "Amount"].sum())
    return {
        "queue": queue,
        "alerts": count,
        "precision": true_positives / count,
        "recall": true_positives / max(1, total_fraud),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "missed_fraud": total_fraud - true_positives,
        "captured_amount": captured,
        "total_fraud_amount": total_fraud_value,
        "captured_amount_share": captured / max(1e-9, total_fraud_value),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/creditcard.csv")
    args = parser.parse_args()
    result = fit_evaluate(load_data(args.data))
    m = result["metrics"]
    print(f"Rows: {m['total_rows']:,} | Fraud labels: {m['fraud_count']:,}")
    print(f"Chronological holdout: {m['test_rows']:,} rows | prevalence {m['fraud_rate']:.3%}")
    print(f"Logistic regression — PR-AUC {m['pr_auc']:.4f} | ROC-AUC {m['roc_auc']:.4f}")
    for name in ("risk_score", "amount_only_score"):
        op = review_metrics(result["test"], name, .01)
        print(f"Top 1% by {name}: precision {op['precision']:.2%} | recall {op['recall']:.2%} | captured amount {op['captured_amount']:,.2f}")


if __name__ == "__main__":
    main()
