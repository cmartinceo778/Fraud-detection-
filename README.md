# Signal & Shield | Credit Card Fraud Analytics

An end-to-end portfolio case study built around the supplied `creditcard.csv`. It explores fraud exposure, trains a time-aware ranking baseline, and lets an analyst compare the fraud captured at different review-queue sizes.

## Business question

With a very rare fraud label, how many transactions should a payments team review to capture a useful share of fraudulent value without overwhelming investigators?

## Dataset

The supplied file is stored locally at `data/raw/creditcard.csv`. It contains **284,807 transactions**, **492 fraud labels** (0.173%), and 31 columns: elapsed `Time`, anonymized `V1`–`V28` features, `Amount`, and binary `Class`. The holdout is the final 20% by elapsed time. The model uses `Amount` and `V1`–`V28`; `Time` is used only to order the split, and `Class` is used only as the target.

The file's schema matches the [ULB Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Its published description says the transactions span two days and `V1`–`V28` are PCA components; the original feature meanings are unavailable. The dataset listing identifies ODbL 1.0. The provided CSV is retained in this local workspace and excluded from Git because it is about 151 MB. For a public portfolio repository, document the dataset attribution and provide retrieval instructions rather than committing the raw CSV. Confirm the terms attached to your copy before redistributing it.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run fraud_detection.py
```

The dashboard expects `data/raw/creditcard.csv`. If you clone the source without the ignored data file, obtain the dataset from the linked Kaggle page and place the CSV in that folder. To print the command-line evaluation summary:

```bash
python -m src.analyze --data data/raw/creditcard.csv
```

## What the dashboard shows

- Transaction and fraud-label activity over elapsed time.
- Transaction amount distribution for legitimate and fraudulent classes.
- A logistic-regression ranking model compared with an amount-only ranking baseline.
- A precision–recall curve across review budgets, with an adjustable selected review share.
- Captured fraud count and amount, recall, precision, false alerts, and missed fraud.
- A downloadable retrospective review queue for the chronological holdout.

## Method

1. Validate the expected columns and binary target; sort by `Time`.
2. Train on the first 80% of transaction time and evaluate on the final 20%.
3. Standardize numeric features and fit class-weighted logistic regression.
4. Rank holdout transactions and measure performance at a chosen queue share.
5. Compare against ranking solely by `Amount` to contextualize the model.

PR-AUC and queue-level precision/recall are emphasized over accuracy because a classifier that predicts every transaction as legitimate would still appear highly accurate on this dataset. The class-weighted model's score is a ranking signal, not a calibrated fraud probability. Amount captured is a retrospective sum in the source data's units; it is not a realized savings estimate.

## Limitations

This dataset covers only two days, has anonymized PCA features, and lacks feature descriptions, operational outcomes, customer-review costs, and modern transaction patterns. A chronological holdout from this same period is useful for demonstrating evaluation design but does not establish performance on future production data. A live program would require out-of-time validation across later periods, label-delay handling, calibration, alert-cost measurement, drift monitoring, and a review process that does not expose retrospective labels.

## Project structure

```text
fraud-analytics/
├── fraud_detection.py
├── data/raw/creditcard.csv   # supplied locally; git-ignored
├── src/analyze.py
├── sql/analysis.sql          # reusable warehouse query patterns
├── requirements.txt
└── README.md
```

## Stack

Python, pandas, NumPy, scikit-learn, Streamlit, Plotly, and SQL.
