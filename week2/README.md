# Week 2 — Logistics data collection and preprocessing

Author: Rishabh Bansal

[Download the Word report](week2.docx)

This submission implements a reproducible preprocessing demonstration using
**invented records shaped like two Olist tables**. It is not an analysis of the
downloaded Olist dataset. Data quality counts in `demo_audit.json` describe only
the synthetic demonstration. No model performance or business savings are claimed.

## Run the demonstration

From the repository root, on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r week2/requirements.txt
python week2/preprocess.py --demo --output outputs/week2
python -m unittest discover -s week2 -p 'test_*.py' -v
```

Tested with pandas 2.2.3, NumPy 2.3.5 and scikit-learn 1.8.0.

## What the pipeline does

1. Validates required fields and standardizes whitespace and category casing.
2. Removes identical records and quarantines ambiguous or missing identity.
3. Parses timestamps, validates Brazilian state codes and checks customer joins.
4. Separates valid delivered orders from canceled or invalid records.
5. Preserves original dates and KPI eligibility; never imputes outcomes.
6. Fits median imputation, standardization and state encoding on historical training data only.
7. Flags long delivery durations with training-derived IQR thresholds, retaining valid tail delays.
8. Writes row counts, reasons, transformation parameters and separate feature matrices.

## Demonstration results

- 82 input order rows; one exact duplicate removed and three identity rows quarantined.
- 78 retained order records; 74 valid duration outcomes and 71 on-time KPI-eligible orders.
- 44 training and 30 holdout rows; eight predictor columns.
- Two training promises and one holdout promise imputed for features only.
- Two unusual delivery durations retained; no non-finite feature values.
- Four focused unit tests passed, including a check that changing holdout values cannot change fitted training parameters.

The 120-day holdout promise transforms to approximately 83.95 standard deviations.
This deliberately shows that scaling does not repair outliers or guarantee a
bounded range. Investigate such inputs before operational model use.

## Output files

- `audited_orders.csv`: all unambiguous orders, with quality and eligibility flags;
  invalid/unresolved records remain here for audit and must be filtered for analysis.
- `quarantined_orders.csv`, `quarantined_customers.csv`: records with ambiguous identity.
- `train_features.csv`, `holdout_features.csv`: eight predictors plus an order identifier
  and the observed target. Exclude the identifier and target from model inputs.
- `audit.json`: counts, training parameters, feature names and package versions.

## Use real CSVs later

Reference: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
Download manually under its access and licence terms. Record the retrieval date
and preserve unchanged source files. Put `olist_orders_dataset.csv` and
`olist_customers_dataset.csv` in a local `data/` folder, then run:

```bash
python week2/preprocess.py --data data --output outputs/week2_real
```

The real-file mode records SHA-256 hashes. Timestamp parsing expects
`YYYY-MM-DD HH:MM:SS` with consistent local time; review formats and timezone
semantics before combining sources. Required fields are documented in the report.
The defaults use a 1 May 2018 cutoff and purchases before 1 July 2018 for holdout.
Review these windows and outcome follow-up before using other data. Training
outcomes must already be known at cutoff. The completed-order subset cannot
describe unresolved or canceled orders without separate analysis.

Raw source data and generated row-level outputs are not committed. The only
committed results are the synthetic audit summary. The source CSVs have not been
downloaded or processed for this submission.
