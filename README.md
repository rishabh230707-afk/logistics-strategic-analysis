# Logistics Strategic Analysis

Author: Rishabh Bansal

Week 1 internship project: strategic planning for e-commerce delivery reliability.
The Word report explains the scenario, KPIs, research, methods and roadmap.
This repository contains an illustrative Python starting point, not completed
business analysis. No improvement or model accuracy is claimed.

## Data
Download the Brazilian E-Commerce Public Dataset by Olist from:
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
Review and comply with its dataset licence and attribution requirements.
Place `olist_orders_dataset.csv` and `olist_customers_dataset.csv` in `data/`.
Data is not redistributed. Historical Brazilian data does not establish
performance in another country or in current operations.

## Run on macOS or Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python analysis.py --data data --output outputs
```

Outputs: `summary.json` with audit, KPIs and model scores, and
`regional_kpis.csv`. The model estimates purchase-to-delivery duration from
information assumed available at purchase. The script uses completed orders;
report this selection bias and monitor unresolved orders separately.
Training labels must be available before 1 May 2018. May and June 2018 form
the illustrative holdout. Date cutoffs must be reviewed before other datasets.
Full execution on the Olist dataset remains pending. Synthetic smoke testing
checks the pipeline only and does not validate business performance.

## Submission
Repository: https://github.com/rishabh230707-afk/logistics-strategic-analysis

[Download the Week 1 Word report](Week_1_Logistics_Strategic_Planning_Rishabh_Bansal.docx).
Submit the Word report and paste the repository URL into the compulsory
technical-project field of the internship form.

## Files
- Word report: scenario, KPIs, research, roadmap, code excerpts and references.
- `analysis.py`: data checks, delivery KPIs and regression illustration.
- `requirements.txt`: Python dependencies.

## Project roadmap
Collect and audit data → establish KPIs → explore delay patterns → build and
validate a temporal model → design a controlled operational pilot → monitor.
Clustering and vehicle routing are proposed extensions described in the report;
they are not implemented in this initial script.

## Week 2 — Data preprocessing

[Week 2 report and runnable demonstration](week2/README.md) cover missing values,
duplicate identity, timestamp validation, outlier screening and normalization.
The pipeline was executed on explicitly synthetic Olist-shaped records; four
focused tests passed. [Download week2.docx](week2/week2.docx).
