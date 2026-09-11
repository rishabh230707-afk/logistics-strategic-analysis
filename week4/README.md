# Week 4 — Predictive Modeling and Optimization in Logistics Systems

Author: Rishabh Bansal

[Download the Word report](week4.docx).

This is an executed **synthetic demonstration**, not a real company dataset or
verified savings claim. It extends the workload-planning theme from Week 3 with
a separate 420-day demand simulation. It does not reuse Olist observations.

## Run

Use Python 3.11 or newer. From the repository root on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r week4/requirements.txt
python week4/model.py --output week4/results
```

Dependencies are pinned to the executed environment. Fixed seed: 42.
The script regenerates data, CSV summaries, JSON results and four PNG charts;
it does not rebuild the Word report.

## Prediction and validation

Forecast day t arrivals after day t-1 is finalized. Predictors include prior
counts, shifted rolling averages, calendar effects, known promotions and trend.
There are 392 usable rows after a 28-day history period:

- Development: 308 days, 2025-01-29 through 2025-12-02.
- Calibration: 28 days, 2025-12-03 through 2025-12-30.
- Final holdout: 56 days, 2025-12-31 through 2026-02-24.

Four expanding training windows with 35-day validation windows compare a
seasonal-naive baseline, three Ridge penalties and four random-forest settings.
Preprocessing is fitted within each training fold. Lowest mean validation MAE
selects the model. The selected model is fitted on development data and frozen
before calibration and holdout. Historical observed counts update each day;
this is rolling next-day evaluation, not a single 56-day-ahead forecast.

Ridge alpha 0.1 was selected. Holdout MAE: 11.476 shipments/day; RMSE: 14.504;
R-squared: 0.854. Seasonal-naive MAE: 28.143. Only the selected model and the
predeclared baseline are evaluated on the final holdout.

A nonnegative empirical 90th-percentile calibration residual supplies a 16.278
shipment buffer. It covers 87.5% of holdout actual counts before worker rounding.
This is a heuristic planning buffer, not guaranteed interval coverage.

## Staffing decision

Choose 2–8 integer workers, each with capacity 45 shipments/day and assumed
cost INR 900/day. Minimize labour while covering the planning target up to the
maximum capacity. If the target exceeds capacity, use eight workers and plan
external processing. Enumerating seven choices solves this constraint exactly.
This is not joint optimization of expected labour and outsourcing cost.

Realized overflow is outsourced at assumed INR 40/shipment. Same-day external
processing is assumed unlimited. Shift constraints, skill differences and
staffing-change costs are omitted. Realized overflow is not measured lateness.

| Policy | Worker-days | Outsourced shipments | Total cost INR |
| --- | ---: | ---: | ---: |
| Fixed six workers | 336 | 260 | 312,800 |
| Forecast only | 326 | 68 | 296,120 |
| Buffered forecast | 346 | 0 | 311,400 |

All policies face the same 13,511 holdout shipments. Forecast-only cost is 5.33%
below fixed staffing in this simulation. Buffered staffing costs more than
forecast-only staffing and has no observed overflow. A real pilot must determine
which trade-off is appropriate. Sensitivity tables re-solve staffing at 40 and
50 shipments per worker, retaining all other assumptions.

## Outputs and checks

`results/` includes the synthetic daily history, holdout predictions and policy
allocations, cross-validation scores and windows, policy comparison, capacity
sensitivity and summary metadata. The script also regenerates four charts in
`results/charts/`; chart images are embedded in the Word report.

Assertions check chronology, unique dates, missingness, nonnegative demand,
exclusion of the target from its own features, staffing limits and cost
reconciliation. They are executed in an ordinary Python run (without `-O`).

## References

- [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [Evaluation metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
