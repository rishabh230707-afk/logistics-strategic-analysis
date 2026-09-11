# Week 3 — Logistics EDA and visualization

Author: Rishabh Bansal

[Download the Word report](week3.docx).

This is an explicitly **hypothetical** logistics dataset, separate from Week 2.
All observations, costs, deadlines and relationships are simulated. No real
company performance, causal effects or savings are claimed.

## Reproduce from the repository root

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r week3/requirements.txt
python week3/analyze.py --output outputs/week3
```

Seed: 42. Tested with NumPy 2.3.5, pandas 2.2.3 and Matplotlib 3.10.8.
The script generates 2,329 completed shipments across 84 days and checks unique
IDs, completeness, numeric validity, duration reconciliation and daily counts.

## Outputs

The script writes synthetic_shipments.csv, descriptive statistics, daily,
weekly and regional summaries, Pearson and Spearman correlation tables,
summary.json, and six PNG charts. Summary CSV/JSON tables are published in
`results/`; regenerate the row-level dataset and standalone images with the
command above. All six figures are embedded in the report.

## Findings in this simulation

- On-time delivery: 85.27%; 343 late shipments.
- Mean delivery duration: 40.47 hours; median: 40.19; P90: 58.10.
- Mean allocated transport cost: INR 509.33 per shipment.
- Weekly average volume: 168.00 in the first eight weeks, 246.25 in the last four.
- On-time rate drops from 95.01% to 71.98% across these phases.
- Daily volume and daily mean hub time have Pearson correlation 0.914.
- Distance and cost have Pearson correlation 0.982 by construction.

The generator deliberately makes hub time increase with daily volume, uses
longer regional distances, applies different regional promises, and sets costs
using distance and weight. These findings illustrate those assumptions.
See the report for why the fastest region can have the lowest on-time rate,
why a weak weight-cost pairwise correlation does not imply no cost effect,
and why severe delays must remain visible.

## Units and interpretation

One row is one completed shipment. `dispatch_date` is network-entry date;
`delivery_hours = hub_hours + transit_hours`. `cost_inr` is allocated transport
cost, not freight charged to a customer. Shipment route kilometres are not fleet
vehicle kilometres. No missing or canceled shipments are represented.
Weekly rates are calculated from shipments, not averages of daily percentages.
Daily workload analysis uses one observation per day. No significance tests,
predictive model or operational optimization are claimed.
