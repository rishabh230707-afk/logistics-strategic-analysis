"""Focused checks for row accounting, KPI treatment and leakage."""
import unittest
import numpy as np
import pandas as pd
from preprocess import DATES, clean, make_demo, preprocess


class PreprocessingChecks(unittest.TestCase):
    def test_row_accounting_and_no_imputed_outcomes(self):
        o, c = make_demo()
        df, q, _, audit = clean(o, c)
        self.assertEqual(len(o), len(df)+len(q)+audit['exact_order_duplicates_removed'])
        self.assertTrue(df.order_id.is_unique)
        self.assertEqual(audit['duration_eligible'], 74)
        self.assertEqual(audit['otd_eligible'], 71)
        self.assertFalse(df.set_index('order_id').loc['SYN002', 'duration_eligible'])
        self.assertTrue(pd.isna(df.set_index('order_id').loc['SYN001', 'on_time']))

    def test_holdout_does_not_fit_preprocessing(self):
        o, c = make_demo()
        *_, a = preprocess(o, c)
        future = pd.to_datetime(o[DATES[0]], errors='coerce') >= pd.Timestamp('2018-05-01')
        for i in o.index[future]:
            o.loc[i, DATES[2]] = str(pd.Timestamp(o.loc[i, DATES[0]])+pd.Timedelta(days=300))
        *_, b = preprocess(o, c)
        for key in ['training_median_promise_days', 'training_scaler_mean', 'training_scaler_scale']:
            self.assertEqual(a['preprocessing'][key], b['preprocessing'][key])

    def test_tail_delays_and_unseen_categories_survive(self):
        df, _, _, train, test, audit = preprocess(*make_demo())
        self.assertEqual(int(df.duration_outlier.sum()), 2)
        self.assertEqual(audit['preprocessing']['unseen_holdout_states'], ['AM'])
        self.assertTrue(np.isfinite(test.drop(columns='order_id').to_numpy()).all())
        self.assertIn(60, train.target_delivery_days.tolist())
        self.assertIn(50, test.target_delivery_days.tolist())

    def test_customer_conflict_does_not_multiply_orders(self):
        o, c = make_demo()
        conflict = c.iloc[[20]].copy()
        conflict['customer_state'] = 'AM'
        c = pd.concat([c, conflict], ignore_index=True)
        df, _, rejected, audit = clean(o, c)
        self.assertEqual(len(df), 78)
        self.assertEqual(len(rejected), 2)
        self.assertEqual(audit['unmatched_customers'], 2)


if __name__ == '__main__':
    unittest.main()
