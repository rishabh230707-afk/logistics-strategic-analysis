"""Week 2 preprocessing; --demo uses invented Olist-shaped records only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATES = ['order_purchase_timestamp', 'order_delivered_customer_date',
         'order_estimated_delivery_date']
ORDER_COLS = ['order_id', 'customer_id', 'order_status'] + DATES
STATES = set('AC AL AP AM BA CE DF ES GO MA MT MS MG PA PB PR PE PI RJ RN RS RO RR SC SP SE TO'.split())


def make_demo():
    """Deterministic invented records with deliberately injected defects."""
    orders, customers = [], []
    for i in range(80):
        day = (pd.Timestamp('2018-02-01') + pd.Timedelta(days=i)
               if i < 50 else pd.Timestamp('2018-05-01') + pd.Timedelta(days=i-50))
        day += pd.Timedelta(hours=8+i % 10)
        actual = day + pd.Timedelta(days=4+i % 8)
        promise = day.normalize() + pd.Timedelta(days=10+i % 5)
        orders.append([f'SYN{i:03}', f'C{i:03}', 'delivered',
                       str(day), str(actual), str(promise)])
        customers.append([f'C{i:03}', ['SP', 'RJ', 'MG'][i % 3]])
    o = pd.DataFrame(orders, columns=ORDER_COLS)
    c = pd.DataFrame(customers, columns=['customer_id', 'customer_state'])
    o.loc[1, DATES[2]] = None                    # Missing predictor
    o.loc[2, DATES[1]] = None                    # Missing outcome
    o.loc[3, DATES[0]] = 'invalid-date'          # Malformed date
    o.loc[4, DATES[1]] = '2017-01-01 00:00:00'  # Impossible duration
    o.loc[5, 'order_status'] = 'canceled'
    o.loc[5, DATES[1]] = None
    o.loc[6, DATES[2]] = '2017-01-01 00:00:00'  # Bad predictor only
    o.loc[7, DATES[1]] = str(pd.Timestamp(o.loc[7, DATES[0]]) + pd.Timedelta(days=60))
    o.loc[8, 'customer_id'] = 'UNMATCHED'
    o.loc[9, 'order_id'] = None
    o.loc[10, 'order_status'] = ' DELIVERED '
    o.loc[60, DATES[2]] = None                   # Holdout missing value
    o.loc[61, DATES[2]] = str(pd.Timestamp(o.loc[61, DATES[0]]) + pd.Timedelta(days=120))
    o.loc[62, DATES[1]] = str(pd.Timestamp(o.loc[62, DATES[0]]) + pd.Timedelta(days=50))
    c.loc[11, 'customer_state'] = ' sp '
    c.loc[12, 'customer_state'] = None
    c.loc[13, 'customer_state'] = 'ZZ'
    c.loc[65, 'customer_state'] = 'AM'           # Category unseen in training
    conflict = o.iloc[[14]].copy()
    conflict['order_status'] = 'shipped'
    o = pd.concat([o, o.iloc[[0]], conflict], ignore_index=True)
    c = pd.concat([c, c.iloc[[0]]], ignore_index=True)
    return o, c


def clean(orders, customers):
    for frame, columns in [(orders, ORDER_COLS),
                           (customers, ['customer_id', 'customer_state'])]:
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f'Missing required columns: {sorted(missing)}')
    o, c = orders[ORDER_COLS].copy(), customers[['customer_id', 'customer_state']].copy()
    audit = {'input_orders': len(o), 'input_customers': len(c)}
    for frame in (o, c):
        for col in frame:
            frame[col] = frame[col].astype('string').str.strip().replace('', pd.NA)
    o['order_status'] = o.order_status.str.lower()
    c['customer_state'] = c.customer_state.str.upper()
    audit['missing_before_cleaning'] = o.isna().sum().astype(int).to_dict()
    audit['exact_order_duplicates_removed'] = int(o.duplicated().sum())
    audit['exact_customer_duplicates_removed'] = int(c.duplicated().sum())
    o, c = o.drop_duplicates().copy(), c.drop_duplicates().copy()
    # Missing or conflicting identity cannot be repaired by guessing.
    bad_key = o.order_id.isna() | o.order_id.duplicated(keep=False)
    quarantined = o.loc[bad_key].copy()
    quarantined['reason'] = np.where(quarantined.order_id.isna(),
                                      'missing_order_id', 'conflicting_order_id')
    audit['order_identity_rows_quarantined'] = int(bad_key.sum())
    o = o.loc[~bad_key].copy()
    bad_customer = c.customer_id.isna() | c.customer_id.duplicated(keep=False)
    audit['customer_identity_rows_quarantined'] = int(bad_customer.sum())
    rejected_customers = c.loc[bad_customer].copy()
    c = c.loc[~bad_customer].copy()
    invalid_state = c.customer_state.notna() & ~c.customer_state.isin(STATES)
    audit['invalid_states_set_missing'] = int(invalid_state.sum())
    c.loc[invalid_state, 'customer_state'] = pd.NA
    audit['malformed_dates'] = {}
    for col in DATES:
        original = o[col]
        o[col] = pd.to_datetime(original, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        audit['malformed_dates'][col] = int((original.notna() & o[col].isna()).sum())
    df = o.merge(c, on='customer_id', how='left', validate='many_to_one', indicator=True)
    audit['unmatched_customers'] = int(df['_merge'].eq('left_only').sum())
    df = df.drop(columns='_merge')
    if len(df) != len(o) or not df.order_id.is_unique:
        raise ValueError('Join changed the order grain')
    df['delivery_days'] = (df[DATES[1]]-df[DATES[0]]).dt.total_seconds()/86400
    df['promise_days'] = (df[DATES[2]]-df[DATES[0]]).dt.total_seconds()/86400
    negative_promise = df.promise_days.lt(0)
    audit['negative_promises_set_missing'] = int(negative_promise.sum())
    df.loc[negative_promise, 'promise_days'] = np.nan
    df['duration_eligible'] = (df.order_status.eq('delivered').fillna(False)
                               & df.delivery_days.ge(0))
    df['exclusion_reason'] = ''
    df.loc[~df.duration_eligible, 'exclusion_reason'] = 'not_delivered'
    delivered = df.order_status.eq('delivered').fillna(False)
    df.loc[delivered & df.delivery_days.isna(), 'exclusion_reason'] = 'missing_or_invalid_date'
    df.loc[delivered & df.delivery_days.lt(0), 'exclusion_reason'] = 'negative_delivery_duration'
    # KPI eligibility never relies on an imputed promise.
    df['otd_eligible'] = df.duration_eligible & df.promise_days.notna()
    df['on_time'] = pd.Series(pd.NA, index=df.index, dtype='boolean')
    eligible = df.otd_eligible
    df.loc[eligible, 'on_time'] = (df.loc[eligible, DATES[1]].dt.normalize()
                                  <= df.loc[eligible, DATES[2]].dt.normalize())
    df['promise_missing'] = df.promise_days.isna().astype(int)
    # Cyclic encoding avoids treating hour 23 and hour 0 as far apart.
    hour = df[DATES[0]].dt.hour
    df['hour_sin'] = np.sin(2*np.pi*hour/24)
    df['hour_cos'] = np.cos(2*np.pi*hour/24)
    audit.update(retained_orders=len(df), duration_eligible=int(df.duration_eligible.sum()),
                 otd_eligible=int(df.otd_eligible.sum()),
                 excluded_duration_reasons=df.loc[~df.duration_eligible, 'exclusion_reason'].value_counts().to_dict())
    return df, quarantined, rejected_customers, audit


def preprocess(orders, customers, cutoff='2018-05-01', test_end='2018-07-01'):
    df, quarantine, rejected_customers, audit = clean(orders, customers)
    cutoff, test_end = pd.Timestamp(cutoff), pd.Timestamp(test_end)
    if test_end <= cutoff:
        raise ValueError('test_end must follow cutoff')
    usable = df.loc[df.duration_eligible].copy()
    train = usable.loc[(usable[DATES[0]] < cutoff) & (usable[DATES[1]] < cutoff)].copy()
    test = usable.loc[(usable[DATES[0]] >= cutoff) & (usable[DATES[0]] < test_end)].copy()
    if len(train) < 10 or test.empty or train.promise_days.notna().sum() == 0:
        raise ValueError('Insufficient training or holdout data; review date windows')
    q1, q3 = train.delivery_days.quantile([.25, .75])
    low, high = float(q1-1.5*(q3-q1)), float(q3+1.5*(q3-q1))
    df['duration_outlier'] = df.duration_eligible & ((df.delivery_days < low) | (df.delivery_days > high))
    # This retrospective outcome flag is never a model predictor.
    features = ['promise_days', 'customer_state', 'promise_missing', 'hour_sin', 'hour_cos']
    for part in (train, test):
        part['customer_state'] = part.customer_state.fillna('Unknown').astype(str)
    transformer = ColumnTransformer([
        ('numeric', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), ['promise_days']),
        ('state', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['customer_state']),
        ('other', 'passthrough', ['promise_missing', 'hour_sin', 'hour_cos'])])
    xtrain = transformer.fit_transform(train[features])
    xtest = transformer.transform(test[features])
    if not np.isfinite(xtrain).all() or not np.isfinite(xtest).all():
        raise ValueError('Non-finite features after preprocessing')
    numeric = transformer.named_transformers_['numeric']
    audit['preprocessing'] = {
        'train_orders': len(train), 'holdout_orders': len(test),
        'outside_training_or_holdout': len(usable)-len(train)-len(test),
        'feature_count': xtrain.shape[1],
        'train_missing_promises_imputed': int(train.promise_days.isna().sum()),
        'holdout_missing_promises_imputed': int(test.promise_days.isna().sum()),
        'training_median_promise_days': float(numeric.named_steps['simpleimputer'].statistics_[0]),
        'training_scaler_mean': float(numeric.named_steps['standardscaler'].mean_[0]),
        'training_scaler_scale': float(numeric.named_steps['standardscaler'].scale_[0]),
        'train_scaled_mean': float(xtrain[:, 0].mean()),
        'train_scaled_std': float(xtrain[:, 0].std()),
        'holdout_scaled_max': float(xtest[:, 0].max()),
        'duration_iqr_lower_days': low, 'duration_iqr_upper_days': high,
        'duration_outliers_flagged_and_retained': int(df.duration_outlier.sum()),
        'nonfinite_feature_values': int((~np.isfinite(xtrain)).sum()+(~np.isfinite(xtest)).sum()),
        'unseen_holdout_states': sorted(set(test.customer_state)-set(train.customer_state)),
        'feature_names': transformer.get_feature_names_out().tolist()}
    train_features = pd.DataFrame(xtrain, columns=transformer.get_feature_names_out())
    test_features = pd.DataFrame(xtest, columns=transformer.get_feature_names_out())
    for output, part in [(train_features, train), (test_features, test)]:
        output.insert(0, 'order_id', part.order_id.to_numpy())
        output['target_delivery_days'] = part.delivery_days.to_numpy()
    return df, quarantine, rejected_customers, train_features, test_features, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--demo', action='store_true')
    group.add_argument('--data', type=Path, help='Directory containing the two Olist CSVs')
    parser.add_argument('--output', type=Path, default=Path('outputs/week2'))
    parser.add_argument('--cutoff', default='2018-05-01')
    parser.add_argument('--test-end', default='2018-07-01')
    args = parser.parse_args()
    if args.demo:
        orders, customers = make_demo()
        source = {'mode': 'SYNTHETIC DEMONSTRATION', 'seed': 'deterministic formulas; no randomness'}
    else:
        files = [args.data/'olist_orders_dataset.csv', args.data/'olist_customers_dataset.csv']
        orders, customers = [pd.read_csv(p, dtype='string') for p in files]
        source = {'mode': 'USER SUPPLIED CSVs', 'reference': 'https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce',
                  'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    *frames, audit = preprocess(orders, customers, args.cutoff, args.test_end)
    audit['source'] = source
    audit['versions'] = {'pandas': pd.__version__, 'numpy': np.__version__, 'scikit-learn': sklearn.__version__}
    args.output.mkdir(parents=True, exist_ok=True)
    names = ['audited_orders', 'quarantined_orders', 'quarantined_customers', 'train_features', 'holdout_features']
    for name, frame in zip(names, frames):
        frame.to_csv(args.output/f'{name}.csv', index=False)
    (args.output/'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
