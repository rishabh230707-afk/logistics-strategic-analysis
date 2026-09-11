"""Week 1 illustration. Run on manually downloaded Olist CSV files."""
from pathlib import Path
import argparse
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error


def prepare(data):
    orders = pd.read_csv(data / 'olist_orders_dataset.csv')
    customers = pd.read_csv(data / 'olist_customers_dataset.csv')
    assert orders.order_id.is_unique, 'Duplicate order IDs'
    assert customers.customer_id.is_unique, 'Duplicate customer IDs'
    cols = ['order_purchase_timestamp', 'order_delivered_customer_date',
            'order_estimated_delivery_date']
    for col in cols:
        orders[col] = pd.to_datetime(orders[col], errors='coerce')
    df = orders.merge(customers[['customer_id', 'customer_state']],
                      on='customer_id', how='left', validate='many_to_one')
    complete = df[cols].notna().all(axis=1)
    eligible = df.order_status.eq('delivered') & complete
    valid = df.order_delivered_customer_date.ge(df.order_purchase_timestamp)
    valid &= df.order_estimated_delivery_date.ge(df.order_purchase_timestamp)
    audit = {'input_orders': len(df), 'eligible_orders': int(eligible.sum()),
             'invalid_eligible_orders': int((eligible & ~valid).sum())}
    df = df.loc[eligible & valid].copy()
    df['delivery_days'] = (df.order_delivered_customer_date -
                           df.order_purchase_timestamp).dt.total_seconds()/86400
    # Compare calendar dates: an estimated date may have midnight time.
    df['on_time'] = (df.order_delivered_customer_date.dt.normalize() <=
                     df.order_estimated_delivery_date.dt.normalize())
    df['purchase_weekday'] = df.order_purchase_timestamp.dt.dayofweek
    df['promise_days'] = (df.order_estimated_delivery_date -
                          df.order_purchase_timestamp).dt.total_seconds()/86400
    return df, audit


def run(data, output):
    df, audit = prepare(data)
    if df.empty:
        raise ValueError('No valid completed orders')
    output.mkdir(parents=True, exist_ok=True)
    kpis = {'on_time_delivery_pct': float(100*df.on_time.mean()),
            'mean_delivery_days': float(df.delivery_days.mean()),
            'p90_delivery_days': float(df.delivery_days.quantile(.9))}
    regional = df.groupby('customer_state', dropna=False).agg(
        orders=('order_id', 'size'), on_time_rate=('on_time', 'mean'),
        mean_days=('delivery_days', 'mean'))
    regional.to_csv(output / 'regional_kpis.csv')
    # Fixed prospective holdout; train only on labels known before cutoff.
    cutoff = pd.Timestamp('2018-05-01')
    test_end = pd.Timestamp('2018-07-01')
    train = df[(df.order_purchase_timestamp < cutoff) &
               (df.order_delivered_customer_date < cutoff)]
    test = df[(df.order_purchase_timestamp >= cutoff) &
              (df.order_purchase_timestamp < test_end)]
    if len(train) < 20 or len(test) < 5:
        raise ValueError('Insufficient data in selected temporal windows')
    features = ['customer_state', 'purchase_weekday', 'promise_days']
    prep = ColumnTransformer([
        ('region', make_pipeline(SimpleImputer(strategy='most_frequent'),
          OneHotEncoder(handle_unknown='ignore')), ['customer_state']),
        ('numeric', SimpleImputer(strategy='median'), features[1:])])
    model = make_pipeline(prep, RandomForestRegressor(
        n_estimators=100, min_samples_leaf=10, random_state=42, n_jobs=-1))
    model.fit(train[features], train.delivery_days)
    pred = model.predict(test[features])
    base = DummyRegressor(strategy='median').fit(
        train[['promise_days']], train.delivery_days)
    baseline = base.predict(test[['promise_days']])
    scores = {'train_orders': len(train), 'test_orders': len(test),
              'model_mae_days': float(mean_absolute_error(test.delivery_days, pred)),
              'median_baseline_mae_days': float(mean_absolute_error(
                  test.delivery_days, baseline)),
              'promise_baseline_mae_days': float(mean_absolute_error(
                  test.delivery_days, test.promise_days))}
    (output / 'summary.json').write_text(json.dumps(
        {'audit': audit, 'kpis': kpis, 'evaluation': scores}, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    args = parser.parse_args()
    run(args.data, args.output)
