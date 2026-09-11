"""Synthetic next-day demand forecasting and constrained staffing demonstration."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NUM = ['trend','promotion','annual_sin','annual_cos','lag1','lag7','lag14','mean7','mean28']
FEATURES = NUM + ['weekday']

def simulate(seed=42):
    rng=np.random.default_rng(seed)
    dates=pd.date_range('2025-01-01',periods=420)
    t=np.arange(len(dates))
    weekday=dates.dayofweek.to_numpy()
    promotion=((t%35)>=28).astype(int) # published calendar, known ahead of time
    seasonal=18*np.sin(2*np.pi*t/365.25)
    demand=165+np.array([40,30,20,15,35,-25,-40])[weekday]+.12*t+45*promotion+seasonal+rng.normal(0,14,len(t))
    df=pd.DataFrame({'date':dates,'shipments':np.maximum(40,np.rint(demand)).astype(int),'trend':t,'weekday':weekday,'promotion':promotion,'annual_sin':np.sin(2*np.pi*t/365.25),'annual_cos':np.cos(2*np.pi*t/365.25)})
    return df

def features(raw):
    df=raw.sort_values('date').copy()
    for lag in [1,7,14]: df[f'lag{lag}']=df.shipments.shift(lag)
    for window in [7,28]: df[f'mean{window}']=df.shipments.shift(1).rolling(window).mean()
    return df.dropna().reset_index(drop=True)

def pipeline(model):
    prep=ColumnTransformer([('num',StandardScaler(),NUM),('day',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['weekday'])])
    return Pipeline([('prep',prep),('model',model)])

def metrics(y,p):
    return {'MAE':float(mean_absolute_error(y,p)),'RMSE':float(np.sqrt(mean_squared_error(y,p))),'R2':float(r2_score(y,p)),'bias_pred_minus_actual':float(np.mean(p-y))}

def solve_staff(target,capacity=45):
    """Exact enumeration of minimum labour cost subject to target coverage.
    Workers are integers 2..8; overflow above maximum capacity is outsourced.
    """
    target=max(0,float(target))
    choices=range(2,9)
    feasible=[w for w in choices if capacity*w>=target]
    return min(feasible,key=lambda w:900*w) if feasible else 8

def run(output):
    output.mkdir(parents=True,exist_ok=True)
    raw=simulate(); df=features(raw)
    assert len(raw)==420 and len(df)==392 and raw.date.is_unique
    assert not df.isna().any().any() and (df.shipments>=0).all()
    # A future-target perturbation must not change that day's predictors.
    changed=raw.copy(); changed.loc[100,'shipments']+=1000
    pd.testing.assert_series_equal(features(changed).loc[72,FEATURES],df.loc[72,FEATURES])
    dev,cal,test=df.iloc[:-84],df.iloc[-84:-56],df.iloc[-56:]
    assert dev.date.max()<cal.date.min()<test.date.min()
    candidates={'Seasonal naive':None}
    for a in [.1,1,10]: candidates[f'Ridge alpha {a}']=pipeline(Ridge(alpha=a))
    for depth in [6,None]:
        for leaf in [3,8]: candidates[f'Forest depth {depth} leaf {leaf}']=pipeline(RandomForestRegressor(n_estimators=160,max_depth=depth,min_samples_leaf=leaf,random_state=42,n_jobs=1))
    cv=TimeSeriesSplit(n_splits=4,test_size=35)
    scores=[]; folds=[]
    for k,(tr,va) in enumerate(cv.split(dev),1):
        folds.append({'fold':k,'train_start':str(dev.iloc[tr].date.min().date()),'train_end':str(dev.iloc[tr].date.max().date()),'validation_start':str(dev.iloc[va].date.min().date()),'validation_end':str(dev.iloc[va].date.max().date()),'train_days':len(tr),'validation_days':len(va)})
        for name,est in candidates.items():
            if est is None: pred=dev.iloc[va].lag7.to_numpy()
            else:
                fit=clone(est).fit(dev.iloc[tr][FEATURES],dev.iloc[tr].shipments)
                pred=np.maximum(0,fit.predict(dev.iloc[va][FEATURES]))
            scores.append({'model':name,'fold':k,**metrics(dev.iloc[va].shipments.to_numpy(),pred)})
    cvrows=pd.DataFrame(scores); ranking=cvrows.groupby('model').agg(cv_MAE=('MAE','mean'),cv_MAE_sd=('MAE','std'),cv_RMSE=('RMSE','mean')).sort_values('cv_MAE')
    winner=ranking.index[0]
    best=candidates[winner]
    fitted=None if best is None else clone(best).fit(dev[FEATURES],dev.shipments)
    def predict(frame): return frame.lag7.to_numpy() if fitted is None else np.maximum(0,fitted.predict(frame[FEATURES]))
    # Freeze model before calibration and holdout. Only observed lags update daily.
    cp=predict(cal); buffer=max(0,float(np.quantile(cal.shipments.to_numpy()-cp,.9,method='higher')))
    tp=predict(test); y=test.shipments.to_numpy()
    testmetrics=[]
    # Evaluate selected model and predeclared benchmark; no holdout-driven tuning.
    for name,pred in [(winner,tp),('Seasonal naive',test.lag7.to_numpy())]: testmetrics.append({'model':name,**metrics(y,pred)})
    fixed=solve_staff(float(dev.shipments.quantile(.9)))
    policies={'Fixed staffing':np.repeat(fixed,len(test)),'Forecast only':np.array([solve_staff(v) for v in tp]),'Buffered forecast':np.array([solve_staff(v+buffer) for v in tp])}
    policy=[]; daily=test[['date','shipments']].copy();daily['prediction']=tp;daily['buffered_target']=tp+buffer
    for name,workers in policies.items():
        external=np.maximum(y-45*workers,0); labour=900*workers; cost=labour+40*external
        assert ((workers>=2)&(workers<=8)).all() and np.allclose(cost,labour+40*external)
        policy.append({'policy':name,'worker_days':int(workers.sum()),'mean_workers':float(workers.mean()),'labour_inr':int(labour.sum()),'outsourced_shipments':int(external.sum()),'outsourced_pct':float(100*external.sum()/y.sum()),'days_without_overflow':int((external==0).sum()),'total_cost_inr':int(cost.sum())})
        daily[name+'_workers']=workers; daily[name+'_outsourced']=external
    sensitivity=[]
    for cap in [40,45,50]:
        workers=np.array([solve_staff(v+buffer,cap) for v in tp]); external=np.maximum(y-cap*workers,0)
        sensitivity.append({'capacity_per_worker':cap,'worker_days':int(workers.sum()),'outsourced_shipments':int(external.sum()),'cost_inr':int((900*workers+40*external).sum())})
    regional=None
    summary={'scope':'SYNTHETIC ONLY; no measured business savings','seed':42,'raw_days':420,'usable_days':392,'winner':winner,'buffer_shipments':buffer,'calibration_empirical_coverage':float(np.mean(cal.shipments.to_numpy()<=cp+buffer)),'test_buffer_coverage':float(np.mean(y<=tp+buffer)),'test_shipments':int(y.sum()),'fixed_workers':fixed,'test_metrics':testmetrics,'policies':policy,'sensitivity':sensitivity,'splits':{name:{'days':len(f),'start':str(f.date.min().date()),'end':str(f.date.max().date())} for name,f in [('development',dev),('calibration',cal),('test',test)]},'checks':'Chronology, unique dates, missingness, nonnegative demand, no target leakage in features, staffing bounds and cost reconciliation passed','versions':{'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'matplotlib':matplotlib.__version__}}
    raw.to_csv(output/'synthetic_daily_demand.csv',index=False);daily.to_csv(output/'holdout_predictions.csv',index=False)
    ranking.to_csv(output/'cv_ranking.csv');cvrows.to_csv(output/'cv_folds_metrics.csv',index=False);pd.DataFrame(folds).to_csv(output/'cv_windows.csv',index=False)
    pd.DataFrame(policy).to_csv(output/'staffing_comparison.csv',index=False);pd.DataFrame(sensitivity).to_csv(output/'capacity_sensitivity.csv',index=False)
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    charts=output/'charts';charts.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.text(.01,.005,'Synthetic demonstration | Seed 42 | Costs and capacity are assumptions',fontsize=8,color='#555555')
        fig.tight_layout(rect=[0,.025,1,1]);fig.savefig(charts/name,dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3.5));ax.plot(raw.date,raw.shipments,color='#176B87',lw=1)
    ax.axvspan(cal.date.min(),cal.date.max(),alpha=.15,color='orange',label='Calibration')
    ax.axvspan(test.date.min(),test.date.max(),alpha=.15,color='green',label='Final holdout')
    ax.set(title='Daily demand and chronological evaluation periods',ylabel='Shipments per day',xlabel='Target date');ax.legend();save(fig,'01_demand.png')
    fig,ax=plt.subplots(figsize=(9,3.5));ax.plot(test.date,y,'o-',ms=3,label='Actual simulated demand',color='#176B87');ax.plot(test.date,tp,label='Selected model',color='#D17A22');ax.plot(test.date,tp+buffer,'--',label='Buffered planning target',color='#34866F');ax.set(title='Rolling next-day forecasts on the untouched holdout',ylabel='Shipments per day',xlabel='Target date');ax.legend(fontsize=8);save(fig,'02_forecast.png')
    fig,axs=plt.subplots(1,2,figsize=(9,3.4));axs[0].scatter(tp,y-tp,color='#176B87');axs[0].axhline(0,color='black',lw=1);axs[0].set(xlabel='Predicted shipments',ylabel='Actual minus predicted',title='Holdout residuals')
    axs[1].hist(y-tp,bins=12,color='#176B87',edgecolor='white');axs[1].axvline(buffer,color='#D17A22',linestyle='--',label='Planning buffer');axs[1].set(xlabel='Actual minus predicted',ylabel='Days',title='Forecast error distribution');axs[1].legend(fontsize=8);save(fig,'03_residuals.png')
    fig,axs=plt.subplots(1,2,figsize=(9,3.5));labels=['Fixed','Forecast','Buffered'];pol=pd.DataFrame(policy)
    axs[0].bar(labels,pol.total_cost_inr/1000,color=['#777777','#176B87','#34866F']);axs[0].set(ylabel='Total cost (thousand INR)',title='Labour plus outsourced processing')
    axs[1].bar(labels,pol.outsourced_pct,color=['#777777','#176B87','#34866F']);axs[1].set(ylabel='Share of shipments outsourced (%)',title='Dependence on external processing');save(fig,'04_staffing.png')
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path('week4/results'));args=parser.parse_args();print(json.dumps(run(args.output),indent=2))
