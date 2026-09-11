"""Reproducible hypothetical logistics EDA. No real company observations."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REGIONS = ['North', 'West', 'East', 'South']
COLORS = ['#176B87', '#D17A22', '#7655A3', '#34866F']
NUMERIC = ['distance_km','weight_kg','hub_hours','transit_hours','delivery_hours','cost_inr']


def simulate(seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for day_index, date in enumerate(pd.date_range('2026-01-05', periods=84)):
        expected = 21 + (6 if date.dayofweek < 5 else -4) + (12 if day_index >= 56 else 0)
        volume = max(8, int(rng.poisson(expected)))
        for _ in range(volume):
            region = rng.choice(REGIONS, p=[.3,.25,.25,.2])
            distance = max(40, rng.normal(dict(North=180,West=420,East=620,South=850)[region],75))
            weight = float(np.clip(rng.lognormal(1.3,.55),.3,25))
            hub = max(2, 8 + max(volume-25,0)*1.1 + (7 if region=='East' else 0) + rng.normal(0,2.5))
            transit = max(5, 9 + distance/35 + rng.normal(0,3))
            if rng.random() < .03:
                transit += rng.uniform(12,30)
            delivery = hub+transit
            promise = dict(North=36,West=48,East=60,South=72)[region]
            cost = max(50,120+.72*distance+9*weight+(25 if volume>35 else 0)+rng.normal(0,25))
            rows.append([f'SYN{len(rows)+1:05}',date,region,distance,weight,hub,transit,delivery,promise,cost,volume])
    columns=['shipment_id','dispatch_date','region','distance_km','weight_kg','hub_hours','transit_hours','delivery_hours','promised_hours','cost_inr','daily_shipments']
    df=pd.DataFrame(rows,columns=columns)
    df['on_time']=df.delivery_hours.le(df.promised_hours)
    df['late_hours']=(df.delivery_hours-df.promised_hours).clip(lower=0)
    return df


def analyze(df, output):
    output.mkdir(parents=True,exist_ok=True)
    charts=output/'charts'; charts.mkdir(exist_ok=True)
    if df.isna().any().any() or not df.shipment_id.is_unique:
        raise ValueError('Missing data or duplicate shipment identity')
    if not np.isfinite(df[NUMERIC].to_numpy()).all() or (df[NUMERIC]<=0).any().any():
        raise ValueError('Invalid numeric values')
    if not np.allclose(df.delivery_hours,df.hub_hours+df.transit_hours):
        raise ValueError('Duration components do not reconcile')
    counts=df.groupby('dispatch_date').size()
    if not np.array_equal(df.daily_shipments,df.dispatch_date.map(counts)):
        raise ValueError('Daily volume does not reconcile')
    df.to_csv(output/'synthetic_shipments.csv',index=False)
    desc=df[NUMERIC].describe(percentiles=[.25,.5,.75,.9]).T
    desc.to_csv(output/'descriptive_statistics.csv')
    pearson=df[NUMERIC].corr(); spearman=df[NUMERIC].corr(method='spearman')
    pearson.to_csv(output/'pearson_correlations.csv'); spearman.to_csv(output/'spearman_correlations.csv')
    daily=df.groupby('dispatch_date').agg(shipments=('shipment_id','size'),on_time=('on_time','mean'),hub_hours=('hub_hours','mean'),cost=('cost_inr','sum'))
    weekly=df.set_index('dispatch_date').resample('W-MON',closed='left',label='left').agg(shipments=('shipment_id','size'),on_time=('on_time','mean'),cost=('cost_inr','sum'))
    region=df.groupby('region').agg(shipments=('shipment_id','size'),mean_hours=('delivery_hours','mean'),median_hours=('delivery_hours','median'),on_time=('on_time','mean'),cost=('cost_inr','mean'),distance=('distance_km','mean'),hub_hours=('hub_hours','mean')).reindex(REGIONS)
    daily.to_csv(output/'daily_summary.csv'); weekly.to_csv(output/'weekly_summary.csv'); region.to_csv(output/'regional_summary.csv')
    q1,q3=df.delivery_hours.quantile([.25,.75]); upper=q3+1.5*(q3-q1)
    phase=np.where(df.dispatch_date<pd.Timestamp('2026-03-02'),'First 8 weeks','Last 4 weeks')
    phases=df.groupby(phase).agg(shipments=('shipment_id','size'),on_time=('on_time','mean'),hub=('hub_hours','mean'),cost=('cost_inr','mean'))
    busy=daily.shipments>35
    summary={'mode':'SYNTHETIC ONLY','seed':42,'shipments':len(df),'days':len(daily),'on_time_pct':float(df.on_time.mean()*100),'late_shipments':int((~df.on_time).sum()),'total_cost_inr':float(df.cost_inr.sum()),'mean_cost_inr':float(df.cost_inr.mean()),'mean_delivery_hours':float(df.delivery_hours.mean()),'median_delivery_hours':float(df.delivery_hours.median()),'p90_delivery_hours':float(df.delivery_hours.quantile(.9)),'late_mean_hours':float(df.loc[~df.on_time,'late_hours'].mean()),'iqr_upper_hours':float(upper),'upper_tail_shipments':int((df.delivery_hours>upper).sum()),'distance_cost_pearson':float(pearson.loc['distance_km','cost_inr']),'distance_cost_spearman':float(spearman.loc['distance_km','cost_inr']),'weight_cost_pearson':float(pearson.loc['weight_kg','cost_inr']),'daily_volume_hub_pearson':float(daily.shipments.corr(daily.hub_hours)),'busy_days':int(busy.sum()),'other_days':int((~busy).sum()),'busy_day_mean_hub':float(daily.loc[busy,'hub_hours'].mean()),'other_day_mean_hub':float(daily.loc[~busy,'hub_hours'].mean()),'modal_region':str(df.region.mode().iloc[0]),'phases':phases.to_dict(orient='index'),'versions':{'pandas':pd.__version__,'numpy':np.__version__,'matplotlib':matplotlib.__version__}}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.dpi':130})
    def save(fig,name):
        fig.text(.01,.01,'Synthetic demonstration • Seed 42 • Not real operational performance',fontsize=8,color='#555555')
        fig.tight_layout(rect=[0,.035,1,1]); fig.savefig(charts/name,dpi=160); plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(9,5),sharex=True)
    axs[0].bar(weekly.index,weekly.shipments,width=4,color=COLORS[0]); axs[0].set_ylabel('Shipments'); axs[0].set_title('Weekly shipment volume and on-time delivery')
    axs[1].plot(weekly.index,weekly.on_time*100,'o-',color=COLORS[1]); axs[1].set_ylabel('On-time delivery (%)'); axs[1].set_ylim(0,105); axs[1].set_xlabel('Week starting Monday')
    for ax in axs: ax.axvline(pd.Timestamp('2026-03-02'),color='#777777',linestyle='--',alpha=.7); ax.grid(axis='y',alpha=.2)
    save(fig,'01_weekly.png')
    fig,ax=plt.subplots(figsize=(9,3.8)); ax.hist(df.delivery_hours,bins=np.arange(0,121,5),color=COLORS[0],edgecolor='white')
    for val,label,color in [(summary['mean_delivery_hours'],'Mean',COLORS[1]),(summary['median_delivery_hours'],'Median',COLORS[2]),(summary['p90_delivery_hours'],'90th percentile',COLORS[3])]: ax.axvline(val,color=color,linestyle='--',label=f'{label}: {val:.1f} h')
    ax.set(xlabel='Delivery duration (hours)',ylabel='Shipments',title='Delivery duration distribution'); ax.legend(fontsize=9); save(fig,'02_distribution.png')
    fig,ax=plt.subplots(figsize=(9,3.8)); b=ax.boxplot([df.loc[df.region==r,'delivery_hours'] for r in REGIONS],tick_labels=[f'{r}\nn={int(region.loc[r,"shipments"])}' for r in REGIONS],patch_artist=True)
    for patch,c in zip(b['boxes'],COLORS):patch.set_facecolor(c);patch.set_alpha(.6)
    ax.set(ylabel='Delivery duration (hours)',title='Regional duration spread and tail delays'); ax.grid(axis='y',alpha=.2);save(fig,'03_regions.png')
    fig,ax=plt.subplots(figsize=(9,4)); sc=ax.scatter(df.distance_km,df.cost_inr,c=df.weight_kg,cmap='viridis',s=12,alpha=.65,rasterized=True);fig.colorbar(sc,ax=ax,label='Shipment weight (kg)')
    ax.set(xlabel='Shipment route distance (km)',ylabel='Allocated transport cost (INR / shipment)',title='Distance and shipment weight are built-in cost drivers');save(fig,'04_cost.png')
    fig,ax=plt.subplots(figsize=(8,5)); im=ax.imshow(pearson,vmin=-1,vmax=1,cmap='RdBu_r');labs=['Distance','Weight','Hub time','Transit','Delivery','Cost']
    ax.set_xticks(range(6),labs);ax.set_yticks(range(6),labs)
    for i in range(6):
        for j in range(6):ax.text(j,i,f'{pearson.iloc[i,j]:.2f}',ha='center',va='center',color='white' if abs(pearson.iloc[i,j])>.6 else 'black')
    fig.colorbar(im,ax=ax,label='Pearson correlation');ax.set_title('Relationships among shipment measures');save(fig,'05_correlations.png')
    fig,ax=plt.subplots(figsize=(9,4));ax.scatter(daily.shipments,daily.hub_hours,c=np.where(busy,COLORS[1],COLORS[0]),s=35,alpha=.8)
    ax.axvline(35,color='#777777',linestyle='--',label='Busy-day definition: more than 35 shipments')
    ax.set(xlabel='Shipments dispatched per day',ylabel='Daily mean hub processing time (hours)',title=f'Hub congestion signal across {len(daily)} simulated days');ax.legend(fontsize=8);save(fig,'06_bottleneck.png')
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path('outputs/week3'));args=parser.parse_args()
    print(json.dumps(analyze(simulate(),args.output),indent=2))
