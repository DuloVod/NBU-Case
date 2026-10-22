"""Chronological, predeclared monthly policy-rate benchmark research.

A historical observation is eligible for fitting/calibration only after its
TARGET month has ended. The current-month average is therefore never a label.
This is pseudo-real-time research: historical macro revisions are retained.
"""
from __future__ import annotations
import argparse, hashlib, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT, write_json
from monthly_common import MonthlyData

VALIDATION_END = pd.Period('2024-09', 'M')
TEST_START = pd.Period('2024-10', 'M')
RIDGE_SPECS = {'ridge_policy_10': (False, 10., None),
               'ridge_macro_100': (True, 100., None),
               'ridge_policy_100_60m': (False, 100., 60)}


def information(data, origin, day=22):
    """Build features from rates known on day 22 and macro through prior month."""
    origin = pd.Period(origin, 'M'); date = origin.start_time + pd.Timedelta(days=min(day, origin.days_in_month)-1)
    completed = data.means.loc[:origin-1]
    if len(completed) < 14: raise ValueError('Insufficient historical observations')
    spot = float(data.daily.loc[:date].iloc[-1])
    changes = completed.diff()
    policy = [spot-float(completed.iloc[-1]), *changes.iloc[-3:].tolist(), spot]
    macro = data.raw.loc[:origin-1]
    cpi = macro.CPI_UA_INDEX_NSA_2021_100
    fx = macro.ER_UAH_EUR_AVG
    ecb = macro.IR_ECB_DFR_CALENDAR_AVG
    external = [100*np.log(cpi.iloc[-1]/cpi.iloc[-13]),
                100*np.log(fx.iloc[-1]/fx.iloc[-4]),float(ecb.iloc[-1]),
                float(ecb.iloc[-1]-ecb.iloc[-4])]
    return dict(origin=origin, target=origin+1, as_of=date, spot=spot,
                policy=np.array(policy), macro=np.array(policy+external),
                drift6=float(changes.iloc[-6:].mean()), drift12=float(changes.iloc[-12:].mean()))


def eligible(rows, origin):
    """Month origin is partial; only labels through origin-1 are observed."""
    return [row for row in rows if row['target'] < origin and np.isfinite(row.get('actual', np.nan))]


def ridge_prediction(history, current, macro=False, alpha=10., window=None):
    history = eligible(history, current['origin'])
    if window: history = history[-window:]
    if len(history) < 24: return current['spot']
    key = 'macro' if macro else 'policy'
    X = np.asarray([row[key] for row in history]); x = current[key]
    y = np.array([row['actual']-row['spot'] for row in history])
    center=X.mean(axis=0);scale=X.std(axis=0);scale[scale<1e-8]=1.
    Z=(X-center)/scale;z=(x-center)/scale
    # Intercept is also shrunk: no evidence defaults to the no-change model.
    Z=np.column_stack([np.ones(len(Z)),Z]); z=np.r_[1.,z]
    beta=np.linalg.solve(Z.T@Z+alpha*np.eye(Z.shape[1]),Z.T@y)
    return float(current['spot']+z@beta)


def optimal_shrink(history, origin, base_name, current_base, spot, window=36):
    """Shrink a candidate towards spot using only already-resolved errors."""
    past=eligible(history,origin)[-window:]
    past=[row for row in past if base_name in row['predictions']]
    if len(past)<12:return spot
    d=np.array([row['predictions'][base_name]-row['spot'] for row in past])
    y=np.array([row['actual']-row['spot'] for row in past])
    # Fixed conservative bound; never optimized using held-out outcomes.
    weight=float(np.clip(d@y/(d@d+1.),0.,.5))
    return spot+weight*(current_base-spot)


def interval(history, origin, name, prediction, coverage):
    residuals=np.array([abs(row['actual']-row['predictions'][name]) for row in eligible(history,origin)
                        if name in row['predictions']])
    if len(residuals)<12:return (np.nan,np.nan,len(residuals))
    # Finite-sample conformal order statistic, using all past resolved errors.
    k=min(len(residuals),int(np.ceil((len(residuals)+1)*coverage)))
    radius=float(np.sort(residuals)[k-1])
    return prediction-radius,prediction+radius,len(residuals)


def score(frame):
    rows=[]
    for (split,model),g in frame.groupby(['split','model']):
        error=g.prediction-g.actual
        row=dict(split=split,model=model,n=len(g),MAE_pp=float(error.abs().mean()),
                 RMSE_pp=float(np.sqrt(np.mean(error**2))),bias_pp=float(error.mean()))
        for level in (80,95):
            valid=g[f'lower{level}'].notna()
            row[f'coverage{level}']=float(((g.actual>=g[f'lower{level}'])&(g.actual<=g[f'upper{level}']))[valid].mean())
            row[f'mean_width{level}_pp']=float((g[f'upper{level}']-g[f'lower{level}'])[valid].mean())
            row[f'interval_n{level}']=int(valid.sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['split','MAE_pp','RMSE_pp'])


def paired_bootstrap(frame, baseline='NO_CHANGE', repeats=4000, block=3):
    """Paired circular block bootstrap; descriptive under structural breaks."""
    test=frame.loc[frame.split=='test'].pivot(index='period',columns='model',values=['prediction','actual'])
    rng=np.random.default_rng(20261022);n=len(test);rows=[]
    base=np.abs(test.prediction[baseline]-test.actual[baseline]).to_numpy()
    starts=rng.integers(0,n,size=(repeats,int(np.ceil(n/block))))
    samples=np.concatenate([(starts+i)%n for i in range(block)],axis=1)[:,:n]
    for name in test.prediction.columns:
        loss=np.abs(test.prediction[name]-test.actual[name]).to_numpy()-base
        means=loss[samples].mean(axis=1)
        rows.append(dict(model=name,n=n,MAE_difference_vs_nochange_pp=float(loss.mean()),
                         bootstrap_lower95_pp=float(np.quantile(means,.025)),
                         bootstrap_upper95_pp=float(np.quantile(means,.975)),block_months=block,repeats=repeats))
    return pd.DataFrame(rows)


def run(output=None):
    output=Path(output or ROOT/'results/2026-10-22-research');output.mkdir(parents=True,exist_ok=True)
    data=MonthlyData();current=pd.Period('2026-10','M')
    cached=pd.read_csv(ROOT/'results/2026-10-22-monthly/backtest_predictions.csv')
    latest=pd.read_csv(ROOT/'results/2026-10-22-monthly/next_month_forecast.csv')
    cache={(pd.Period(row.origin,'M'),row.model):float(row.prediction) for row in cached.itertuples()}
    for row in latest.itertuples():cache[(current,row.model)]=float(row.mean)
    history=[];records=[];latest_rows=[]
    for origin in pd.period_range('2011-03',current,freq='M'):
        row=information(data,origin)
        row['actual']=float(data.means.loc[row['target']]) if row['target']<current else np.nan
        spot=row['spot'];predictions={'NO_CHANGE':spot,'DRIFT_6':spot+row['drift6'], 'DRIFT_12':spot+row['drift12']}
        for name,(macro,alpha,window) in RIDGE_SPECS.items():
            predictions[name]=ridge_prediction(history,row,macro,alpha,window)
        for name in ('ARIMA','SARIMA','VAR','QPM_BRIDGE'):
            if (origin,name) in cache:predictions[name]=cache[(origin,name)]
        if 'ARIMA' in predictions:
            predictions['ONLINE_ARIMA_SHRINK']=optimal_shrink(history,origin,'ARIMA',predictions['ARIMA'],spot)
            names=('ARIMA','SARIMA','VAR','QPM_BRIDGE')
            # Inverse past MAE weights, then conservative shrink to no-change.
            past=eligible(history,origin)[-36:]
            common=[r for r in past if all(n in r['predictions'] for n in names)]
            if len(common)>=12:
                loss=np.array([np.mean([abs(r['actual']-r['predictions'][n]) for r in common]) for n in names])
                weights=1/np.maximum(loss,.05);weights/=weights.sum()
                blend=float(sum(w*predictions[n] for w,n in zip(weights,names)))
            else:blend=spot
            predictions['ONLINE_BLEND_RAW']=blend
            predictions['ONLINE_BLEND_SHRINK']=optimal_shrink(history,origin,'ONLINE_BLEND_RAW',blend,spot)
        row['predictions']=predictions
        for name,prediction in predictions.items():
            if name=='ONLINE_BLEND_RAW':continue # internal input, not a selection candidate
            lo80,hi80,ncal=interval(history,origin,name,prediction,.8)
            lo95,hi95,_=interval(history,origin,name,prediction,.95)
            record=dict(model=name,origin=str(origin),as_of=str(row['as_of'].date()),period=str(row['target']),
                        training_label_end=str(origin-1),prediction=float(prediction),actual=row['actual'],
                        baseline=spot,lower80=lo80,upper80=hi80,lower95=lo95,upper95=hi95,calibration_n=ncal)
            if origin==current:latest_rows.append(record)
            elif origin>=pd.Period('2019-01','M') and row['target']<=pd.Period('2026-09','M'):
                record['split']='validation' if row['target']<=VALIDATION_END else 'test';records.append(record)
        history.append(row)
    frame=pd.DataFrame(records);metrics=score(frame)
    validation=metrics.loc[metrics.split=='validation'].sort_values(['MAE_pp','RMSE_pp','model'])
    winner=validation.iloc[0].model
    final=pd.DataFrame(latest_rows);final['selected_by_validation']=final.model==winner
    frame.to_csv(output/'monthly_predictions.csv',index=False)
    metrics.to_csv(output/'monthly_metrics.csv',index=False)
    final.to_csv(output/'monthly_november_forecast.csv',index=False)
    paired_bootstrap(frame).to_csv(output/'monthly_paired_uncertainty.csv',index=False)
    features=[]
    for row in history:
        features.append(dict(origin=str(row['origin']),target=str(row['target']),as_of=str(row['as_of'].date()),
                             spot=row['spot'],actual=row['actual'],spot_minus_prior_average=row['policy'][0],
                             rate_change_lag3=row['policy'][1],rate_change_lag2=row['policy'][2],rate_change_lag1=row['policy'][3],
                             inflation_log_yoy=row['macro'][5],fx_log_change_3m=row['macro'][6],
                             ecb_level=row['macro'][7],ecb_change_3m=row['macro'][8]))
    pd.DataFrame(features).to_csv(output/'monthly_feature_audit.csv',index=False)
    protocol=dict(as_of='2026-10-22',target='November 2026 calendar-day average policy rate, percent per annum',
                  selected_model=winner,selection='Minimum validation MAE; ties broken by RMSE then model name. Selection frozen before inspecting test metrics.',
                  validation='2019-02 to 2024-09 (68 targets)',test='2024-10 to 2026-09 (24 targets)',
                  test_is_preregistered=False,
                  research_warning='This split/protocol was declared before this research run, but earlier project results already included these dates. This is a chronological audit holdout, not a pristine never-seen external test. No test-based tuning is performed here.',
                  labels='Only target months strictly before the partial current month may train models or calibrate intervals/online weights.',
                  candidates=list(final.model),ridge_penalty='10 or 100 fixed; standardized features; penalized intercept; 24 minimum labels; expanding or latest60 labels',
                  features=['spot-minus-prior-month-average','latest3 completed monthly rate changes','current spot',
                            'prior-month CPI year-on-year log change','prior3-month FX log change','ECB prior-month level and3-month change'],
                  macro_availability='Previous calendar month assumed available on day22. Current-vintage macro data, not archived publication vintages.',
                  online='At most0.5 weight on alternative vs nochange; weights from latest36 resolved labels; minimum12; ridge1 penalty on weight denominator.',
                  intervals='Symmetric absolute-error finite-sample quantiles from expanding resolved residuals only. Marginal descriptive coverage, not regime-conditional guarantee.',
                  baseline='Effective rate on day22, held constant throughout nextmonth',
                  limitation='Discrete meeting schedule and surprise decisions are not modeled; rate history source may contain retrospective corrections; war breaks invalidate exchangeability.',
                  seed=20261022,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                  hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'data/snapshots/2026-10-22/policy_archive.json',ROOT/'data/inputs/core_monthly.csv',ROOT/'data/inputs/policy_decisions.csv',ROOT/'results/2026-10-22-monthly/backtest_predictions.csv',ROOT/'results/2026-10-22-monthly/next_month_forecast.csv']})
    write_json(output/'monthly_protocol.json',protocol)
    print(metrics.to_string(index=False));print('Selected:',winner);print(final[['model','prediction','lower80','upper80','selected_by_validation']].to_string(index=False))
    return frame,metrics,final,protocol

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args();run(args.output)
