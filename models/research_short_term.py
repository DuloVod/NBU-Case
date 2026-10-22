"""Three future monthly policy-rate averages and transparent online ensemble.

The horizon is Nov2026--Jan2027, not a medium-term macro projection. Model
selection and ensemble weights use resolved historical targets only. Current
revised vintages and previously inspected audit dates remain limitations.
"""
from pathlib import Path
import argparse,hashlib,importlib,json
import numpy as np
import pandas as pd
from common import ROOT,write_json
from monthly_common import MonthlyData
from research_monthly import interval

OUT=ROOT/'results/2026-10-22-short-term'
BASES=['ARIMA','SARIMA','VAR','QPM_BRIDGE']
MODULES=['arima','sarima','var','qpm']


def ensemble(history,origin,current,spot):
    """Inverse past MAE for four models, then learned shrinkage to current rate."""
    past=[r for r in history if r['target']<origin and np.isfinite(r['actual'])][-36:]
    if len(past)<12:
        return spot,dict(NO_CHANGE=1.,**{name:0. for name in BASES}),spot,len(past)
    losses=np.array([np.mean([abs(r['predictions'][name]-r['actual']) for r in past]) for name in BASES])
    raw_weights=1/np.maximum(losses,.05);raw_weights/=raw_weights.sum()
    blend=float(sum(weight*current[name] for weight,name in zip(raw_weights,BASES)))
    difference=np.array([r['predictions']['BLEND_RAW']-r['spot'] for r in past])
    target=np.array([r['actual']-r['spot'] for r in past])
    alpha=float(np.clip(difference@target/(difference@difference+1.),0.,.5))
    weights=dict(NO_CHANGE=1-alpha,**{name:alpha*float(weight) for name,weight in zip(BASES,raw_weights)})
    return spot+alpha*(blend-spot),weights,blend,len(past)


def summarize(records):
    rows=[]
    for (split,horizon,model),g in records.groupby(['split','horizon','model']):
        error=g.prediction-g.actual
        valid=g.dropna(subset=['lower80','upper80'])
        rows.append(dict(split=split,horizon=int(horizon),model=model,n=len(g),
                         MAE_pp=float(error.abs().mean()),RMSE_pp=float(np.sqrt(np.mean(error**2))),
                         coverage80=float(((valid.actual>=valid.lower80)&(valid.actual<=valid.upper80)).mean()),
                         interval_n80=len(valid)))
    return pd.DataFrame(rows).sort_values(['horizon','split','MAE_pp','model'])


def run(output=OUT,refresh=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    data=MonthlyData();current=pd.Period('2026-10','M')
    source_files=[ROOT/'models'/name for name in ['research_short_term.py','monthly_arima.py','monthly_sarima.py',
        'monthly_var.py','monthly_qpm.py','arima.py','sarima.py','var.py','qpm.py','monthly_common.py','common.py']]
    source_files += [ROOT/'data/inputs/core_monthly.csv',ROOT/'data/inputs/policy_decisions.csv',
                     ROOT/'data/model/quarterly_levels.csv',ROOT/'data/snapshots/2026-10-22/policy_archive.json']
    source_files += list((ROOT/'nbu_challenge_model_2026/model').glob('*'))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files if p.is_file()}
    cache_file=output/'base_forecast_cache.csv';meta_file=output/'base_cache_manifest.json'
    cache={};metadata=[]
    if cache_file.exists() and meta_file.exists() and not refresh:
        if json.loads(meta_file.read_text())['input_sha256']!=hashes:
            raise ValueError('Short-term cache inputs changed; rerun with --refresh')
        for r in pd.read_csv(cache_file).itertuples():cache[(r.origin,r.model,int(r.horizon))]=float(r.prediction)
    history={h:[] for h in [1,2,3]};records=[];latest=[];weight_rows=[];cache_rows=[]
    origins=list(pd.period_range('2019-01','2026-08',freq='M'))+[current]
    for count,origin in enumerate(origins,1):
        date=origin.start_time+pd.Timedelta(days=21)
        spot=float(data.daily.loc[:date].iloc[-1]);base={h:{} for h in [1,2,3]}
        for name,module in zip(BASES,MODULES):
            keys=[(str(origin),name,h) for h in [1,2,3]]
            if all(key in cache for key in keys):
                values=[cache[key] for key in keys]
            else:
                table,meta=importlib.import_module('monthly_'+module).forecast(data,date,horizon=3)
                values=[float(table.loc[table.period==str(origin+h),'mean'].iloc[0]) for h in [1,2,3]]
                metadata.append(dict(origin=str(origin),model=name,metadata=meta))
            for h,value in zip([1,2,3],values):
                base[h][name]=value
                cache_rows.append(dict(origin=str(origin),model=name,horizon=h,prediction=value))
        for h in [1,2,3]:
            predictions=dict(base[h],NO_CHANGE=spot)
            mean,weights,raw,n=ensemble(history[h],origin,predictions,spot)
            predictions.update(ENSEMBLE=mean,BLEND_RAW=raw)
            target=origin+h
            actual=float(data.means.loc[target]) if target<=data.completed_last else np.nan
            row=dict(origin=origin,target=target,actual=actual,spot=spot,predictions=predictions)
            for name,prediction in predictions.items():
                if name=='BLEND_RAW':continue
                lo,hi,ncal=interval(history[h],origin,name,prediction,.8)
                lo95,hi95,_=interval(history[h],origin,name,prediction,.95)
                record=dict(origin=str(origin),as_of=str(date.date()),training_end=str(origin-1),horizon=h,
                            period=str(target),model=name,prediction=prediction,actual=actual,
                            lower80=lo,upper80=hi,lower95=lo95,upper95=hi95,calibration_n=ncal)
                if origin==current:latest.append(record)
                elif np.isfinite(actual):
                    record['split']='validation' if target<=pd.Period('2024-09','M') else 'audit';records.append(record)
            if origin==current:
                for name,weight in weights.items():
                    weight_rows.append(dict(period=str(target),horizon=h,model=name,weight=weight,
                                            resolved_weight_history=n))
            history[h].append(row)
        if count%10==0 or origin==current:print(f'Three-month forecasts: {count}/{len(origins)}; {origin}',flush=True)
    forecasts=pd.DataFrame(latest);backtest=pd.DataFrame(records);metrics=summarize(backtest)
    selected={int(h):g.sort_values(['MAE_pp','RMSE_pp','model']).iloc[0].model
              for h,g in metrics.loc[metrics.split=='validation'].groupby('horizon')}
    forecasts['selected_by_validation']=[r.model==selected[int(r.horizon)] for r in forecasts.itertuples()]
    forecasts.to_csv(output/'forecast.csv',index=False)
    backtest.to_csv(output/'backtest.csv',index=False)
    metrics.to_csv(output/'metrics.csv',index=False)
    pd.DataFrame(weight_rows).to_csv(output/'ensemble_weights.csv',index=False)
    pd.DataFrame(cache_rows).to_csv(cache_file,index=False)
    if metadata:write_json(output/'base_model_metadata.json',metadata)
    write_json(meta_file,dict(as_of='2026-10-22',input_sha256=hashes))
    write_json(output/'protocol.json',dict(as_of='2026-10-22',target_months=['2026-11','2026-12','2027-01'],
        target='calendar-day average NBU policy rate, percent per annum',horizons_months=[1,2,3],
        base_models=BASES,ensemble='inverse prior MAE across four models; learned shrinkage to no-change',
        weight_history='last36 resolved targets, target<origin; minimum12; cap model-blend weight0.5; ridge denominator1',
        selection='minimum validation MAE for each horizon; RMSE then model-name ties; never select on audit',
        selected_models=selected,validation_target_end='2024-09',audit_targets='2024-10 to2026-09',
        historical_origin_end_by_horizon={str(h):str(data.completed_last-h) for h in [1,2,3]},
        caveats=['current revised vintages and approximate publication lags',
                 'audit dates already inspected, not pristine external holdout',
                 'empirical intervals are not conditional coverage guarantees',
                 'QPM quarterly model uses monthly bridge, not invented monthly GDP',
                 'discrete meeting dates not imposed on statistical point expectations'],
        input_sha256=hashes))
    print(forecasts[['period','model','prediction','selected_by_validation']].to_string(index=False))
    return forecasts,metrics


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--refresh',action='store_true')
    args=parser.parse_args();run(refresh=args.refresh)
