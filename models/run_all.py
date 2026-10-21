"""Reproduce four forecasts and a common expanding-window backtest."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import scipy
import statsmodels

from common import ROOT, DataStore, fit_model, write_json

MODELS = ['qpm','arima','sarima','var']


def summarize(backtest):
    rows = []
    for (model,horizon),group in backtest.groupby(['model','horizon']):
        error = group.prediction-group.actual
        benchmark = group.baseline-group.actual
        mae,rmse = float(error.abs().mean()),float(np.sqrt(np.mean(error**2)))
        bmae,brmse = float(benchmark.abs().mean()),float(np.sqrt(np.mean(benchmark**2)))
        rows.append(dict(model=model,horizon=int(horizon),n=len(group),MAE_pp=mae,RMSE_pp=rmse,
                         bias_pp=float(error.mean()),baseline_MAE_pp=bmae,baseline_RMSE_pp=brmse,
                         MAE_ratio_to_no_change=mae/bmae,RMSE_ratio_to_no_change=rmse/brmse,
                         coverage80=float(((group.actual>=group.lower80)&(group.actual<=group.upper80)).mean()),
                         coverage95=float(((group.actual>=group.lower95)&(group.actual<=group.upper95)).mean()),
                         mean_width80_pp=float((group.upper80-group.lower80).mean())))
    return pd.DataFrame(rows).sort_values(['horizon','RMSE_pp'])


def execute(data, output, start='2018Q4', horizon=9):
    store = DataStore(data)
    output.mkdir(parents=True,exist_ok=True)
    store.inventory().to_csv(output/'data_usage.csv',index=False)
    store.context().to_csv(output/'auxiliary_context.csv',index=False)
    forecasts,metadata = [],{}
    latest = store.frame(store.latest)
    latest.to_csv(output/'latest_information_set.csv',index_label='period')
    for name in MODELS:
        print(f'Latest forecast: {name.upper()}',flush=True)
        model,forecast,meta = fit_model(name,latest,horizon)
        sub = output/name; sub.mkdir(exist_ok=True)
        forecast.to_csv(sub/'forecast.csv',index=False)
        write_json(sub/'metadata.json',meta)
        metadata[name] = meta
        forecasts.append(forecast.assign(model=name.upper()))
        if name == 'qpm':
            model.filtered.to_csv(sub/'filtered_states.csv',index_label='period')
            model.paths.to_csv(sub/'forecast_states.csv',index_label='period')
            model.observed.to_csv(sub/'measurement_inputs.csv',index_label='period')
            pd.DataFrame(model.innovations).to_csv(sub/'innovations.csv',index=False)
        if name == 'var': model.nowcasts.to_csv(sub/'ragged_edge_nowcast.csv',index=False)
    forecast_all = pd.concat(forecasts,ignore_index=True)
    forecast_all.to_csv(output/'forecast_comparison.csv',index=False)

    rows,fold_meta = [],[]
    origins = pd.period_range(start,store.latest-1,freq='Q')
    for number,origin in enumerate(origins,1):
        frame = store.frame(origin)
        print(f'Backtest {number}/{len(origins)}: {origin}',flush=True)
        maximum = min(4,store.latest.ordinal-origin.ordinal)
        for name in MODELS:
            _,forecast,meta = fit_model(name,frame,maximum)
            fold_meta.append(dict(model=name,origin=str(origin),metadata=meta))
            for _,point in forecast.iterrows():
                if int(point.horizon) not in [1,2,4]: continue
                target = pd.Period(point.period,freq='Q')
                rows.append(dict(model=name.upper(),origin=str(origin),period=str(target),
                                 horizon=int(point.horizon),prediction=float(point['mean']),
                                 actual=float(store.target.loc[target]),baseline=float(frame.ir_pol.iloc[-1]),
                                 lower80=float(point.lower80),upper80=float(point.upper80),
                                 lower95=float(point.lower95),upper95=float(point.upper95),
                                 evaluation_regime='pre_2022' if target<pd.Period('2022Q1') else '2022_onwards'))
        # Checkpoint allows resuming analysis of already completed folds after interruption.
        pd.DataFrame(rows).to_csv(output/'backtest_predictions.csv',index=False)
    backtest = pd.DataFrame(rows)
    counts = backtest.groupby(['origin','horizon']).model.nunique()
    if not counts.eq(4).all(): raise AssertionError('Models evaluated on different cases')
    metrics = summarize(backtest)
    metrics.to_csv(output/'metrics.csv',index=False)
    regimes = []
    for regime,group in backtest.groupby('evaluation_regime'):
        regimes.append(summarize(group).assign(regime=regime))
    pd.concat(regimes,ignore_index=True).to_csv(output/'metrics_by_regime.csv',index=False)
    write_json(output/'fold_metadata.json',fold_meta)
    source_files = sorted(list((Path(data)/'inputs').glob('*.csv'))+list((Path(data)/'model').glob('*.csv'))+
                          list((ROOT/'nbu_challenge_model_2026/model').glob('*')))
    manifest = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in source_files}
    code_manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'models').glob('*.py'))}
    write_json(output/'run_manifest.json',dict(origin=str(store.latest),as_of='2026-10-21',forecast_horizon=horizon,
               target='quarterly calendar-day average NBU policy rate, percent per annum',
               backtest_start=start,release_proxy='quarter end +21 days; UA and EA GDP lagged one quarter',
               vintage='current revised vintage 2026-10-20/21; not archived real-time vintages',
               stochastic_seed='none; deterministic linear Gaussian computations',
               python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
               scipy=scipy.__version__,statsmodels=statsmodels.__version__,input_sha256=manifest,code_sha256=code_manifest))
    return forecast_all,backtest,metrics,metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=ROOT/'data')
    parser.add_argument('--output',type=Path,default=ROOT/'results'/'2026-10-21')
    parser.add_argument('--start',default='2018Q4')
    parser.add_argument('--horizon',type=int,default=9)
    args = parser.parse_args()
    start = time.perf_counter()
    _,_,metrics,_ = execute(args.data,args.output,args.start,args.horizon)
    print(metrics.to_string(index=False))
    print(f'Completed in {time.perf_counter()-start:.1f}s',flush=True)
