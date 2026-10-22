"""Forecast NEXT month's daily-average NBU rate using a dated information set."""
from pathlib import Path
import argparse,hashlib,importlib,json,time
import numpy as np
import pandas as pd
import platform,scipy,statsmodels
from common import ROOT,write_json
from monthly_common import MonthlyData

MODELS=['qpm','arima','sarima','var']
LABELS={'qpm':'QPM_BRIDGE','arima':'ARIMA','sarima':'SARIMA','var':'VAR'}


def summarize(frame):
    result=[]
    for model,group in frame.groupby('model'):
        error=group.prediction-group.actual;baseline=group.baseline-group.actual
        result.append(dict(model=model,n=len(group),MAE_pp=float(error.abs().mean()),
                           RMSE_pp=float(np.sqrt(np.mean(error**2))),bias_pp=float(error.mean()),
                           baseline_MAE_pp=float(baseline.abs().mean()),baseline_RMSE_pp=float(np.sqrt(np.mean(baseline**2))),
                           coverage80=float(((group.actual>=group.lower80)&(group.actual<=group.upper80)).mean()),
                           coverage95=float(((group.actual>=group.lower95)&(group.actual<=group.upper95)).mean())))
    return pd.DataFrame(result).sort_values('RMSE_pp')


def run(as_of='2026-10-22',start='2019-01',output=None):
    date=pd.Timestamp(as_of);current=date.to_period('M');target=current+1
    output=Path(output or ROOT/f'results/{date.date()}-monthly');output.mkdir(parents=True,exist_ok=True)
    data=MonthlyData()
    if current-1>data.completed_last:raise ValueError('Missing completed previous-month macro data')
    history=data.frame(current-1);history.to_csv(output/'monthly_training_information.csv',index_label='period')
    pd.DataFrame({'rate':data.daily.loc[:date]}).to_csv(output/'policy_daily_as_of.csv')
    measurement=data.partial_measurement(date)
    forecasts=[];latest_metadata={}
    for model in MODELS:
        print('Latest monthly forecast:',model,flush=True)
        table,metadata=importlib.import_module('monthly_'+model).forecast(data,date)
        if table.period.tolist()!=[str(current),str(target)]:raise AssertionError('Wrong month alignment')
        sub=output/model;sub.mkdir(exist_ok=True)
        table.to_csv(sub/'forecast.csv',index=False);write_json(sub/'metadata.json',metadata)
        forecasts.append(table.assign(model=LABELS[model]));latest_metadata[model]=metadata
    comparison=pd.concat(forecasts,ignore_index=True)
    comparison.to_csv(output/'forecast_comparison.csv',index=False)
    comparison.loc[comparison.period==str(target)].to_csv(output/'next_month_forecast.csv',index=False)
    rows=[];fold_metadata=[]
    last_origin=min(data.completed_last,current-1)-1
    for number,origin in enumerate(pd.period_range(start,last_origin,freq='M'),1):
        evaluation_date=origin.start_time+pd.Timedelta(days=min(date.day,origin.days_in_month)-1)
        print(f'Monthly backtest {number}: {evaluation_date.date()} -> {origin+1}',flush=True)
        baseline=float(data.daily.loc[:evaluation_date].iloc[-1])
        for model in MODELS:
            table,metadata=importlib.import_module('monthly_'+model).forecast(data,evaluation_date)
            point=table.loc[table.period==str(origin+1)].iloc[0]
            rows.append(dict(model=LABELS[model],origin=str(origin),as_of=str(evaluation_date.date()),period=str(origin+1),
                             horizon_months=1,training_end=str(origin-1),prediction=float(point['mean']),
                             actual=float(data.means.loc[origin+1]),baseline=baseline,lower80=float(point.lower80),
                             upper80=float(point.upper80),lower95=float(point.lower95),upper95=float(point.upper95),
                             regime='pre_2022' if origin+1<pd.Period('2022-01','M') else '2022_onwards'))
            fold_metadata.append(dict(model=LABELS[model],as_of=str(evaluation_date.date()),metadata=metadata))
        pd.DataFrame(rows).to_csv(output/'backtest_predictions.csv',index=False)
    backtest=pd.DataFrame(rows)
    if not backtest.groupby('as_of').model.nunique().eq(4).all():raise AssertionError('Unequal model evaluation cases')
    metrics=summarize(backtest);metrics.to_csv(output/'metrics.csv',index=False)
    pd.concat([summarize(group).assign(regime=regime) for regime,group in backtest.groupby('regime')],ignore_index=True).to_csv(output/'metrics_by_regime.csv',index=False)
    write_json(output/'fold_metadata.json',fold_metadata)
    files=list((ROOT/'models').glob('*.py'))+[Path(name) for name in [
        'data/snapshots/2026-10-22/policy_archive.json','data/inputs/core_monthly.csv',
        'data/inputs/policy_decisions.csv','data/model/quarterly_levels.csv',
        'data/model/auxiliary_monthly.csv','data/model/auxiliary_quarterly.csv']]
    files=[file if file.is_absolute() else ROOT/file for file in files]
    write_json(output/'run_manifest.json',dict(as_of=str(date.date()),target_month=str(target),frequency='M',
               target='calendar-day average NBU policy rate over next month; percent per annum',
               completed_training_end=str(current-1),current_month_is_partial=True,known_days=measurement['known_days'],
               current_prefix_average=float(measurement['observed'][0]),current_effective_rate=float(measurement['observed'][1]),
               partial_measurement_bias=measurement['bias'].tolist(),partial_measurement_covariance=measurement['noise'].tolist(),
               historical_origins=int(backtest.origin.nunique()),backtest_predictions=len(backtest),
               versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,statsmodels=statsmodels.__version__),
               quarter_to_month='QPM bridge; quarterly structural parameters unchanged',
               data_vintage='macro 2026-10-20/21, policy archive verified 2026-10-22; historical revisions retained',
               input_files={str(file.relative_to(ROOT)):hashlib.sha256(file.read_bytes()).hexdigest() for file in files}))
    return comparison,backtest,metrics


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--as-of',default='2026-10-22')
    parser.add_argument('--start',default='2019-01')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();began=time.perf_counter()
    comparison,backtest,metrics=run(args.as_of,args.start,args.output)
    print(comparison.to_string(index=False));print(metrics.to_string(index=False))
    print(f'Done in {time.perf_counter()-began:.1f} seconds')
