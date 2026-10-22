"""Headline-CPI satellites, separated from a normative interest-rate decision.

All folds refit on prefixes of the current saved vintage. Historical publication
dates are incomplete: these are chronological diagnostics, not real-time tests.
The extra FX feature is lagged; no intervention coefficient is asserted causal.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from common import ROOT, read_periods, write_json

AS_OF = '2026-10-22'


def evaluation_split(origin, target):
    cutoff = pd.Period('2024-08', 'M')
    if pd.Period(target, 'M') <= cutoff:
        return 'validation'
    if pd.Period(origin, 'M') >= cutoff:
        return 'audit'
    return 'embargo'


def feature_data(policy_path=None):
    core = read_periods(ROOT/'data/inputs/core_monthly.csv', 'M').loc[:'2026-09']
    auxiliary = read_periods(ROOT/'data/model/auxiliary_monthly.csv', 'M')
    x = pd.DataFrame(index=core.index)
    x['inflation'] = 100*np.log1p(core.CPI_UA_MOM/100)
    x['core_inflation'] = 100*np.log1p(core.CORE_CPI_UA_MOM/100)
    x['fx_change'] = 100*np.log(core.ER_UAH_EUR_AVG).diff()
    x['brent_change'] = 100*np.log(auxiliary.brent).diff().reindex(x.index)
    if policy_path is not None and Path(policy_path).exists():
        policy = read_periods(policy_path, 'M')
        # Month m pressure is reported only after month m. A further month lag
        # is explicit, rather than assigning an invented publication day.
        reserves = policy.gross_reserves_usd_mn.shift(1)
        pressure = -policy.fx_net_purchases_usd_mn/reserves*100
        x['fx_pressure_lag1'] = pressure.shift(1).reindex(x.index)
    return core, x.replace([np.inf, -np.inf], np.nan)


def forecast_log_monthly(data, model_name, horizon):
    """Forecast one-month log-price changes; no preprocessing sees the future."""
    history=data.inflation.dropna().loc['2010-01':]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if model_name == 'SEASONAL_NAIVE':
            predictions = np.resize(history.iloc[-12:].to_numpy(), horizon)
            return predictions, {'method':'last completed 12 months repeated'}
        if model_name == 'ARIMA':
            fits=[]
            for order in [(0,0,0),(1,0,0),(2,0,0),(0,0,1),(1,0,1)]:
                try:
                    f=ARIMA(history,order=order,trend='c').fit()
                    if np.isfinite(f.aicc): fits.append((f.aicc,order,f))
                except (ValueError,np.linalg.LinAlgError): continue
            if not fits: raise RuntimeError('No finite ARIMA fit')
            _,order,fit=min(fits,key=lambda z:z[0])
            return np.asarray(fit.forecast(horizon)), {'order':order,'aicc':float(fit.aicc)}
        if model_name == 'SARIMA':
            fits=[]
            for order,season in [((1,0,0),(0,1,1,12)),((0,0,1),(1,0,0,12))]:
                try:
                    f=SARIMAX(history,order=order,seasonal_order=season,trend='c',
                              enforce_stationarity=True,enforce_invertibility=True).fit(disp=False,maxiter=150)
                    if np.isfinite(f.aicc) and f.mle_retvals.get('converged',False):
                        fits.append((f.aicc,order,season,f))
                except (ValueError,np.linalg.LinAlgError): continue
            if not fits: raise RuntimeError('No converged finite SARIMA fit')
            _,order,season,fit=min(fits,key=lambda z:z[0])
            return np.asarray(fit.forecast(horizon)), {'order':order,'seasonal_order':season,'aicc':float(fit.aicc)}
        columns=['inflation','core_inflation','fx_change','brent_change']
        if model_name=='VAR_POLICY': columns+=['fx_pressure_lag1']
        joint=data[columns].dropna().loc['2010-01':]
        if len(joint)<72: raise ValueError(f'{model_name}: fewer than 72 complete months')
        # Require a contiguous final segment, never silently compress time gaps.
        breaks=np.flatnonzero(np.diff(joint.index.asi8)!=1)
        if len(breaks): joint=joint.iloc[breaks[-1]+1:]
        if len(joint)<72: raise ValueError(f'{model_name}: final complete segment shorter than 72 months')
        fits=[]
        for lag in [1,2]:
            try:
                f=VAR(joint).fit(lag,trend='c')
                if f.is_stable() and np.isfinite(f.bic): fits.append((f.bic,lag,f))
            except (ValueError,np.linalg.LinAlgError): continue
        if not fits: raise RuntimeError(f'{model_name}: no stable VAR candidate')
        _,lag,fit=min(fits,key=lambda z:z[0])
        return fit.forecast(joint.to_numpy()[-lag:],horizon)[:,0], {
            'lags':lag,'columns':columns,'bic':float(fit.bic),
            'training_start':str(joint.index[0]),'training_n':len(joint)}


def to_headline_yoy(core, origin, monthly_log_forecast):
    """Exact compounding against the observed 12-month lag, with no future base."""
    origin=pd.Period(origin,freq='M')
    index=core.CPI_UA_INDEX_NSA_2021_100.loc[:origin].dropna()
    levels=100*np.log(index)
    path=[];last=float(levels.loc[origin])
    for h,change in enumerate(monthly_log_forecast,1):
        target=origin+h;last+=float(change)
        annual_lag=target-12
        if annual_lag not in levels.index: raise ValueError('Unsupported annual price base')
        path.append(dict(period=str(target),horizon=h,headline_mom_pct=float(100*np.expm1(change/100)),
                         headline_yoy_pct=float(100*np.expm1((last-levels.loc[annual_lag])/100))))
    return path


def run_nowcasts(output, policy_path=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    core,features=feature_data(policy_path)
    models=['SEASONAL_NAIVE','ARIMA','SARIMA','VAR']
    if 'fx_pressure_lag1' in features: models+=['VAR_POLICY']
    rows=[];failures=[]
    # Origins/targets are chosen once before looking at new results.
    for origin in pd.period_range('2022-12','2026-08',freq='M'):
        available=min(4,(pd.Period('2026-09','M')-origin).n)
        if available<1:continue
        for name in models:
            try:
                f,metadata=forecast_log_monthly(features.loc[:origin],name,available)
                for p in to_headline_yoy(core,origin,f):
                    actual=float(core.loc[pd.Period(p['period'],'M'),'CPI_UA_YOY_PUBLISHED'])
                    rows.append(dict(origin=str(origin),model=name,split=evaluation_split(origin,p['period']),actual=actual,
                                     error=p['headline_yoy_pct']-actual,**p))
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as e:
                failures.append(dict(origin=str(origin),model=name,error=str(e)))
    past=pd.DataFrame(rows);past.to_csv(output/'cpi_backtest.csv',index=False)
    pd.DataFrame(failures,columns=['origin','model','error']).to_csv(output/'cpi_fit_failures.csv',index=False)
    metrics=[]
    for (split,h),g in past.loc[past.split!='embargo'].groupby(['split','horizon']):
        # Pair all eligible models on common origins. Report eligibility failures.
        eligibility=g.groupby('model').origin.nunique()
        eligible=[n for n in models if eligibility.get(n,0)>=6]
        common=set.intersection(*(set(g.loc[g.model==n,'origin']) for n in eligible))
        for name in eligible:
            a=g.loc[(g.model==name)&g.origin.isin(common)]
            metrics.append(dict(split=split,horizon=int(h),model=name,n=len(a),
                                MAE_pp=float(a.error.abs().mean()),RMSE_pp=float(np.sqrt((a.error**2).mean()))))
    metrics=pd.DataFrame(metrics);metrics.to_csv(output/'cpi_metrics.csv',index=False)
    forecasts=[];metadata={};last=pd.Period('2026-09','M')
    for name in models:
        try:
            f,meta=forecast_log_monthly(features.loc[:last],name,4);metadata[name]=meta
            forecasts += [dict(model=name,**r) for r in to_headline_yoy(core,last,f)]
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as e:metadata[name]={'failure':str(e)}
    forecasts=pd.DataFrame(forecasts)
    selected={}
    for h in [1,2,3,4]:
        rank=metrics.loc[(metrics.split=='validation')&(metrics.horizon==h)].sort_values(['MAE_pp','RMSE_pp','model'])
        selected[h]=rank.iloc[0].model
    forecasts['selected_by_validation']=[selected[int(h)]==n for h,n in zip(forecasts.horizon,forecasts.model)]
    forecasts.to_csv(output/'cpi_forecast.csv',index=False)
    write_json(output/'cpi_protocol.json',dict(as_of=AS_OF,training_end='2026-09',target='headline CPI, NSA monthly and year-on-year, percent',
        requested_months=['2026-11','2026-12','2027-01'],bridge_month='2026-10',selected_models=selected,
        methods=metadata,selection='validation targets through 2024-08; audit origins from 2024-08; intervening rows embargoed; validation MAE then RMSE on common eligible origins, never audit',
        information='current revised workbook vintage; prefix fitting; auxiliary pressure delayed one month; historical release days unknown',
        interpretation='Statistical satellite, not an interest-rate-policy counterfactual. QPM SA quarterly-average inflation is a different observable.',
        model_failure_count=len(failures),policy_dataset_used=policy_path is not None))
    print('Headline CPI satellites saved:',forecasts.loc[forecasts.selected_by_validation,['period','model','headline_yoy_pct']].to_dict('records'),flush=True)
    return forecasts,metrics


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/2026-10-22-policy-decision/nowcast')
    p.add_argument('--policy-data',type=Path,default=ROOT/'data/policy/2026-10-22/monthly.csv')
    a=p.parse_args();run_nowcasts(a.output,a.policy_data)
