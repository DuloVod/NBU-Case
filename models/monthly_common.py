"""Monthly information sets and partial-month Gaussian measurement bridge.

The target is the calendar-day average of the NEXT month. A partial current
month is never relabelled as a completed monthly observation.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.seasonal import STL
from common import ROOT,DataStore,read_periods,write_json

SNAPSHOT=ROOT/'data/snapshots/2026-10-22/policy_archive.json'


def scalar_forecast_covariance(transition,process_covariance,initial_covariance,design,horizon,measurement_variance=0.):
    """Joint covariance of linear-Gaussian future scalar observations."""
    transition=np.asarray(transition,float)
    design=np.asarray(design,float).reshape(-1)
    covariance=np.asarray(initial_covariance,float).copy()
    future=[]
    for _ in range(horizon):
        covariance=transition@covariance@transition.T+process_covariance
        future.append((covariance+covariance.T)/2)
    result=np.empty((horizon,horizon))
    for i in range(horizon):
        for j in range(i,horizon):
            cross=design@future[i]@np.linalg.matrix_power(transition,j-i).T@design
            if i==j:cross+=measurement_variance
            result[i,j]=result[j,i]=cross
    return result


def sarimax_joint_covariance(fitted,horizon):
    """Use the fitted state-space system; verify its marginal forecast variances."""
    def matrix(name):
        value=np.asarray(fitted.model.ssm[name],float)
        return value[:,:,-1] if value.ndim==3 else value
    transition=matrix('transition');design=matrix('design').reshape(-1)
    selection=matrix('selection');process=selection@matrix('state_cov')@selection.T
    # predicted_state_cov[-1] is already the first FUTURE prior, not filtered P_T.
    first=fitted.filter_results.predicted_state_cov[:,:,-1]
    future=[first]
    for _ in range(1,horizon):future.append(transition@future[-1]@transition.T+process)
    result=np.empty((horizon,horizon));measurement=float(matrix('obs_cov')[0,0])
    for i in range(horizon):
        for j in range(i,horizon):
            value=design@future[i]@np.linalg.matrix_power(transition,j-i).T@design
            if i==j:value+=measurement
            result[i,j]=result[j,i]=value
    np.testing.assert_allclose(np.diag(result),np.asarray(fitted.get_forecast(horizon).var_pred_mean),rtol=1e-6,atol=1e-7)
    return result


def condition_gaussian(mean,covariance,indices,values,noise):
    """Condition jointly, retaining cross-month covariance and noisy observations."""
    mean=np.asarray(mean,float);covariance=np.asarray(covariance,float)
    indices=np.asarray(indices,int);values=np.asarray(values,float)
    innovation_cov=covariance[np.ix_(indices,indices)]+np.asarray(noise,float)
    gain=np.linalg.solve(innovation_cov,covariance[indices,:]).T
    posterior_mean=mean+gain@(values-mean[indices])
    posterior_cov=covariance-gain@covariance[indices,:]
    posterior_cov=(posterior_cov+posterior_cov.T)/2
    if np.linalg.eigvalsh(posterior_cov).min() < -1e-6:
        raise ValueError('Non-positive posterior forecast covariance')
    return posterior_mean,posterior_cov


def forecast_table(index,mean,covariance):
    std=np.sqrt(np.maximum(np.diag(covariance),0))
    return pd.DataFrame(dict(period=index.astype(str),mean=mean,
                            lower80=mean-norm.ppf(.9)*std,upper80=mean+norm.ppf(.9)*std,
                            lower95=mean-norm.ppf(.975)*std,upper95=mean+norm.ppf(.975)*std))


class MonthlyData:
    def __init__(self,data=None,snapshot=None):
        self.root=Path(data or ROOT/'data')
        self.raw=read_periods(self.root/'inputs/core_monthly.csv','M')
        self.quarterly=DataStore(self.root)
        base=pd.read_csv(self.root/'inputs/policy_decisions.csv')
        base=base.rename(columns={'rate':'IR_POLICY'})
        base['date']=pd.to_datetime(base.date)
        self.decisions=base[['date','IR_POLICY']].copy()
        import json
        self.snapshot=json.loads(Path(snapshot or SNAPSHOT).read_text())
        self.as_of=pd.Timestamp(self.snapshot['as_of'])
        year=None;extra=[]
        for row in self.snapshot['rows']:
            if len(row)==1:
                found=re.search(r'\b(\d{4})\b',row[0])
                if found:year=int(found[1])
            elif len(row)>=2 and year:
                match=re.search(r'(\d{2})\.(\d{2})',row[0])
                if match:
                    date=pd.Timestamp(year,int(match[2]),int(match[1]))
                    try:rate=float(row[1].replace(',','.').replace(' ',''))
                    except ValueError:continue
                    if date<=self.as_of:extra.append(dict(date=date,IR_POLICY=rate))
        self.decisions=pd.concat([self.decisions,pd.DataFrame(extra)],ignore_index=True)
        self.decisions=self.decisions.loc[self.decisions.date<=self.as_of].drop_duplicates('date',keep='last').sort_values('date')
        dates=pd.date_range('1999-01-01',self.as_of,freq='D')
        self.daily=self.decisions.set_index('date').IR_POLICY.reindex(dates,method='ffill')
        self.daily.index.name='date'
        self.means=self.daily.groupby(self.daily.index.to_period('M')).mean()
        self.completed_last=min(self.raw.index.max(),self.as_of.to_period('M')-1)
        self.cache={}

    def frame(self,complete_end):
        end=pd.Period(complete_end,freq='M')
        if end>self.completed_last:raise ValueError('Month not fully observed in macro inputs')
        if end in self.cache:return self.cache[end].copy()
        monthly=self.raw.loc[:end].copy()
        index=monthly.loc['2010-01':].index
        frame=pd.DataFrame(index=index)
        frame['ir_pol']=self.means.reindex(index)
        frame['ir_ea']=monthly.IR_ECB_DFR_CALENDAR_AVG.reindex(index)
        frame['l_er']=100*np.log(monthly.ER_UAH_EUR_AVG.reindex(index))
        prices=monthly.CPI_UA_INDEX_NSA_2021_100.dropna()
        logged=np.log(prices.to_numpy())
        adjusted=STL(logged,period=12,seasonal=13,trend=21,low_pass=13,robust=True).fit()
        frame['l_cpi']=pd.Series(100*(logged-adjusted.seasonal-(logged[0]-adjusted.seasonal[0])),index=prices.index).reindex(index)
        if frame.isna().any().any():raise ValueError('Missing monthly core value; no automatic interpolation')
        self.cache[end]=frame.copy()
        return frame

    def partial_measurement(self,as_of):
        """Historical prefix/spot errors provide a noisy full-month observation.

        Both current prefix mean and current effective rate are known. Their
        correlated errors relative to the FINAL monthly average are estimated
        from COMPLETED past months only. Independence from model innovations is
        an explicit measurement approximation, not a true real-time vintage.
        """
        date=pd.Timestamp(as_of)
        if date>self.as_of:raise ValueError('Requested information beyond saved source snapshot')
        month=date.to_period('M');sample=self.daily.loc['2010-01-01':date]
        history=[]
        for period,values in sample.groupby(sample.index.to_period('M')):
            if period>=month:continue
            prefix=values.loc[values.index.day<=date.day]
            history.append([prefix.mean()-values.mean(),prefix.iloc[-1]-values.mean()])
        errors=np.asarray(history,float)
        bias=errors.mean(axis=0)
        noise=np.cov(errors,rowvar=False,ddof=1)+np.eye(2)*1e-6
        current=sample.loc[str(month)]
        observed=np.array([current.mean(),current.iloc[-1]])
        return dict(month=month,as_of=str(date.date()),observed=observed,bias=bias,noise=noise,
                    adjusted=observed-bias,n=len(errors),known_days=len(current),days_in_month=month.days_in_month)


def monthly_standalone(name):
    import importlib
    parser=argparse.ArgumentParser(description='Forecast next-month calendar-day average NBU policy rate')
    parser.add_argument('--as-of',default='2026-10-22')
    parser.add_argument('--output',type=Path,default=ROOT/'results/2026-10-22-monthly'/name)
    args=parser.parse_args()
    data=MonthlyData();date=pd.Timestamp(args.as_of)
    module=importlib.import_module('monthly_'+name)
    result,metadata=module.forecast(data,date)
    args.output.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output/'forecast.csv',index=False)
    write_json(args.output/'metadata.json',metadata)
    print(result.to_string(index=False))
