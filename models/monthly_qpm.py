"""Quarterly structural QPM + explicit Gaussian monthly disaggregation bridge.

This is not a newly calibrated monthly structural QPM. The bridge preserves
the uncertain quarterly aggregate prior, then uses observed monthly rates and
the current-month prefix. No quarterly number is relabelled as November.
"""
import numpy as np
import pandas as pd
from qpm import QPM
from monthly_common import scalar_forecast_covariance,condition_gaussian,forecast_table,monthly_standalone


def disaggregate(quarter_means,quarter_covariance,last_month_rate,monthly_variance,months):
    n=len(months);nq=len(quarter_means)
    weights=np.zeros((nq,n))
    for i in range(nq):
        days=np.array([month.days_in_month for month in months[i*3:i*3+3]],float)
        weights[i,i*3:i*3+3]=days/days.sum()
    prior_mean=np.full(n,last_month_rate)
    prior_cov=monthly_variance*np.minimum.outer(np.arange(1,n+1),np.arange(1,n+1))
    gain=np.linalg.solve(weights@prior_cov@weights.T,weights@prior_cov).T
    mean=prior_mean+gain@(np.asarray(quarter_means)-weights@prior_mean)
    covariance=prior_cov-gain@weights@prior_cov+gain@quarter_covariance@gain.T
    covariance=(covariance+covariance.T)/2
    np.testing.assert_allclose(weights@mean,quarter_means,atol=1e-8)
    np.testing.assert_allclose(weights@covariance@weights.T,quarter_covariance,atol=1e-8)
    return mean,covariance,weights


def forecast(data,as_of,horizon=1):
    if horizon not in (1,2,3):raise ValueError('Monthly horizon must be 1, 2 or 3')
    month=pd.Timestamp(as_of).to_period('M');current_q=month.asfreq('Q')
    last_q=current_q-1;target=month+horizon
    # Fixed two-quarter bridge for all 1/2/3-month requests: a longer displayed
    # horizon must not revise the earlier marginal forecast.
    nq=max(2,target.asfreq('Q').ordinal-current_q.ordinal+1)
    frame=data.quarterly.frame(last_q,gdp_lag=1)
    model=QPM();quarter_forecast,metadata=model.fit_forecast(frame,nq)
    design=np.zeros(model.n);design[model.ix['ir_pol']]=1
    impact=model.impact*model.shock_std()
    quarter_cov=scalar_forecast_covariance(model.transition,impact@impact.T,model.last_covariance,design,nq)
    np.testing.assert_allclose(np.diag(quarter_cov),((quarter_forecast.upper95-quarter_forecast['mean'])/1.959963984540054)**2,atol=1e-8)
    months=pd.period_range(current_q.asfreq('M','start'),periods=3*nq,freq='M')
    historical=data.means.loc['2010-01':str(last_q.asfreq('M','end'))]
    variance=max(float(historical.diff().dropna().var()),1e-4)
    mean,covariance,weights=disaggregate(quarter_forecast['mean'].to_numpy(),quarter_cov,
                                       float(historical.iloc[-1]),variance,months)
    # Completed months in the current quarter are observed, not forecasts.
    known=[i for i,m in enumerate(months) if m<month]
    if known:
        mean,covariance=condition_gaussian(mean,covariance,known,[data.means.loc[months[i]] for i in known],np.eye(len(known))*1e-8)
    measurement=data.partial_measurement(as_of)
    current=months.get_loc(month)
    mean,covariance=condition_gaussian(mean,covariance,[current,current],measurement['adjusted'],measurement['noise'])
    table=forecast_table(months,mean,covariance)
    table=table.loc[table.period.isin(pd.period_range(month,target,freq='M').astype(str))].reset_index(drop=True)
    table['status']=['current_month_nowcast','next_month_forecast']+['future_month_forecast']*(horizon-1)
    metadata.update(frequency='M',structural_frequency='Q',model_label='QPM + monthly bridge',
                    as_of=str(pd.Timestamp(as_of).date()),target='calendar_day_average_policy_rate',
                    next_month=str(month+1),last_forecast_month=str(target),quarterly_history_end=str(last_q),
                    gdp_available_through=str(last_q-1),known_days=measurement['known_days'],
                    bridge='Gaussian random walk conditioned on uncertain QPM quarter-average prior',
                    monthly_innovation_variance=variance,quarter_prior=quarter_forecast.to_dict('records'),
                    measurement='prefix mean and spot rate; correlated historical errors versus final month mean')
    return table,metadata


if __name__=='__main__':monthly_standalone('qpm')
