"""Monthly SARIMA uses period 12, not the quarterly period 4."""
import pandas as pd
from sarima import SARIMAModel
from monthly_common import sarimax_joint_covariance,condition_gaussian,forecast_table,monthly_standalone


def forecast(data,as_of,horizon=1):
    if horizon not in (1,2,3):raise ValueError('Monthly horizon must be 1, 2 or 3')
    month=pd.Timestamp(as_of).to_period('M')
    frame=data.frame(month-1)
    steps=horizon+1
    model=SARIMAModel();prior,metadata=model.fit_forecast(frame,steps)
    covariance=sarimax_joint_covariance(model.result,steps)
    measurement=data.partial_measurement(as_of)
    mean,covariance=condition_gaussian(prior['mean'],covariance,[0,0],measurement['adjusted'],measurement['noise'])
    table=forecast_table(pd.period_range(month,periods=steps,freq='M'),mean,covariance)
    table['status']=['current_month_nowcast','next_month_forecast']+['future_month_forecast']*(horizon-1)
    metadata.update(frequency='M',annual_seasonal_period=12,target='calendar_day_average_policy_rate',
                    as_of=str(pd.Timestamp(as_of).date()),training_end=str(month-1),next_month=str(month+1),
                    partial_measurement_history=measurement['n'],known_days=measurement['known_days'],
                    measurement='prefix mean and spot rate; correlated historical errors versus final month mean')
    return table,metadata


if __name__=='__main__':monthly_standalone('sarima')
