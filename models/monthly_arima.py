"""True monthly ARIMA, conditioned on noisy current-month rate information."""
import pandas as pd
from arima import ARIMAModel
from monthly_common import sarimax_joint_covariance,condition_gaussian,forecast_table,monthly_standalone


def forecast(data,as_of):
    month=pd.Timestamp(as_of).to_period('M')
    frame=data.frame(month-1)
    model=ARIMAModel();prior,metadata=model.fit_forecast(frame,2)
    covariance=sarimax_joint_covariance(model.result,2)
    measurement=data.partial_measurement(as_of)
    mean,covariance=condition_gaussian(prior['mean'],covariance,[0,0],measurement['adjusted'],measurement['noise'])
    table=forecast_table(pd.period_range(month,periods=2,freq='M'),mean,covariance)
    table['status']=['current_month_nowcast','next_month_forecast']
    metadata.update(frequency='M',target='calendar_day_average_policy_rate',training_end=str(month-1),
                    as_of=str(pd.Timestamp(as_of).date()),next_month=str(month+1),
                    partial_measurement_history=measurement['n'],known_days=measurement['known_days'],
                    measurement='prefix mean and spot rate; correlated historical errors versus final month mean')
    return table,metadata


if __name__=='__main__':monthly_standalone('arima')
