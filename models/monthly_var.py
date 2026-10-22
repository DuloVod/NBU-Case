"""Monthly VAR: policy-rate changes, price/FX changes, and ECB rate changes."""
import pandas as pd
from var import VARModel
from monthly_common import condition_gaussian,forecast_table,monthly_standalone


class MonthlyVAR(VARModel):
    def __init__(self):super().__init__(lag_candidates=(1,2,3,6))

    def features(self,frame):
        return pd.DataFrame(dict(d_ir_pol=frame.ir_pol.diff(),inflation_month_annualized=12*frame.l_cpi.diff(),
                                 depreciation_month_annualized=12*frame.l_er.diff(),d_ir_ea=frame.ir_ea.diff()),index=frame.index)


def forecast(data,as_of,horizon=1):
    if horizon not in (1,2,3):raise ValueError('Monthly horizon must be 1, 2 or 3')
    month=pd.Timestamp(as_of).to_period('M')
    frame=data.frame(month-1)
    steps=horizon+1
    model=MonthlyVAR();prior,metadata=model.fit_forecast(frame,steps)
    measurement=data.partial_measurement(as_of)
    mean,covariance=condition_gaussian(prior['mean'],model.forecast_covariance,[0,0],
                                       measurement['adjusted'],measurement['noise'])
    table=forecast_table(pd.period_range(month,periods=steps,freq='M'),mean,covariance)
    table['status']=['current_month_nowcast','next_month_forecast']+['future_month_forecast']*(horizon-1)
    metadata.update(frequency='M',target='calendar_day_average_policy_rate',as_of=str(pd.Timestamp(as_of).date()),
                    next_month=str(month+1),known_days=measurement['known_days'],
                    gdp='Not interpolated to months; ECB rate used as the fourth monthly variable',
                    measurement='prefix mean and spot rate; correlated historical errors versus final month mean')
    return table,metadata


if __name__=='__main__':monthly_standalone('var')
