"""Seasonal ARIMA challenger: annual period = 4 quarterly observations.

Annual seasonality of a policy rate is a hypothesis to test, not an assumption
that this model must outperform a nonseasonal ARIMA or a no-change baseline.
"""
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


class SARIMAModel:
    name = 'SARIMA'

    def fit_forecast(self, frame, horizon=9):
        series = frame.ir_pol.astype(float)
        candidates,successes = [],[]
        orders = [(1,0,0),(1,1,0),(0,1,1),(1,1,1)]
        seasons = [(1,0,0,4),(0,0,1,4),(0,1,1,4)]
        for order in orders:
            for season in seasons:
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        result = SARIMAX(series,order=order,seasonal_order=season,
                                         trend='c' if order[1]+season[1]==0 else 'n',
                                         enforce_stationarity=True,enforce_invertibility=True).fit(disp=False,maxiter=300)
                    converged = bool(result.mle_retvals.get('converged',False))
                    k,n = len(result.params),result.nobs-result.loglikelihood_burn
                    aicc = float(result.aic+2*k*(k+1)/(n-k-1)) if n>k+1 else np.inf
                    candidates.append(dict(order=list(order),seasonal_order=list(season),aicc=aicc,
                                           converged=converged,warnings=sorted(set(str(w.message) for w in caught))))
                    if converged and np.isfinite(aicc): successes.append((aicc,result,order,season))
                except (ValueError,np.linalg.LinAlgError) as error:
                    candidates.append(dict(order=list(order),seasonal_order=list(season),error=str(error)))
        if not successes: raise RuntimeError('No converged SARIMA candidate')
        _,self.result,order,season = min(successes,key=lambda x:x[0])
        predicted = self.result.get_forecast(horizon)
        ci80,ci95 = np.asarray(predicted.conf_int(alpha=.2)),np.asarray(predicted.conf_int(alpha=.05))
        forecast = pd.DataFrame(dict(period=[str(x) for x in pd.period_range(frame.index[-1]+1,periods=horizon,freq='Q')],
                                     mean=np.asarray(predicted.predicted_mean),lower80=ci80[:,0],upper80=ci80[:,1],
                                     lower95=ci95[:,0],upper95=ci95[:,1],horizon=np.arange(1,horizon+1)))
        return forecast,dict(order=list(order),seasonal_order=list(season),criterion='AICc on training data only',
                             parameters={str(k):float(v) for k,v in self.result.params.items()},
                             candidates=candidates,intervals='Gaussian conditional on selected specification')


if __name__ == '__main__':
    from common import standalone
    standalone('sarima')
