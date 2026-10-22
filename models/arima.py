"""Univariate ARIMA with per-training-window AICc specification selection."""
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel:
    name = 'ARIMA'

    def fit_forecast(self, frame, horizon=9):
        series = frame.ir_pol.astype(float)
        if series.isna().any(): raise ValueError('ARIMA cannot silently drop target gaps')
        candidates, successes = [], []
        for d in [0,1]:
            for p in range(3):
                for q in range(2):
                    order = (p,d,q)
                    try:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter('always')
                            result = ARIMA(series,order=order,trend='c' if d==0 else 'n',
                                           enforce_stationarity=True,enforce_invertibility=True).fit(
                                               method_kwargs={'maxiter':300})
                        converged = bool(result.mle_retvals.get('converged',False))
                        k,n = len(result.params),result.nobs-result.loglikelihood_burn
                        aicc = float(result.aic+2*k*(k+1)/(n-k-1)) if n>k+1 else np.inf
                        record = dict(order=list(order),aicc=aicc,converged=converged,
                                      warnings=sorted(set(str(w.message) for w in caught)))
                        candidates.append(record)
                        if converged and np.isfinite(aicc): successes.append((aicc,result,order))
                    except (ValueError,np.linalg.LinAlgError) as error:
                        candidates.append(dict(order=list(order),error=str(error)))
        if not successes: raise RuntimeError('No converged ARIMA candidate')
        _,self.result,order = min(successes,key=lambda x:x[0])
        predicted = self.result.get_forecast(horizon)
        ci80,ci95 = np.asarray(predicted.conf_int(alpha=.2)),np.asarray(predicted.conf_int(alpha=.05))
        forecast = pd.DataFrame(dict(period=[str(x) for x in pd.period_range(frame.index[-1]+1,periods=horizon,freq=frame.index.freq)],
                                     mean=np.asarray(predicted.predicted_mean),lower80=ci80[:,0],upper80=ci80[:,1],
                                     lower95=ci95[:,0],upper95=ci95[:,1],horizon=np.arange(1,horizon+1)))
        return forecast,dict(order=list(order),criterion='AICc on training data only',
                             parameters={str(k):float(v) for k,v in self.result.params.items()},
                             candidates=candidates,intervals='Gaussian conditional on selected specification')


if __name__ == '__main__':
    from common import standalone
    standalone('arima')
