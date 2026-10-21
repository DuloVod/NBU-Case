"""Small stationary-transform VAR, BIC lags, conditional ragged-edge nowcast."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.api import VAR


class VARModel:
    name = 'VAR'

    def features(self, frame):
        return pd.DataFrame(dict(d_ir_pol=frame.ir_pol.diff(),
                                 inflation_qoq=4*frame.l_cpi.diff(),
                                 growth_qoq=4*frame.l_gdp.diff(),
                                 depreciation_qoq=4*frame.l_er.diff()),index=frame.index)

    def fit_forecast(self, frame, horizon=9):
        features = self.features(frame)
        train = features.dropna()
        if not train.index.equals(pd.period_range(train.index[0],train.index[-1],freq='Q')):
            raise ValueError('Internal training gaps must not be silently deleted')
        candidates,successes = [],[]
        for lag in [1,2]:
            # With about 30--65 quarters, large VARs are severely overparameterized.
            # Require five usable observations per coefficient per equation.
            if len(train)-lag < 5*(1+len(features.columns)*lag):
                candidates.append(dict(lag=lag,excluded='fewer than 5 observations per coefficient'))
                continue
            result = VAR(train).fit(lag,trend='c')
            stable = bool(result.is_stable())
            candidates.append(dict(lag=lag,bic=float(result.bic),stable=stable))
            successes.append((not stable,result.bic,result))
        # Prefer stable fits. If neither lag is stable, retain the finite-horizon
        # BIC fit and expose instability; never silently omit a difficult fold.
        unstable,_,self.result = min(successes,key=lambda x:(x[0],x[1]))
        k,p = len(features.columns),self.result.k_ar
        history = train.to_numpy().tolist()
        covariance = np.asarray(self.result.sigma_u)
        conditioning,nowcasts = [],[]
        # Build lag-state covariance and condition on all known current variables.
        # Augment the VAR lag state with the policy-rate LEVEL. Its covariance
        # carries all cross-horizon covariance of accumulated rate changes.
        dimension = k*p+1
        state_cov = np.zeros((dimension,dimension))
        transition = np.zeros_like(state_cov)
        transition[:k,:k*p] = np.concatenate(self.result.coefs,axis=1)
        if p>1: transition[k:k*p,:k*(p-1)] = np.eye(k*(p-1))
        transition[-1,:] = transition[0,:]; transition[-1,-1] = 1
        noise = np.zeros_like(state_cov); noise[:k,:k] = covariance
        noise[-1,:k] = covariance[0,:]; noise[:k,-1] = covariance[:,0]
        noise[-1,-1] = covariance[0,0]
        state = np.r_[np.concatenate([history[-j] for j in range(1,p+1)]),frame.loc[train.index[-1],'ir_pol']]
        intercept = np.r_[self.result.intercept,np.zeros(k*(p-1)),self.result.intercept[0]]
        for period in pd.period_range(train.index[-1]+1,frame.index[-1],freq='Q'):
            state = transition@state+intercept
            state_cov = transition@state_cov@transition.T+noise
            row = features.loc[period]
            known = np.flatnonzero(row.notna().to_numpy())
            missing = np.flatnonzero(row.isna().to_numpy())
            if len(known):
                gain = np.linalg.solve(state_cov[np.ix_(known,known)],state_cov[known,:]).T
                state = state+gain@(row.iloc[known].to_numpy()-state[known])
                state_cov = state_cov-gain@state_cov[known,:]
                state_cov = (state_cov+state_cov.T)/2
            conditioning.append(dict(period=str(period),observed=[features.columns[i] for i in known],
                                     nowcasted=[features.columns[i] for i in missing]))
            nowcasts.append(dict(period=str(period),ir_pol=state[-1],**dict(zip(features.columns,state[:k]))))
        rows = []
        for step,period in enumerate(pd.period_range(frame.index[-1]+1,periods=horizon,freq='Q'),1):
            state = transition@state+intercept
            state_cov = transition@state_cov@transition.T+noise
            mean,std = state[-1],np.sqrt(max(0,state_cov[-1,-1]))
            rows.append(dict(period=str(period),mean=mean,lower80=mean-norm.ppf(.9)*std,
                             upper80=mean+norm.ppf(.9)*std,lower95=mean-norm.ppf(.975)*std,
                             upper95=mean+norm.ppf(.975)*std,horizon=step))
        self.nowcasts = pd.DataFrame(nowcasts)
        return pd.DataFrame(rows),dict(lags=p,criterion='BIC on training data only, preferring stable candidates',
                                       stable=not unstable,policy_rate_transform='first difference; forecast integrated back to level',
                                       variables=list(features.columns),candidates=candidates,
                                       training_end=str(train.index[-1]),conditioning=conditioning,
                                       intervals='Gaussian, includes ragged-edge nowcast uncertainty; fixed parameters')


if __name__ == '__main__':
    from common import standalone
    standalone('var')
