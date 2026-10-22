"""Independent checks of CPI units, temporal isolation and paired scoring."""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'models'))
import policy_nowcast as nowcast


def synthetic_core():
    index=pd.period_range('2010-01','2026-09',freq='M')
    mom=np.full(len(index),.2)
    level=100*np.cumprod(1+mom/100)
    return pd.DataFrame({
        'CPI_UA_MOM':mom,
        'CORE_CPI_UA_MOM':mom*.8,
        'CPI_UA_INDEX_NSA_2021_100':level,
        'CPI_UA_YOY_PUBLISHED':100*((1+.2/100)**12-1),
        'ER_UAH_EUR_AVG':30*np.exp(np.arange(len(index))*.001),
    },index=index)


class PolicyNowcastTest(unittest.TestCase):
    def test_selection_split_embargoes_unfinished_validation_targets(self):
        self.assertEqual(nowcast.evaluation_split('2024-07','2024-08'),'validation')
        self.assertEqual(nowcast.evaluation_split('2024-07','2024-09'),'embargo')
        self.assertEqual(nowcast.evaluation_split('2024-08','2024-09'),'audit')
        for origin in pd.period_range('2024-05','2024-10',freq='M'):
            for horizon in range(1,5):
                target=origin+horizon
                label=nowcast.evaluation_split(origin,target)
                if label=='validation':self.assertLessEqual(target,pd.Period('2024-08',freq='M'))
                if label=='audit':self.assertGreaterEqual(origin,pd.Period('2024-08',freq='M'))

    def test_exact_compounding_and_each_annual_base(self):
        core=synthetic_core();origin=pd.Period('2026-05',freq='M')
        ordinary_monthly=np.array([2.,-1.,.5,0.])
        log_monthly=100*np.log1p(ordinary_monthly/100)
        rows=nowcast.to_headline_yoy(core,origin,log_monthly)
        price=float(core.loc[origin,'CPI_UA_INDEX_NSA_2021_100'])
        for h,row in enumerate(rows,1):
            price*=1+ordinary_monthly[h-1]/100
            lag=origin+h-12
            expected=100*(price/core.loc[lag,'CPI_UA_INDEX_NSA_2021_100']-1)
            self.assertAlmostEqual(row['headline_yoy_pct'],expected,places=10)
            self.assertAlmostEqual(row['headline_mom_pct'],ordinary_monthly[h-1],places=10)
            self.assertEqual(row['period'],str(origin+h))

    def test_future_prices_and_renormalization_do_not_change_yoy(self):
        core=synthetic_core();origin=pd.Period('2024-06',freq='M')
        changes=np.array([.3,.2,-.4,.1])
        expected=nowcast.to_headline_yoy(core,origin,changes)
        changed=core.copy()
        changed.loc[changed.index>origin,'CPI_UA_INDEX_NSA_2021_100']=1e9
        self.assertEqual(expected,nowcast.to_headline_yoy(changed,origin,changes))
        normalized=core.copy()
        normalized['CPI_UA_INDEX_NSA_2021_100']*=13.7
        actual=nowcast.to_headline_yoy(normalized,origin,changes)
        np.testing.assert_allclose([r['headline_yoy_pct'] for r in actual],
                                   [r['headline_yoy_pct'] for r in expected],atol=1e-10,rtol=0)

    def test_rejects_unknown_annual_price_base(self):
        core=synthetic_core().loc['2026-01':]
        with self.assertRaisesRegex(ValueError,'annual price base'):
            nowcast.to_headline_yoy(core,'2026-05',[.1])

    def test_features_have_one_month_pressure_lag_and_no_future_tail_dependence(self):
        core=synthetic_core();idx=core.index
        auxiliary=pd.DataFrame({'brent':60*np.exp(np.arange(len(idx))*.002)},index=idx)
        policy=pd.DataFrame({'gross_reserves_usd_mn':1000+np.arange(len(idx))*10.,
                             'fx_net_purchases_usd_mn':-np.arange(len(idx),dtype=float)},index=idx)
        origin=pd.Period('2024-06',freq='M')
        def sources(path,frequency):
            name=Path(path).name
            return {'core_monthly.csv':core,'auxiliary_monthly.csv':auxiliary,
                    'policy.csv':policy}[name].copy()
        with tempfile.TemporaryDirectory() as tmp:
            filename=Path(tmp)/'policy.csv';filename.touch()
            with patch.object(nowcast,'read_periods',side_effect=sources):
                _,original=nowcast.feature_data(filename)
            expected=-policy.loc[origin-1,'fx_net_purchases_usd_mn']/policy.loc[origin-2,'gross_reserves_usd_mn']*100
            self.assertAlmostEqual(original.loc[origin,'fx_pressure_lag1'],expected,places=12)
            # The pressure in row m must not depend on month m's policy flows.
            policy.loc[origin:,'fx_net_purchases_usd_mn']=999999.
            policy.loc[origin-1:,'gross_reserves_usd_mn']=999999.
            core.loc[core.index>origin,['CPI_UA_MOM','ER_UAH_EUR_AVG']]=888888.
            auxiliary.loc[auxiliary.index>origin,'brent']=777777.
            with patch.object(nowcast,'read_periods',side_effect=sources):
                _,changed=nowcast.feature_data(filename)
            pd.testing.assert_frame_equal(original.loc[:origin],changed.loc[:origin])

    def test_paired_eligibility_and_prefix_calls(self):
        core=synthetic_core()
        features=pd.DataFrame({'inflation':100*np.log1p(core.CPI_UA_MOM/100),
                               'fx_pressure_lag1':0.},index=core.index)
        call_origins=[]
        def fake_forecast(data,name,horizon):
            origin=data.index[-1]
            self.assertTrue(data.index.equals(features.loc[:origin].index))
            call_origins.append((name,origin))
            if name=='ARIMA' and origin.month%3==0 and origin<pd.Period('2026-09',freq='M'):
                raise RuntimeError('deliberate eligibility hole')
            # Too few validation folds makes this model ineligible there.
            if name=='VAR_POLICY' and origin<=pd.Period('2024-08',freq='M') and origin not in pd.period_range('2023-01','2023-03',freq='M'):
                raise ValueError('deliberate short policy history')
            offset={'SEASONAL_NAIVE':0.,'ARIMA':.1,'SARIMA':.2,'VAR':.3,'VAR_POLICY':.4}[name]
            return np.full(horizon,100*np.log1p(.2/100)+offset),{'synthetic_test':True}
        with tempfile.TemporaryDirectory() as out:
            with patch.object(nowcast,'feature_data',return_value=(core,features)), \
                 patch.object(nowcast,'forecast_log_monthly',side_effect=fake_forecast), \
                 contextlib.redirect_stdout(io.StringIO()):
                forecasts,metrics=nowcast.run_nowcasts(out)
            backtest=pd.read_csv(Path(out)/'cpi_backtest.csv')
            for (split,horizon),group in metrics.groupby(['split','horizon']):
                self.assertEqual(group.n.nunique(),1)
                history=backtest.loc[(backtest.split==split)&(backtest.horizon==horizon)]
                eligible=list(group.model)
                common=set.intersection(*(set(history.loc[history.model==name,'origin']) for name in eligible))
                for row in group.itertuples():
                    errors=history.loc[(history.model==row.model)&history.origin.isin(common),'error']
                    self.assertEqual(row.n,len(errors))
                    self.assertAlmostEqual(row.MAE_pp,float(errors.abs().mean()),places=10)
            self.assertNotIn('VAR_POLICY',set(metrics.loc[metrics.split=='validation','model']))
            self.assertNotIn('embargo',set(metrics.split))
            self.assertTrue((pd.PeriodIndex(backtest.loc[backtest.split=='validation','period'],freq='M')<=pd.Period('2024-08',freq='M')).all())
            self.assertTrue((pd.PeriodIndex(backtest.loc[backtest.split=='audit','origin'],freq='M')>=pd.Period('2024-08',freq='M')).all())
            self.assertEqual(set(forecasts.loc[forecasts.selected_by_validation,'model']),{'SEASONAL_NAIVE'})
            self.assertTrue(all(origin<=pd.Period('2026-09',freq='M') for _,origin in call_origins))
            # Change audit outcomes enough to favor the largest forecast;
            # validation selection and every current forecast must remain fixed.
            changed=core.copy()
            changed.loc[changed.index>pd.Period('2024-08',freq='M'),'CPI_UA_YOY_PUBLISHED']+=50.
            with patch.object(nowcast,'feature_data',return_value=(changed,features)), \
                 patch.object(nowcast,'forecast_log_monthly',side_effect=fake_forecast), \
                 contextlib.redirect_stdout(io.StringIO()):
                revised,revised_metrics=nowcast.run_nowcasts(Path(out)/'changed_audit')
            pd.testing.assert_frame_equal(forecasts,revised)
            best_audit=revised_metrics.loc[(revised_metrics.split=='audit')&(revised_metrics.horizon==1)].sort_values('MAE_pp').iloc[0].model
            self.assertEqual(best_audit,'VAR_POLICY')

    def test_var_policy_requires_contiguous_complete_training_segment(self):
        idx=pd.period_range('2015-01','2026-09',freq='M')
        rng=np.random.default_rng(713)
        data=pd.DataFrame(rng.normal(size=(len(idx),5)),index=idx,
            columns=['inflation','core_inflation','fx_change','brent_change','fx_pressure_lag1'])
        # Total complete observations exceed 72, but fewer than 72 after a hole.
        data.loc['2023-01','fx_pressure_lag1']=np.nan
        with self.assertRaisesRegex(ValueError,'final complete segment shorter'):
            nowcast.forecast_log_monthly(data,'VAR_POLICY',4)


if __name__=='__main__':unittest.main()
