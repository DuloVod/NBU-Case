"""Meaningful invariants: RE equations, information leakage, and metrics."""
import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"models"))
import numpy as np
import pandas as pd
from common import DataStore, fit_model
from qpm import QPM
from run_all import summarize


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = DataStore()

    def test_qpm_rational_expectations_equations_and_shocks(self):
        model = QPM()
        p,b = model.transition,model.impact
        np.testing.assert_allclose(model.ap@p@p+model.a0@p+model.am,0,atol=1e-8)
        np.testing.assert_allclose((model.a0+model.ap@p)@b,model.e,atol=1e-8)
        np.testing.assert_allclose(model.a0@model.intercept+model.ap@(p@model.intercept+model.intercept),model.c,atol=1e-8)
        self.assertEqual(model.solver_info['stable_generalized_roots'],model.n)

    def test_monetary_tightening_reduces_activity(self):
        model = QPM()
        shock = model.impact[:,model.shocks.index('res_ir_tlr')]
        self.assertGreater(shock[model.ix['ir_pol']],0)
        self.assertLess((model.transition@shock)[model.ix['l_gdp_gap']],0)

    def test_future_values_cannot_change_past_information_set(self):
        origin = pd.Period('2019Q4',freq='Q')
        expected = self.data.frame(origin)
        altered = DataStore()
        altered.monthly.loc[altered.monthly.index>origin.asfreq('M'),:] *= 9
        altered.levels.loc[altered.levels.index>origin,:] *= 7
        # The contemporary unreleased GDP must also be unavailable.
        altered.levels.loc[origin,['gdp','gdp_ea']] *= 5
        actual = altered.frame(origin)
        pd.testing.assert_frame_equal(expected,actual)
        self.assertTrue(actual.iloc[-1][['l_gdp','l_gdp_ea']].isna().all())

    def test_seasonal_adjustment_uses_each_fold_not_full_sample(self):
        short = self.data.frame('2019Q4')
        long = self.data.frame(self.data.latest)
        difference = short.l_cpi.iloc[-1]-long.loc[short.index[-1],'l_cpi']
        self.assertGreater(abs(difference),1e-5)

    def test_all_latest_forecasts_finite_and_ordered(self):
        frame = self.data.frame(self.data.latest)
        for name in ['qpm','arima','sarima','var']:
            with self.subTest(model=name):
                model,forecast,_ = fit_model(name,frame,4)
                self.assertEqual(forecast.period.iloc[0],'2026Q4')
                self.assertEqual(len(forecast),4)
                if name=='qpm':
                    path = model.paths
                    np.testing.assert_allclose(path.l_rer,path.l_er-path.l_cpi+path.l_cpi_ea,atol=1e-6)
                    self.assertGreater(np.linalg.eigvalsh(model.last_covariance).min(),-1e-7)
                    previous = model.filtered.iloc[-1].to_numpy()
                    for current in path.to_numpy():
                        expected = model.transition@current+model.intercept
                        np.testing.assert_allclose(model.am@previous+model.a0@current+model.ap@expected,model.c,atol=1e-6)
                        previous=current

    def test_metrics_independently_known_example(self):
        frame = pd.DataFrame(dict(model=['M']*2,horizon=[1]*2,prediction=[12.,11.],actual=[10.,10.],
                                  baseline=[13.,13.],lower80=[9.,9.],upper80=[13.,13.],lower95=[8.,8.],upper95=[14.,14.]))
        row = summarize(frame).iloc[0]
        self.assertAlmostEqual(row.MAE_pp,1.5)
        self.assertAlmostEqual(row.RMSE_pp,np.sqrt(2.5))
        self.assertAlmostEqual(row.MAE_ratio_to_no_change,.5)
        self.assertAlmostEqual(row.coverage80,1.)

    def test_var_integrates_changes_and_cross_horizon_covariance(self):
        from var import VARModel
        from scipy.stats import norm
        frame = self.data.frame('2026Q2',gdp_lag=0)
        model = VARModel()
        forecast,_ = model.fit_forecast(frame,4)
        features = model.features(frame).dropna()
        delta = model.result.forecast(features.to_numpy()[-model.result.k_ar:],steps=4)
        np.testing.assert_allclose(forecast['mean'],frame.ir_pol.iloc[-1]+delta[:,0].cumsum(),atol=1e-9)
        impulse = model.result.ma_rep(maxn=4)
        sigma = np.asarray(model.result.sigma_u)
        cumulative_weights = impulse[:,0,:].cumsum(axis=0)
        variance = np.cumsum([weight@sigma@weight for weight in cumulative_weights[:4]])
        np.testing.assert_allclose((forecast.upper95-forecast['mean'])/norm.ppf(.975),np.sqrt(variance),atol=1e-9)


if __name__ == '__main__': unittest.main(verbosity=2)
