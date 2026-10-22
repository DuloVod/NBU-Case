"""Monthly targets, no future observations, covariance, and bridge consistency."""
import unittest,sys,importlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'models'))
import numpy as np
import pandas as pd
from monthly_common import MonthlyData,scalar_forecast_covariance,condition_gaussian
from monthly_qpm import disaggregate


class MonthlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=MonthlyData()

    def test_target_is_november_not_quarter_and_october_not_an_actual(self):
        for name in ['qpm','arima','sarima','var']:
            with self.subTest(model=name):
                table,meta=importlib.import_module('monthly_'+name).forecast(self.data,'2026-10-22')
                self.assertEqual(table.period.tolist(),['2026-10','2026-11'])
                self.assertEqual(table.status.tolist(),['current_month_nowcast','next_month_forecast'])
                self.assertEqual(meta['target'],'calendar_day_average_policy_rate')
                self.assertTrue(np.isfinite(table.select_dtypes('number')).all().all())
                self.assertTrue((table.lower95<=table['mean']).all() and (table['mean']<=table.upper95).all())
                if name=='sarima':self.assertEqual(meta['seasonal_order'][-1],12)
                if name=='qpm':self.assertEqual(meta['structural_frequency'],'Q')

    def test_known_partial_month_is_not_completed_training_data(self):
        frame=self.data.frame('2026-09')
        self.assertEqual(str(frame.index[-1]),'2026-09')
        measurement=self.data.partial_measurement('2026-10-22')
        self.assertEqual(measurement['known_days'],22)
        self.assertEqual(measurement['days_in_month'],31)
        self.assertEqual(measurement['observed'][1],16.)
        self.assertEqual(measurement['n'],201)
        with self.assertRaises(ValueError):self.data.frame('2026-10')

    def test_future_daily_and_monthly_values_do_not_change_a_past_forecast(self):
        date=pd.Timestamp('2019-02-22');past=self.data.partial_measurement(date)
        other=MonthlyData()
        other.daily.loc[other.daily.index>date]*=3
        other.means=other.daily.groupby(other.daily.index.to_period('M')).mean()
        other.raw.loc[other.raw.index>date.to_period('M')-1,:]*=7
        other.quarterly.monthly.loc[other.quarterly.monthly.index>date.to_period('M'),:]*=9
        after=other.partial_measurement(date)
        for key in ['observed','bias','noise']:np.testing.assert_allclose(past[key],after[key],atol=1e-12)
        pd.testing.assert_frame_equal(self.data.frame('2019-01'),other.frame('2019-01'))
        for name in ['arima','qpm']:
            expected,_=importlib.import_module('monthly_'+name).forecast(self.data,date)
            actual,_=importlib.import_module('monthly_'+name).forecast(other,date)
            pd.testing.assert_frame_equal(expected,actual,rtol=1e-9,atol=1e-9)

    def test_quarterly_moments_preserved_before_monthly_news(self):
        months=pd.period_range('2026-10',periods=6,freq='M')
        quarter=np.array([15.,14.]);cov=np.array([[1.,.4],[.4,1.5]])
        mean,monthly_cov,weights=disaggregate(quarter,cov,16.,2.,months)
        np.testing.assert_allclose(weights@mean,quarter,atol=1e-10)
        np.testing.assert_allclose(weights@monthly_cov@weights.T,cov,atol=1e-10)
        self.assertGreater(np.linalg.eigvalsh(monthly_cov).min(),-1e-8)

    def test_joint_forecast_covariance_closed_form(self):
        cov=scalar_forecast_covariance(np.array([[.8]]),np.array([[1.]]),np.array([[0.]]),np.array([1.]),3)
        np.testing.assert_allclose(cov,[[1.,.8,.64],[.8,1.64,1.312],[.64,1.312,2.0496]],atol=1e-12)

    def test_correlated_partial_observations_conditioned_jointly(self):
        mean,cov=condition_gaussian([10.,10.],np.array([[1.,.8],[.8,2.]]),[0,0],[12.,12.],np.array([[1.,.5],[.5,1.]]))
        self.assertGreater(mean[1],10.)
        self.assertLess(cov[1,1],2.)
        self.assertGreater(np.linalg.eigvalsh(cov).min(),0.)


if __name__=='__main__':unittest.main(verbosity=2)
