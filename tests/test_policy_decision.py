"""Independent calendar, price bridge and policy-counterfactual checks."""
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'models'))
import policy_decision as decision
from common import DataStore
from policy_qpm import PolicyQPM,anticipated_path
from policy_history import direct_equation_residual


class PolicyDecisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store=DataStore();cls.frame=cls.store.frame('2026Q3',gdp_lag=1)
        cls.model=PolicyQPM();cls.model.fit_forecast(cls.frame,16)
        cls.initial=cls.model.filtered.iloc[-1].to_numpy()

    def test_calendar_average_matches_each_effective_day(self):
        days=pd.date_range('2026-10-01','2026-12-31',freq='D')
        for october,december in [(15.5,16.5),(16.,16.),(20.,14.)]:
            values=np.where(days<pd.Timestamp('2026-10-30'),16.,
                    np.where(days<pd.Timestamp('2026-12-18'),october,december))
            self.assertEqual(np.count_nonzero(days<pd.Timestamp('2026-10-30')),29)
            self.assertEqual(np.count_nonzero((days>=pd.Timestamp('2026-10-30'))&(days<pd.Timestamp('2026-12-18'))),49)
            self.assertEqual(np.count_nonzero(days>=pd.Timestamp('2026-12-18')),14)
            self.assertAlmostEqual(decision.calendar_average(october,december),float(values.mean()),places=12)

    def test_within_quarter_timing_is_not_identified_by_same_average(self):
        self.assertAlmostEqual(decision.calendar_average(16.,17.75),decision.calendar_average(16.5,16.),places=12)

    def test_bridge_removes_seasonal_component_and_preserves_historical_anchor(self):
        idx=pd.period_range('2010-01','2026-12',freq='M')
        number=np.arange(len(idx))
        seasonal=pd.Series(.02*np.sin(2*np.pi*number/12),index=idx)
        trend=pd.Series(np.exp(.003*number),index=idx)
        nsa=370*trend*np.exp(seasonal)
        monthly=pd.DataFrame({'CPI_UA_INDEX_NSA_2021_100':nsa},index=idx)
        historical=idx[idx<=pd.Period('2026-09',freq='M')]
        expected_q3=100*np.log((100*trend.loc['2026-07':'2026-09']).mean())
        fake_store=SimpleNamespace(monthly=monthly,
            frame=lambda origin:pd.DataFrame({'l_cpi':[expected_q3]},index=pd.PeriodIndex(['2026Q3'],freq='Q')))
        forecast=pd.DataFrame([
            dict(period=str(p),model='selected',selected_by_validation=True,
                 headline_mom_pct=100*(nsa.loc[p]/nsa.loc[p-1]-1),headline_yoy_pct=0.)
            for p in pd.period_range('2026-10','2026-12',freq='M')])
        def known_seasonality(values,**kwargs):
            # The seasonal fit must see September and no forecast observations.
            self.assertEqual(len(values),len(historical))
            return SimpleNamespace(fit=lambda:SimpleNamespace(seasonal=seasonal.loc[historical].to_numpy()))
        with patch.object(decision,'STL',side_effect=known_seasonality):
            target,metadata=decision.inflation_bridge(fake_store,forecast)
            monthly.loc['2026-10':,'CPI_UA_INDEX_NSA_2021_100']=1e15
            target_after,_=decision.inflation_bridge(fake_store,forecast)
        expected_sa=100*trend.loc['2026-10':'2026-12'].to_numpy()
        np.testing.assert_allclose([r['sa_index'] for r in metadata['months']],expected_sa,atol=1e-10,rtol=0)
        self.assertAlmostEqual(target,100*np.log(expected_sa.mean()),places=10)
        self.assertAlmostEqual(target,target_after,places=12)

    def test_real_data_bridge_matches_datastore_normalization(self):
        forecast=pd.DataFrame([dict(period=str(p),model='fixed_test',selected_by_validation=True,
            headline_mom_pct=.5,headline_yoy_pct=0.) for p in pd.period_range('2026-10','2026-12',freq='M')])
        target,info=decision.inflation_bridge(self.store,forecast)
        self.assertAlmostEqual(info['historical_q3_l_cpi'],self.frame.loc['2026Q3','l_cpi'],places=9)
        self.assertAlmostEqual(target,100*np.log(np.mean([r['sa_index'] for r in info['months']])),places=10)

    def test_bridge_shock_frozen_when_rate_control_changes(self):
        model=self.model;target=float(model.paths.iloc[0].l_cpi+1.)
        bridge,diagnostic=decision.fixed_bridge_shock(model,self.initial,target,16)
        self.assertEqual(diagnostic['n_constraints'],1)
        self.assertEqual(diagnostic['required_shocks'][0]['control_shock'],'res_dl_cpi')
        frozen=deepcopy(bridge)
        uncontrolled,diag=anticipated_path(model,self.initial,16,shocks=bridge)
        self.assertAlmostEqual(uncontrolled[0,model.ix['l_cpi']],target,places=8)
        prices=[]
        for rate in [15.,17.]:
            states,diag=anticipated_path(model,self.initial,16,constraints={0:rate},shocks=bridge)
            self.assertAlmostEqual(states[0,model.ix['ir_pol']],rate,places=9)
            self.assertLess(direct_equation_residual(model,self.initial,states,diag,bridge),1e-7)
            prices.append(states[0,model.ix['l_cpi']])
        self.assertEqual(bridge,frozen)
        self.assertGreater(abs(prices[0]-prices[1]),1e-4)

    def test_affine_control_paths_match_direct_structural_solutions(self):
        model=self.model;shocks={(0,'res_dl_cpi'):2.,(2,'res_dl_cpi'):1.}
        p0,_=anticipated_path(model,self.initial,16,constraints={0:16.},shocks=shocks)
        p1,_=anticipated_path(model,self.initial,16,constraints={0:17.},shocks=shocks)
        for rate in [14.,16.35,20.]:
            direct,diag=anticipated_path(model,self.initial,16,constraints={0:rate},shocks=shocks)
            affine=p0+(rate-16.)*(p1-p0)
            np.testing.assert_allclose(affine,direct,atol=1e-8,rtol=0)
            self.assertLess(direct_equation_residual(model,self.initial,direct,diag,shocks),1e-7)

    def test_loss_units_discount_and_horizon(self):
        path=pd.DataFrame({'inflation_yoy_pct':[6.,6.,6.,100.],
                           'l_gdp_gap':[2.,2.,2.,100.], 'ir_pol':[16.,16.,16.,100.]})
        loss=decision.policy_loss(path,16.,16.,.25,1.,3)
        discount=1+.99+.99**2
        self.assertAlmostEqual(loss['inflation_loss'],discount,places=12)
        self.assertAlmostEqual(loss['output_loss'],4*discount,places=12)
        self.assertEqual(loss['smoothing_loss'],0.)
        self.assertAlmostEqual(loss['total_loss'],2*discount,places=12)

    def test_smoothing_uses_announcement_changes_not_quarter_average(self):
        average=decision.calendar_average(16.5,16.25)
        path=pd.DataFrame({'inflation_yoy_pct':[5.,5.,5.], 'l_gdp_gap':[0.,0.,0.],
                           'ir_pol':[average,16.25,16.25]})
        loss=decision.policy_loss(path,16.5,16.25,1.,1.,3)
        expected=(16.5-16.)**2+.99*(16.25-16.5)**2
        self.assertAlmostEqual(loss['smoothing_loss'],expected,places=12)
        self.assertAlmostEqual(loss['total_loss'],expected,places=12)

    def test_shock_scale_only_prewar_and_shared_scenario_base(self):
        idx=pd.period_range('2014Q1','2026Q3',freq='Q')
        frame=pd.DataFrame({'res_l_gdp_gap':np.arange(len(idx),dtype=float),
                            'res_prem_uip_gap':2*np.arange(len(idx),dtype=float)},index=idx)
        model=SimpleNamespace(filtered_shocks=frame)
        expected=float(frame.loc['2015Q1':'2019Q4','res_l_gdp_gap'].std(ddof=1))
        bridge={(0,'res_dl_cpi'):4.}
        cases,scales=decision.scenario_shocks(model,bridge)
        self.assertAlmostEqual(scales['demand_conditional_sd'],expected,places=12)
        for name in ['satellite_bridge','persistent_cost','negative_demand','risk_premium_stress']:
            self.assertEqual(cases[name][(0,'res_dl_cpi')],4.)
        self.assertEqual(cases['persistent_cost'][(1,'res_dl_cpi')],2.)
        self.assertEqual(cases['persistent_cost'][(3,'res_dl_cpi')],.5)
        frame.loc['2020Q1':]=1e12
        self.assertAlmostEqual(decision.shock_scale(model,'res_l_gdp_gap'),expected,places=12)
        cases['negative_demand'][(0,'res_dl_cpi')]=99.
        self.assertEqual(bridge[(0,'res_dl_cpi')],4.)
        self.assertEqual(cases['satellite_bridge'][(0,'res_dl_cpi')],4.)


if __name__=='__main__':unittest.main()
