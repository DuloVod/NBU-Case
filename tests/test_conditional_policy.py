"""Checks that matter to interpretation of the conditional decision experiment."""
import sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'models'))
from conditional_policy import rank_robust
from policy_qpm import PolicyQPM,anticipated_path
from common import DataStore
from policy_decision import calendar_average
from research_qpm import enrich_path

class ConditionalDecisionTests(unittest.TestCase):
    def test_diagnostic_shocks_cannot_select_policy(self):
        rows=[]
        for name,tier,values in [('central','decision',[5.,4.,6.]),('pure','diagnostic',[100.,0.,1.])]:
            for r,v in zip([15.5,16.,16.5],values):
                rows.append(dict(tier=tier,initial_state='x',beta=.7,scenario=name,
                    output_weight=1,smoothing_weight=.1,horizon=8,rate=r,total_loss=v))
        detail,summary=rank_robust(pd.DataFrame(rows))
        self.assertEqual(set(detail.scenario),{'central'})
        self.assertEqual(float(summary.loc[summary.worst_regret_vs_hold.idxmin(),'rate']),16.)
        self.assertEqual(summary.contexts.tolist(),[1,1,1])

    def test_rate_effect_preserves_potential_and_has_disinflation_cost(self):
        model=PolicyQPM();frame=DataStore().frame('2026Q3',gdp_lag=1)
        model.fit_forecast(frame,16);initial=model.filtered.iloc[-1].to_numpy()
        shocks={(0,'res_dl_cpi'):2.}
        hold,_=anticipated_path(model,initial,16,constraints={0:16.},shocks=shocks)
        hike,_=anticipated_path(model,initial,16,
            constraints={0:calendar_average(16.5,16.5)},shocks=shocks)
        ix=model.ix
        np.testing.assert_allclose(hold[:,ix['l_gdp_tnd']],hike[:,ix['l_gdp_tnd']],atol=1e-8)
        a=enrich_path(model,frame,hold);b=enrich_path(model,frame,hike)
        self.assertLess(b.inflation_yoy_pct.iloc[3],a.inflation_yoy_pct.iloc[3])
        self.assertLess(b.l_gdp.iloc[2],a.l_gdp.iloc[2])

    def test_shock_provenance_is_same_across_candidate_rates(self):
        folder=ROOT/'results/2026-10-23-conditional-decision'
        if not folder.exists():self.skipTest('Standalone scenario run not yet saved')
        loss=pd.read_csv(folder/'losses.csv')
        self.assertEqual(loss.groupby(['initial_state','beta','scenario','output_weight',
                                      'smoothing_weight','horizon']).rate.nunique().unique().tolist(),[13])
        self.assertTrue((loss.regret>=-1e-9).all())
        checks=pd.read_csv(folder/'numerical_checks.csv')
        self.assertLess(checks[['structural_residual','constraint_residual','affine_max_error']].to_numpy().max(),1e-7)

if __name__=='__main__':unittest.main()
