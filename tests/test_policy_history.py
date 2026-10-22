"""Temporal isolation and interpretation checks for historical policy replay."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'models'))
from common import DataStore
from research_qpm import ResearchQPM
from policy_history import (HistoricalReplayRunner, run_historical_validation,
                            regime, benchmark_values, direct_equation_residual)


class RecordingStore(DataStore):
    def __init__(self):
        super().__init__()
        self.calls=[]
    def frame(self,origin,gdp_lag=1):
        self.calls.append((str(origin),gdp_lag))
        return super().frame(origin,gdp_lag)


class PolicyHistoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store=DataStore()

    def test_regime_is_target_quarter(self):
        self.assertEqual(regime('2021Q4'),'prewar_2015_2021')
        self.assertEqual(regime('2022Q1'),'war_2022_2023')
        self.assertEqual(regime('2023Q4'),'war_2022_2023')

    def test_predictions_materialized_before_future_label_retrieval(self):
        store=RecordingStore()
        class CheckedModel(ResearchQPM):
            def fit_forecast(self,frame,horizon):
                self_test.assertEqual(store.calls,[('2019Q4',1)])
                self_test.assertTrue(frame.loc['2019Q4',['l_gdp','l_gdp_ea']].isna().all())
                return super().fit_forecast(frame,horizon)
        self_test=self
        rows=HistoricalReplayRunner(store=store,model_factory=CheckedModel).forecasts(['2019Q4'])
        self.assertEqual(store.calls,[('2019Q4',1),('2020Q1',0)])
        self.assertEqual(len(rows),12)
        self.assertTrue(np.isfinite(rows.prediction).all())
        self.assertTrue((rows.latest_training_gdp=='2019Q3').all())

    def test_benchmark_uses_last_available_gdp(self):
        frame=self.store.frame('2019Q4',gdp_lag=1)
        self.assertAlmostEqual(benchmark_values(frame)['gdp_log_level'],frame.loc['2019Q3','l_gdp'])

    def test_replay_constraints_and_equations(self):
        runner=HistoricalReplayRunner(store=self.store)
        paths,differences,checks,fidelity=runner.replay('pandemic','2019Q4',horizon=8,control_quarters=2)
        paths=pd.DataFrame(paths); checks=pd.DataFrame(checks); probes=pd.DataFrame(fidelity)
        self.assertTrue((checks.direct_equation_residual<1e-7).all())
        self.assertTrue((checks.max_control_error<1e-8).all())
        actual=paths.loc[paths.policy=='actual_rate_path_ex_post']
        flat=paths.loc[paths.policy=='flat_last_quarter_average']
        np.testing.assert_allclose(actual.ir_pol.iloc[:2],self.store.levels.loc[['2020Q1','2020Q2'],'ir_pol'],atol=1e-9)
        np.testing.assert_allclose(flat.ir_pol.iloc[:2],self.store.levels.loc['2019Q4','ir_pol'],atol=1e-9)
        self.assertFalse(pd.DataFrame(differences).causal_success_claim.any())
        loss=probes.loc[probes.probe=='potential_level_loss_unit']
        np.testing.assert_allclose(loss.delta_gdp_log_level,-1.,atol=1e-7)
        np.testing.assert_allclose(loss.delta_inflation_yoy_log_pp,0.,atol=1e-7)

    def test_direct_audit_detects_broken_path(self):
        runner=HistoricalReplayRunner(store=self.store)
        model,frame=runner.fit('2019Q4',4)
        initial=model.filtered.iloc[-1].to_numpy()
        states,diag=runner.path_solver(model,initial,4)
        states=states.copy();states[0,model.ix['ir_pol']]+=1.
        self.assertGreater(direct_equation_residual(model,initial,states,diag),.1)

    def test_protocol_declares_ex_post_and_vintage_limits(self):
        with tempfile.TemporaryDirectory() as out:
            result=run_historical_validation(out,store=self.store,origins=['2021Q4'],episodes={})
            protocol=json.loads((Path(out)/'historical_protocol.json').read_text())
            self.assertEqual(protocol['forecast_rows'],12)
            self.assertIn('NOT strict real-time',protocol['vintage'])
            self.assertIn('EX POST',protocol['replay'])
            self.assertEqual(set(result['forecasts'].regime),{'war_2022_2023'})

    def test_corrected_factory_and_solver_keep_judgment_disabled(self):
        from policy_qpm import PolicyQPM,anticipated_path
        runner=HistoricalReplayRunner(store=self.store,
            model_factory=lambda settings:PolicyQPM(settings,gap_judgments={}),
            path_solver=anticipated_path)
        paths,_,checks,_=runner.replay('wartime','2023Q2',horizon=8,control_quarters=2)
        self.assertTrue(all(c['direct_equation_residual']<1e-7 for c in checks))
        self.assertEqual(runner.model_configurations[0]['gap_judgments'],{})
        self.assertTrue(runner.model_configurations[0]['exact_targets'])


if __name__=='__main__':unittest.main()
