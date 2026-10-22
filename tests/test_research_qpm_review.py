"""Independent structural checks for anticipated QPM policy experiments."""
import sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'models'))
from common import DataStore
from research_qpm import ResearchQPM,anticipated_path,model_vector

class QPMResearchReviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame=DataStore().frame('2026Q3',gdp_lag=1)
        cls.model=ResearchQPM();cls.model.fit_forecast(cls.frame,8)
        cls.initial=cls.model.filtered.iloc[-1].to_numpy()

    def test_yoy_target_uses_correct_four_quarter_lag(self):
        class Dummy:pass
        dummy=Dummy();dummy.paths=pd.DataFrame([dict(ir_pol=10.,l_cpi=12.,l_er=30.,l_gdp=40.)])
        frame=pd.DataFrame(dict(l_cpi=np.arange(1.,9.)))
        # Target is quarter9; annual lag is quarter5 (frame[-4]), not quarter4.
        self.assertEqual(model_vector(dummy,frame)[1],7.)

    def test_stack_equals_recursive_without_news(self):
        states,diag=anticipated_path(self.model,self.initial,8)
        np.testing.assert_allclose(states,self.model.paths.to_numpy(),atol=2e-6)
        self.assertLess(diag['structural_residual'],1e-7)

    def test_anticipated_shock_satisfies_direct_structural_equations(self):
        model=self.model;horizon=8;at=3;shock_name='res_dl_cpi';shock_size=2.
        states,_=anticipated_path(model,self.initial,horizon,shocks={(at,shock_name):shock_size})
        previous=self.initial
        for t,state in enumerate(states):
            following=states[t+1] if t+1<horizon else model.transition@state+model.intercept
            eps=np.zeros(model.k)
            if t==at:eps[model.shocks.index(shock_name)]=shock_size
            residual=model.am@previous+model.a0@state+model.ap@following-model.c-model.e@eps
            np.testing.assert_allclose(residual,0.,atol=1e-7)
            previous=state
        # Future inflation news changes earlier forward-looking variables.
        self.assertGreater(abs(states[0,model.ix['l_er']]-model.paths.iloc[0].l_er),1e-7)

    def test_constraint_shock_sign_and_exact_timing(self):
        model=self.model;constraint={0:(29*16.+63*16.5)/92,2:16.}
        states,diag=anticipated_path(model,self.initial,8,constraints=constraint)
        for t,target in constraint.items():self.assertAlmostEqual(states[t,model.ix['ir_pol']],target,places=9)
        previous=self.initial
        for t,state in enumerate(states):
            following=states[t+1] if t<7 else model.transition@state+model.intercept
            eps=np.zeros(model.k)
            if str(t) in diag['required_rule_shocks']:
                eps[model.shocks.index('res_ir_tlr')]=diag['required_rule_shocks'][str(t)]
            residual=model.am@previous+model.a0@state+model.ap@following-model.c-model.e@eps
            np.testing.assert_allclose(residual,0.,atol=1e-7)
            previous=state
        self.assertAlmostEqual(sum([29,63]),pd.Period('2026Q4','Q').end_time.day+30+31)

if __name__=='__main__':unittest.main()
