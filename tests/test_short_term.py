"""Forecast horizon and ensemble information-set invariants."""
from pathlib import Path
import sys,unittest,importlib
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'models'))
from monthly_common import MonthlyData
from research_short_term import ensemble,BASES

class ShortTermTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=MonthlyData()

    def test_horizon_prefix_consistency_and_target_dates(self):
        for module in ['arima','sarima','var','qpm']:
            with self.subTest(module=module):
                model=importlib.import_module('monthly_'+module)
                short,_=model.forecast(self.data,'2026-10-22',1)
                long,meta=model.forecast(self.data,'2026-10-22',3)
                self.assertEqual(long.period.tolist(),['2026-10','2026-11','2026-12','2027-01'])
                np.testing.assert_allclose(short['mean'],long['mean'].iloc[:2],atol=1e-8)
                np.testing.assert_allclose(short[['lower80','upper80']],long[['lower80','upper80']].iloc[:2],atol=1e-8)
                if module=='qpm':self.assertEqual(meta['structural_frequency'],'Q')

    def test_ensemble_weights_ignore_unresolved_targets(self):
        origin=pd.Period('2024-01','M');history=[]
        for i,target in enumerate(pd.period_range('2020-01','2024-03',freq='M')):
            predictions={name:15.+j*.1 for j,name in enumerate(BASES)}
            history.append(dict(target=target,actual=15.2+i*.001,spot=15.,predictions=dict(predictions,BLEND_RAW=15.15)))
        current={name:16.+i*.1 for i,name in enumerate(BASES)}
        expected=ensemble(history,origin,current,16.)
        for row in history:
            if row['target']>=origin:row['actual']=99999.;row['predictions']['BLEND_RAW']=-99999.
        actual=ensemble(history,origin,current,16.)
        self.assertEqual(expected,actual)
        mean,weights,_,_=actual
        self.assertAlmostEqual(sum(weights.values()),1.)
        self.assertGreaterEqual(weights['NO_CHANGE'],.5)
        self.assertAlmostEqual(mean,weights['NO_CHANGE']*16+sum(weights[n]*current[n] for n in BASES))

if __name__=='__main__':unittest.main()
