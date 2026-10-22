"""Independent checks for the chronological calibration research."""
import sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'models'))
from monthly_common import MonthlyData
from research_monthly import information,eligible,ridge_prediction,optimal_shrink,interval

class MonthlyResearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=MonthlyData()

    def test_partial_month_label_is_excluded(self):
        origin=pd.Period('2024-04','M')
        rows=[dict(target=pd.Period(p,'M'),actual=1) for p in ('2024-03','2024-04','2024-05')]
        self.assertEqual([str(r['target']) for r in eligible(rows,origin)],['2024-03'])

    def test_features_ignore_future_data(self):
        origin=pd.Period('2023-04','M');before=information(self.data,origin)
        changed=MonthlyData();changed.raw.loc['2023-04':,:]*=100
        changed.daily.loc['2023-04-23':]=99
        after=information(changed,origin)
        for name in ('policy','macro'):np.testing.assert_allclose(before[name],after[name])
        self.assertEqual(before['spot'],after['spot'])

    def sample(self):
        rows=[]
        for i,p in enumerate(pd.period_range('2015-01','2019-01',freq='M')):
            rows.append(dict(origin=p,target=p+1,actual=10.+i/100,spot=10.,
                             policy=np.array([i/10,1.]),macro=np.array([i/10,1.,2.]),
                             predictions={'CANDIDATE':10.+i/200,'NO_CHANGE':10.}))
        return rows

    def test_ridge_ignores_unresolved_labels(self):
        history=self.sample();current=dict(origin=pd.Period('2018-06','M'),spot=10.,policy=np.array([3.,1.]))
        before=ridge_prediction(history,current)
        for row in history:
            if row['target']>=current['origin']:row['actual']=10000.
        self.assertEqual(before,ridge_prediction(history,current))

    def test_online_weights_ignore_unresolved_errors(self):
        rows=self.sample();origin=pd.Period('2018-06','M')
        before=optimal_shrink(rows,origin,'CANDIDATE',11.,10.)
        for row in rows:
            if row['target']>=origin:row['actual']=-10000.
        self.assertEqual(before,optimal_shrink(rows,origin,'CANDIDATE',11.,10.))
        self.assertGreaterEqual(before,10.);self.assertLessEqual(before,10.5)

    def test_interval_ignores_unresolved_errors(self):
        rows=self.sample();origin=pd.Period('2018-06','M')
        before=interval(rows,origin,'CANDIDATE',11.,.8)
        for row in rows:
            if row['target']>=origin:row['actual']=-10000.
        self.assertEqual(before,interval(rows,origin,'CANDIDATE',11.,.8))

    def test_current_target_and_calendar_mean(self):
        row=information(self.data,'2026-10')
        self.assertEqual(str(row['target']),'2026-11')
        self.assertEqual(str(row['as_of'].date()),'2026-10-22')
        self.assertEqual(row['spot'],16.)
        sample=self.data.daily.loc['2024-02-01':'2024-02-29']
        self.assertEqual(len(sample),29)
        self.assertAlmostEqual(sample.mean(),self.data.means.loc['2024-02'])

if __name__=='__main__':unittest.main()
