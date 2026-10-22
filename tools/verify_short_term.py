"""Independent verification of three-month forecasts, weights and metrics."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/2026-10-22-short-term'

def main():
    forecasts=pd.read_csv(OUT/'forecast.csv');past=pd.read_csv(OUT/'backtest.csv')
    weights=pd.read_csv(OUT/'ensemble_weights.csv');metrics=pd.read_csv(OUT/'metrics.csv')
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert forecasts.period.unique().tolist()==['2026-11','2026-12','2027-01']
    assert len(forecasts)==18 and forecasts.actual.isna().all()
    assert forecasts.training_end.eq('2026-09').all()
    assert len(past)==1638 and past.groupby(['horizon','origin']).model.nunique().eq(6).all()
    assert past.groupby('horizon').origin.nunique().to_dict()=={1:92,2:91,3:90}
    assert (pd.PeriodIndex(past.period,freq='M').asi8-pd.PeriodIndex(past.origin,freq='M').asi8==past.horizon).all()
    assert (pd.PeriodIndex(past.training_end,freq='M')==pd.PeriodIndex(past.origin,freq='M')-1).all()
    assert past.groupby(['horizon','period']).actual.nunique().eq(1).all()
    for (split,h,model),g in past.groupby(['split','horizon','model']):
        row=metrics.loc[(metrics.split==split)&(metrics.horizon==h)&(metrics.model==model)].iloc[0]
        error=g.prediction-g.actual
        assert int(row.n)==len(g)
        np.testing.assert_allclose([row.MAE_pp,row.RMSE_pp],[error.abs().mean(),np.sqrt(np.mean(error**2))],atol=1e-10)
    assert metrics.loc[metrics.split=='audit'].n.eq(24).all()
    for h in [1,2,3]:
        selected=metrics.loc[(metrics.split=='validation')&(metrics.horizon==h)].sort_values(['MAE_pp','RMSE_pp','model']).iloc[0].model
        assert protocol['selected_models'][str(h)]==selected
        assert forecasts.loc[(forecasts.horizon==h)&forecasts.selected_by_validation,'model'].tolist()==[selected]
        w=weights.loc[weights.horizon==h].set_index('model').weight
        assert (w>=0).all() and w.NO_CHANGE>=.5-1e-12
        np.testing.assert_allclose(w.sum(),1,atol=1e-12)
        f=forecasts.loc[forecasts.horizon==h].set_index('model').prediction
        np.testing.assert_allclose(float((w*f.reindex(w.index)).sum()),f.ENSEMBLE,atol=1e-10)
    import sys
    sys.path.insert(0,str(ROOT/'models'))
    from monthly_common import MonthlyData
    daily=MonthlyData().daily
    means=daily.groupby(daily.index.to_period('M')).mean()
    np.testing.assert_allclose(past.actual,[means.loc[pd.Period(p,'M')] for p in past.period],atol=1e-10)
    for file,key in [('base_cache_manifest.json','input_sha256'),('run_manifest.json','input_sha256'),('run_manifest.json','output_sha256')]:
        for relative,digest in json.loads((OUT/file).read_text())[key].items():
            assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest,relative
    report=(ROOT/'reports/final/RESEARCH_UA.md').read_text()
    assert '2028' not in report and '2029' not in report,'Long-horizon material leaked into primary report'
    print('Three-month audit PASSED: 18 current forecasts, 1638 historical forecasts, horizon-specific weights, metrics, dates, source hashes.')

if __name__=='__main__':main()
