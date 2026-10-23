"""Independent saved-output audit; no calibration or policy choice is changed."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/2026-10-23-conditional-decision'

def main():
    loss=pd.read_csv(OUT/'losses.csv');paths=pd.read_csv(OUT/'paths.csv')
    summary=pd.read_csv(OUT/'robustness.csv');states=pd.read_csv(OUT/'initial_states.csv')
    protocol=json.loads((OUT/'protocol.json').read_text())
    keys=['initial_state','beta','scenario','output_weight','smoothing_weight','horizon']
    assert len(loss)==11232 and len(paths)==6160
    assert loss.groupby(keys).rate.nunique().eq(13).all()
    assert set(loss.tier)=={'decision'} and loss.groupby(keys).size().eq(13).all()
    assert states.groupby('initial_state').gap.nunique().eq(1).all(), 'Initial state changed with beta'
    np.testing.assert_allclose(loss.total_loss,loss.inflation_loss+loss.output_weight*loss.output_loss+
                               loss.smoothing_weight*loss.smoothing_loss,atol=1e-9)
    hold=loss.loc[loss.rate==16.,keys+['total_loss']].rename(columns={'total_loss':'hold'})
    audit=loss.merge(hold,on=keys,validate='many_to_one')
    group=audit.groupby(keys).total_loss;best=group.transform('min');worst=group.transform('max')
    regret=audit.total_loss-best
    np.testing.assert_allclose(audit.regret_vs_hold_scale,regret/np.maximum(audit.hold,1.),atol=1e-12)
    np.testing.assert_allclose(audit.regret_vs_range_scale,regret/np.maximum(worst-best,1e-9),atol=1e-12)
    maxima=audit.groupby('rate')[['regret_vs_hold_scale','regret_vs_range_scale']].max()
    np.testing.assert_allclose(summary.worst_regret_vs_hold,maxima.regret_vs_hold_scale,atol=1e-12)
    np.testing.assert_allclose(summary.worst_regret_vs_range,maxima.regret_vs_range_scale,atol=1e-12)
    selected=json.loads((OUT/'summary.json').read_text())
    assert selected['primary_rate']==float(maxima.regret_vs_hold_scale.idxmin())
    assert selected['alternative_normalization_rate']==float(maxima.regret_vs_range_scale.idxmin())
    q4=paths.loc[paths.period=='2026Q4']
    expected=(29*16+63*q4.rate)/92
    np.testing.assert_allclose(q4.ir_pol,expected,atol=1e-8)
    np.testing.assert_allclose(paths.loc[paths.rate==16.,['delta_inflation_pp','delta_gdp_pct','delta_fx_pct']],0,atol=1e-9)
    checks=pd.read_csv(OUT/'numerical_checks.csv')
    assert checks[['structural_residual','constraint_residual','affine_max_error']].to_numpy().max()<1e-7
    manifest=json.loads((OUT/'manifest.json').read_text())
    for rel,digest in manifest['inputs'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
    for rel,digest in manifest['outputs'].items():assert hashlib.sha256((OUT/rel).read_bytes()).hexdigest()==digest,rel
    rendering=json.loads((OUT/'report_manifest.json').read_text())
    for rel,digest in rendering['inputs'].items():assert hashlib.sha256((OUT/rel).read_bytes()).hexdigest()==digest,rel
    for rel,digest in rendering['output_images'].items():
        assert hashlib.sha256((ROOT/'reports/policy-decision'/rel).read_bytes()).hexdigest()==digest,rel
    evidence=ROOT/'data/policy/2026-10-23-scenarios'
    source_count=0
    for item in json.loads((evidence/'sources_manifest.json').read_text()):
        if not item.get('available'):continue
        f=ROOT/item['file'] if item['file'].startswith('data/') else evidence/item['file']
        assert hashlib.sha256(f.read_bytes()).hexdigest()==item['sha256'],item['id'];source_count+=1
    print(json.dumps(dict(status='PASS',loss_rows=len(loss),path_rows=len(paths),contexts=selected['decision_contexts'],
                         primary_rate=selected['primary_rate'],alternative_normalization_rate=selected['alternative_normalization_rate'],
                         sources=source_count,meaning='arithmetic, source integrity and fixed controls; not identified optimal policy'),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
