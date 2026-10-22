#!/usr/bin/env python3
"""Independent checks of saved inputs, splits, arithmetic and policy comparisons."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'models'))
from common import write_json
from policy_nowcast import evaluation_split
from policy_calibration import predictive_score


def verify(output):
    output=Path(output)
    manifest=json.loads((output/'run_manifest.json').read_text())
    for key,digest in manifest['inputs'].items():
        assert hashlib.sha256((ROOT/key).read_bytes()).hexdigest()==digest,('Input changed',key)
    for key,digest in manifest['outputs'].items():
        assert hashlib.sha256((output/key).read_bytes()).hexdigest()==digest,('Output changed',key)
    extension=ROOT/'data/policy/2026-10-22'
    data=pd.read_csv(extension/'long.csv')
    assert not data.duplicated(['series','period']).any()
    assert data.period.max()=='2026-09'
    known=data.loc[data.published_at!='unknown','published_at']
    assert known.empty or known.le('2026-10-22').all()
    for row in json.loads((extension/'downloads_manifest.json').read_text()):
        assert hashlib.sha256((ROOT/row['file']).read_bytes()).hexdigest()==row['sha256']
    calibration=output/'calibration'
    protocol=json.loads((calibration/'policy_calibration_protocol.json').read_text())
    predictions=pd.read_csv(calibration/'policy_calibration_predictions.csv')
    grid=pd.read_csv(calibration/'policy_calibration_grid.csv')
    for row in grid.itertuples():
        score=predictive_score(predictions.loc[(predictions.candidate==row.candidate)&
             (predictions.split=='validation')],protocol['scales_from_corrected_template_development_rmse'])
        assert np.isclose(score,row.validation_score,atol=1e-12)
    best=grid.sort_values(['validation_score','candidate']).iloc[0]
    expected=int(best.candidate) if protocol['validation_improvement_fraction']>=.05-1e-12 else 0
    assert expected==protocol['selected_candidate']
    assert protocol['rate_used_in_selection'] is False
    cpi=pd.read_csv(output/'nowcast/cpi_backtest.csv')
    assert all(evaluation_split(r.origin,r.period)==r.split for r in cpi.itertuples())
    assert cpi.loc[cpi.split=='validation','period'].max()<='2024-08'
    assert cpi.loc[cpi.split=='audit','origin'].min()>='2024-08'
    nowcast=pd.read_csv(output/'nowcast/cpi_metrics.csv')
    current=pd.read_csv(output/'nowcast/cpi_forecast.csv')
    for h in range(1,5):
        rank=nowcast.loc[(nowcast.horizon==h)&(nowcast.split=='validation')].sort_values(['MAE_pp','RMSE_pp','model'])
        assert current.loc[(current.horizon==h)&current.selected_by_validation,'model'].item()==rank.iloc[0].model
    folder=output/'decision'
    paths=pd.read_csv(folder/'policy_paths.csv')
    losses=pd.read_csv(folder/'policy_losses.csv')
    winners=pd.read_csv(folder/'conditional_winners.csv')
    assert np.isfinite(losses.select_dtypes('number').to_numpy()).all()
    assert np.isfinite(paths.select_dtypes('number').to_numpy()).all()
    calculated=losses.inflation_loss+losses.output_weight*losses.output_loss+losses.smoothing_weight*losses.smoothing_loss
    np.testing.assert_allclose(calculated,losses.total_loss,atol=1e-9)
    expected_average=(29*16+49*losses.october_rate+14*losses.december_rate)/92
    np.testing.assert_allclose(expected_average,losses.q4_average_rate,atol=1e-12)
    numerical=pd.read_csv(folder/'numerical_checks.csv')
    residual=float(numerical[['affine_path_max_error','structural_residual','constraint_residual']].to_numpy().max())
    assert residual<1e-7
    keys=['variant','scenario','horizon_quarters','output_weight','smoothing_weight']
    minima=losses.groupby(keys).total_loss.min().reset_index(name='minimum')
    paired=winners.merge(minima,on=keys,validate='one_to_one')
    np.testing.assert_allclose(paired.total_loss,paired.minimum,atol=1e-9)
    assert winners.grid_boundary.equals(winners.october_rate.isin([14.,20.])|winners.december_rate.isin([14.,20.]))
    held=paths.loc[(paths.october_rate==16)&(paths.december_rate==16)]
    np.testing.assert_allclose(held[['delta_inflation_pp_vs_hold16','delta_gdp_percent_vs_hold16','delta_fx_percent_vs_hold16']],0.,atol=1e-10)
    for name in ['legacy','corrected_no_retrospective_judgment']:
        historical=output/'history'/name
        hp=json.loads((historical/'historical_protocol.json').read_text())
        for path,digest in hp['sha256'].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
        checks=pd.read_csv(historical/'historical_replay_checks.csv')
        numeric=checks.select_dtypes('number')
        # Direct equation checks are separately recorded by historical runner.
        for col in numeric.columns:
            if 'residual' in col or 'error' in col:
                assert numeric[col].abs().max()<1e-7,(name,col)
    result=dict(status='PASS',as_of='2026-10-22',input_hashes=len(manifest['inputs']),
        output_hashes=len(manifest['outputs']),dataset_values=len(data),calibration_predictions=len(predictions),
        cpi_historical_forecasts=len(cpi),policy_comparisons=len(losses),conditional_winners=len(winners),
        max_numerical_residual=residual,
        scope='source integrity, chronological split rules, selection, units/arithmetic, structural equations; not empirical proof of causality')
    write_json(output/'verification.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/2026-10-22-policy-decision')
    verify(p.parse_args().output)
