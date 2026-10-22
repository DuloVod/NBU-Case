"""Small, predeclared predictive calibration of the corrected PolicyQPM.

The candidate score excludes the policy rate. This estimates neither causal
transmission coefficients nor central-bank preferences. Historical outcomes
were previously inspected and data are revised/current-vintage, so the final
chronological audit is not a pristine unseen test or a real-time-vintage test.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd

from common import ROOT, DataStore, write_json
from policy_qpm import PolicyQPM


AS_OF = '2026-10-22'
OUT = ROOT/'results/2026-10-22-policy-decision/calibration'
SELECTION_METRICS = ('inflation_yoy_pct', 'gdp_log_percent', 'fx_log_percent')
DIAGNOSTIC_METRICS = (*SELECTION_METRICS, 'rate_pp')
SPLITS = {
    'development': ('2015Q4', '2019Q4'),
    'validation': ('2020Q1', '2022Q4'),
    'audit': ('2023Q1', '2026Q1'),
}
REQUIRED_VALIDATION_GAIN = .05
THRESHOLD_ROUNDOFF_TOL = 1e-12
HUBER_CUTOFF = 2.
SCALE_FLOOR = 1.


def candidate_settings():
    """Nine distinct effective pairs, with corrected-template default first."""
    candidates = [{}]
    seen = {(.7, .4)}
    for beta, memory in product((.5, .7, .9), (.3, .4, .6)):
        if (beta, memory) not in seen:
            candidates.append(dict(beta=beta, inflation_memory=memory))
            seen.add((beta, memory))
    return candidates


def prediction_vector(model, frame):
    row = model.paths.iloc[0]
    # Forecast quarter t+1 minus observed quarter (t+1)-4 = frame[-4].
    inflation_log = float(row.l_cpi-frame.l_cpi.iloc[-4])
    return {
        'inflation_yoy_pct': float(100*np.expm1(inflation_log/100)),
        'gdp_log_percent': float(row.l_gdp),
        'fx_log_percent': float(row.l_er),
        'rate_pp': float(row.ir_pol),
    }


def target_vector(store, period):
    """Evaluation labels only; never supply this frame to model.fit_forecast."""
    actual = store.frame(period, gdp_lag=0)
    row = actual.iloc[-1]
    inflation_log = float(row.l_cpi-actual.l_cpi.iloc[-5])
    values = {
        'inflation_yoy_pct': float(100*np.expm1(inflation_log/100)),
        'gdp_log_percent': float(row.l_gdp),
        'fx_log_percent': float(row.l_er),
        'rate_pp': float(row.ir_pol),
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError(f'Incomplete evaluation labels at {period}: {values}')
    return values


def _predictions(settings, origins, training_frames, candidate, split):
    model = PolicyQPM(settings=settings, gap_judgments={}, exact_targets=True)
    rows = []
    for origin in origins:
        frame = training_frames[origin]
        model.fit_forecast(frame, horizon=1)
        predictions = prediction_vector(model, frame)
        for metric, prediction in predictions.items():
            if not np.isfinite(prediction):
                raise ValueError(f'Nonfinite candidate {candidate} prediction, {origin}, {metric}')
            rows.append(dict(candidate=candidate, split=split, origin=str(origin),
                period=str(origin+1), metric=metric, prediction=prediction,
                used_in_selection=metric in SELECTION_METRICS,
                settings=json.dumps(settings, sort_keys=True)))
    return pd.DataFrame(rows)


def _attach_labels(predictions, labels):
    result = predictions.copy()
    result['actual'] = [labels[pd.Period(row.period, freq='Q')][row.metric]
                        for row in result.itertuples()]
    result['error'] = result.prediction-result.actual
    return result


def predictive_score(predictions, scales):
    """Equal-weight normalized Huber score over the three outcome metrics only."""
    selected = predictions.loc[predictions.metric.isin(SELECTION_METRICS)]
    if selected.empty or selected.groupby('origin').metric.nunique().ne(3).any():
        raise ValueError('Each calibration origin needs all three outcome metrics')
    standardized = selected.error.to_numpy()/selected.metric.map(scales).to_numpy()
    magnitude = np.abs(standardized)
    value = np.where(magnitude <= HUBER_CUTOFF, .5*magnitude**2,
                     HUBER_CUTOFF*(magnitude-.5*HUBER_CUTOFF))
    return float(value.mean())


def _metrics(predictions):
    rows = []
    for (candidate, split, metric), group in predictions.groupby(['candidate', 'split', 'metric']):
        error = group.error.to_numpy()
        rows.append(dict(candidate=int(candidate), split=split, metric=metric,
            n=len(error), used_in_selection=metric in SELECTION_METRICS,
            MAE=float(np.abs(error).mean()), RMSE=float(np.sqrt(np.mean(error**2))),
            bias=float(error.mean())))
    return pd.DataFrame(rows)


def calibrate(store, output=OUT):
    """Run the fixed grid; return (selected_settings, full_protocol).

    Corrected template is retained unless validation score improves by at least
    five percent. Development scales, candidates, score and threshold are fixed
    before validation; audit labels never enter selection. DataStore frames are
    cached and all predictions for a split are formed before its labels attach.
    """
    began = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    candidates = candidate_settings()
    origins = {name: pd.period_range(start, end, freq='Q')
               for name, (start, end) in SPLITS.items()}
    all_origins = [origin for group in origins.values() for origin in group]
    training = {origin: store.frame(origin, gdp_lag=1).copy() for origin in all_origins}
    # Labels occupy a separate dictionary; no model call receives this frame.
    labels = {}

    def label_split(predictions):
        for value in predictions.period.unique():
            period = pd.Period(value, freq='Q')
            if period not in labels:
                labels[period] = target_vector(store, period)
        return _attach_labels(predictions, labels)

    development_default = label_split(_predictions(
        candidates[0], origins['development'], training, 0, 'development'))
    scales = development_default.loc[development_default.metric.isin(SELECTION_METRICS)]
    scales = scales.groupby('metric').error.apply(lambda x: float(np.sqrt(np.mean(x**2))))
    scales = scales.clip(lower=SCALE_FLOOR).reindex(SELECTION_METRICS)
    if not np.isfinite(scales.to_numpy()).all():
        raise ValueError('Nonfinite development scales')

    validation_raw = pd.concat([_predictions(setting, origins['validation'], training, i, 'validation')
                               for i, setting in enumerate(candidates)], ignore_index=True)
    validation = label_split(validation_raw)
    scores = []
    for i, setting in enumerate(candidates):
        score = predictive_score(validation.loc[validation.candidate == i], scales)
        scores.append(dict(candidate=i, beta=float(setting.get('beta', .7)),
            inflation_memory=float(setting.get('inflation_memory', .4)),
            validation_score=score, settings=json.dumps(setting, sort_keys=True)))
    grid = pd.DataFrame(scores).sort_values(['validation_score', 'candidate'])
    default_score = float(grid.loc[grid.candidate == 0, 'validation_score'].iloc[0])
    best_candidate = int(grid.iloc[0].candidate)
    best_score = float(grid.iloc[0].validation_score)
    improvement = 1-best_score/default_score if default_score > 0 else 0.
    reaches_threshold = improvement >= REQUIRED_VALIDATION_GAIN or np.isclose(
        improvement, REQUIRED_VALIDATION_GAIN, rtol=0., atol=THRESHOLD_ROUNDOFF_TOL)
    selected = best_candidate if reaches_threshold else 0
    chosen = dict(candidates[selected])
    # Selection is now final; materialize audit predictions without using them
    # to revise settings, score, scale, candidate set or threshold.
    audit = label_split(pd.concat([
        _predictions(setting, origins['audit'], training, i, 'audit')
        for i, setting in enumerate(candidates)], ignore_index=True))
    development_others = label_split(pd.concat([
        _predictions(setting, origins['development'], training, i, 'development')
        for i, setting in enumerate(candidates) if i != 0], ignore_index=True))
    predictions = pd.concat([development_default, development_others, validation, audit], ignore_index=True)
    predictions['selected'] = predictions.candidate.eq(selected)
    predictions['corrected_template'] = predictions.candidate.eq(0)
    grid['improvement_fraction_vs_corrected_template'] = 1-grid.validation_score/default_score if default_score > 0 else 0.
    grid['selected'] = grid.candidate.eq(selected)
    grid['best_validation'] = grid.candidate.eq(best_candidate)
    protocol = dict(as_of=AS_OF,
        splits={name: dict(first_origin=str(index[0]), last_origin=str(index[-1]),
                           n_origins=len(index), first_target=str(index[0]+1),
                           last_target=str(index[-1]+1)) for name, index in origins.items()},
        candidate_count=len(candidates), candidates=candidates,
        objective='equal-weight standardized Huber loss of exact quarterly-average CPI yoy percent, GDP log level, FX log level',
        selection_metrics=list(SELECTION_METRICS), rate_used_in_selection=False,
        diagnostic_metrics=list(DIAGNOSTIC_METRICS),
        huber_cutoff=HUBER_CUTOFF, scale_floor=SCALE_FLOOR,
        scales_from_corrected_template_development_rmse=scales.to_dict(),
        default_validation_score=default_score, best_validation_score=best_score,
        best_validation_candidate=best_candidate, best_validation_settings=candidates[best_candidate],
        validation_improvement_fraction=improvement,
        required_validation_gain=REQUIRED_VALIDATION_GAIN,
        threshold_roundoff_tolerance=THRESHOLD_ROUNDOFF_TOL,
        selection='retain corrected template unless validation score improves at least 5%; never select on audit',
        selected_candidate=selected, selected_settings=chosen,
        gap_judgments='disabled for all historical predictive origins; 2026 teaching judgment unavailable in 2022',
        exact_targets=True, inflation_target_pct=5., foreign_inflation_target_pct=2.,
        train_information='DataStore.frame(origin,gdp_lag=1), current revised vintage truncated before preprocessing',
        labels='DataStore.frame(target,gdp_lag=0), labels attached only after split predictions exist',
        data_availability='audit last target 2026Q2, within available actual GDP',
        inflation_definition='exact yoy percent of seasonally adjusted quarterly-average CPI, not December/December or monthly CPI',
        gdp_definition='100 log real GDP level; squared forecast errors in log points',
        fx_definition='100 log quarterly-average UAH per EUR',
        calibration_status='bounded predictive calibration of two coefficients; not causal structural identification or policy loss-weight estimation',
        caveats=['historical outcomes already inspected; audit not pristine unseen holdout',
                 'revised current-vintage data; actual historical release vintages not reconstructed',
                 'one-quarter GDP availability lag is a proxy, not a verified release calendar',
                 'historical regimes differ; fixed teaching calibration is retrospective',
                 'predictive gain does not establish policy welfare dominance'],
        elapsed_seconds=float(time.perf_counter()-began))
    files = [Path(__file__), ROOT/'models/policy_qpm.py', ROOT/'models/research_qpm.py',
             ROOT/'models/qpm.py', ROOT/'models/common.py',
             ROOT/'data/inputs/core_monthly.csv', ROOT/'data/model/quarterly_levels.csv']
    files += list((ROOT/'nbu_challenge_model_2026/model').glob('*'))
    protocol['sha256'] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in files if path.is_file()}
    grid.to_csv(output/'policy_calibration_grid.csv', index=False)
    predictions.to_csv(output/'policy_calibration_predictions.csv', index=False)
    _metrics(predictions).to_csv(output/'policy_calibration_metrics.csv', index=False)
    write_json(output/'policy_calibration_protocol.json', protocol)
    print(json.dumps(dict(selected_settings=chosen, best_validation_settings=candidates[best_candidate],
                          validation_improvement_fraction=improvement, candidate_count=len(candidates),
                          elapsed_seconds=protocol['elapsed_seconds']), ensure_ascii=False), flush=True)
    return chosen, protocol


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    calibrate(DataStore(), args.output)
