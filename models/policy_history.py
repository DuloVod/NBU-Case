"""Current-vintage predictive checks and explicitly ex-post policy replays.

The observed future rate path is used ONLY in a labelled conditional replay,
never in the forecasting information set. Neither replay path is an estimate
of the historical outcome caused by a different policy. Unobserved future
non-policy disturbances are zero in both paths, rather than inferred from the
realized macro outcome to make the actual policy appear successful.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from common import ROOT, DataStore, write_json
from research_qpm import ResearchQPM, anticipated_path


DEFAULT_EPISODES = {
    '2015_fx_inflation_stress': '2014Q4',
    '2020Q2_pandemic_stress': '2020Q1',
    '2022Q2_wartime_rate_stress': '2022Q1',
    '2023Q3_wartime_disinflation_stress': '2023Q2',
}
METRICS = ('rate_pp', 'inflation_yoy_pct', 'inflation_qoq_annualized_log_pp',
           'gdp_log_level', 'gdp_yoy_pct', 'fx_log_level')


def regime(period):
    """Classify by TARGET/event quarter, not the origin quarter."""
    period = pd.Period(period, freq='Q')
    if period < pd.Period('2022Q1', freq='Q'):
        return 'prewar_2015_2021'
    if period <= pd.Period('2023Q4', freq='Q'):
        return 'war_2022_2023'
    return 'later_current_vintage'


def _exact_change(log_change):
    return float(100 * np.expm1(float(log_change) / 100))


def predictive_values(model, frame, target):
    """One-quarter state forecast; annual GDP uses an observed annual lag."""
    target = pd.Period(target, freq='Q')
    state = model.paths.iloc[0]
    return {
        'rate_pp': float(state.ir_pol),
        'inflation_yoy_pct': _exact_change(state.d4l_cpi),
        'inflation_qoq_annualized_log_pp': float(state.dl_cpi),
        'gdp_log_level': float(state.l_gdp),
        'gdp_yoy_pct': _exact_change(state.l_gdp - frame.loc[target - 4, 'l_gdp']),
        'fx_log_level': float(state.l_er),
    }


def benchmark_values(frame):
    """Metric-wise persistence using last AVAILABLE values, not missing GDP."""
    gdp = frame.l_gdp.dropna()
    last_gdp = gdp.index[-1]
    return {
        'rate_pp': float(frame.ir_pol.iloc[-1]),
        'inflation_yoy_pct': _exact_change(frame.l_cpi.iloc[-1] - frame.l_cpi.iloc[-5]),
        'inflation_qoq_annualized_log_pp': float(4*(frame.l_cpi.iloc[-1]-frame.l_cpi.iloc[-2])),
        'gdp_log_level': float(gdp.iloc[-1]),
        'gdp_yoy_pct': _exact_change(gdp.iloc[-1] - gdp.loc[last_gdp - 4]),
        'fx_log_level': float(frame.l_er.iloc[-1]),
    }


def realized_values(actual):
    """Evaluation-only current-vintage labels; no labels passed to fitting."""
    row = actual.iloc[-1]
    return {
        'rate_pp': float(row.ir_pol),
        'inflation_yoy_pct': _exact_change(row.l_cpi - actual.l_cpi.iloc[-5]),
        'inflation_qoq_annualized_log_pp': float(4*(row.l_cpi-actual.l_cpi.iloc[-2])),
        'gdp_log_level': float(row.l_gdp),
        'gdp_yoy_pct': _exact_change(row.l_gdp - actual.l_gdp.iloc[-5]),
        'fx_log_level': float(row.l_er),
    }


def direct_equation_residual(model, initial, states, diagnostics, shocks=None):
    """Audit the returned solution with direct structural equations.

    Solver contract: required_rule_shocks maps zero-based quarters to the
    endogenized res_ir_tlr values; the terminal state follows the stable law.
    """
    shocks = dict(shocks or {})
    maximum = 0.
    previous = np.asarray(initial, float)
    for t, state in enumerate(states):
        following = states[t + 1] if t + 1 < len(states) else model.transition @ state + model.intercept
        eps = np.zeros(model.k)
        for (when, name), size in shocks.items():
            if when == t:
                eps[model.shocks.index(name)] += size
        rule = diagnostics.get('required_rule_shocks', {})
        eps[model.shocks.index('res_ir_tlr')] += float(rule.get(str(t), rule.get(t, 0.)))
        error = model.am @ previous + model.a0 @ state + model.ap @ following - model.c - model.e @ eps
        maximum = max(maximum, float(np.max(np.abs(error))))
        previous = state
    return maximum


class HistoricalReplayRunner:
    """Inject a model factory/solver without changing existing model modules."""

    def __init__(self, settings=None, store=None, model_factory=None, path_solver=None):
        self.settings = dict(settings or {})
        self.store = store if store is not None else DataStore()
        self.model_factory: Callable = model_factory or ResearchQPM
        self.path_solver: Callable = path_solver or anticipated_path
        self.model_source_paths = set()
        self.model_configurations = []

    def fit(self, origin, horizon):
        origin = pd.Period(origin, freq='Q')
        frame = self.store.frame(origin, gdp_lag=1)
        if frame.index[-1] != origin or frame.loc[frame.index > origin - 1, 'l_gdp'].notna().any():
            raise ValueError('Historical training frame violates GDP availability restriction')
        model = self.model_factory(self.settings)
        source = inspect.getsourcefile(type(model))
        if source:
            self.model_source_paths.add(Path(source).resolve())
        config = dict(model_class=type(model).__name__,
                      effective_settings=dict(getattr(model,'settings',self.settings)),
                      exact_targets=getattr(model,'exact_targets',None),
                      gap_judgments={str(k):v for k,v in getattr(model,'gap_judgments',{}).items()})
        if config not in self.model_configurations:
            self.model_configurations.append(config)
        model.fit_forecast(frame, horizon)
        return model, frame

    def forecasts(self, origins):
        rows = []
        for origin in origins:
            origin = pd.Period(origin, freq='Q')
            target = origin + 1
            # All predictions, including benchmarks, are materialized BEFORE
            # retrieving the future label frame.
            model, frame = self.fit(origin, 1)
            candidates = {'QPM': predictive_values(model, frame, target),
                          'PERSISTENCE_METRIC_WISE': benchmark_values(frame)}
            actual = realized_values(self.store.frame(target, gdp_lag=0))
            for name, predictions in candidates.items():
                for metric in METRICS:
                    prediction, label = predictions[metric], actual[metric]
                    available = bool(np.isfinite(label))
                    rows.append(dict(origin=str(origin), target=str(target), regime=regime(target),
                                     model=name, metric=metric, prediction=prediction,
                                     actual=label, error=prediction-label if available else np.nan,
                                     label_available=available,
                                     latest_training_gdp=str(frame.l_gdp.dropna().index[-1]),
                                     information_convention='quarter_end_plus_21d_proxy_current_vintage'))
        return pd.DataFrame(rows)

    def replay(self, episode, origin, horizon=8, control_quarters=2):
        """Ex-post rate conditioning with zero future non-policy shocks.

        Actual future rate values are not assumed to have been known by the
        policymaker. Both paths start at the same historical filtered state;
        this experiment makes no claim to recover historical shocks/causality.
        """
        if horizon < 1 or not 1 <= control_quarters <= horizon:
            raise ValueError('Require 1 <= control_quarters <= horizon')
        origin = pd.Period(origin, freq='Q')
        model, frame = self.fit(origin, horizon)
        initial = model.filtered.iloc[-1].to_numpy()
        baseline, baseline_diag = self.path_solver(model, initial, horizon)
        rate_ix = model.ix['ir_pol']
        spot_proxy = float(frame.ir_pol.iloc[-1])
        # Direct raw current-vintage rate labels, used solely as ex-post controls.
        controls = {}
        for t in range(control_quarters):
            period = origin + 1 + t
            rate = float(self.store.levels.loc[period, 'ir_pol'])
            if not np.isfinite(rate):
                raise ValueError(f'Missing ex-post rate label for {period}')
            controls[t] = rate
        alternatives = {
            'endogenous_rule_reference': {},
            'actual_rate_path_ex_post': controls,
            'flat_last_quarter_average': {t: spot_proxy for t in range(control_quarters)},
        }
        paths, checks, by_policy = [], [], {}
        for name, constraint in alternatives.items():
            states, diag = self.path_solver(model, initial, horizon, constraints=constraint)
            residual = direct_equation_residual(model, initial, states, diag)
            control_error = max((abs(states[t,rate_ix]-value) for t,value in constraint.items()), default=0.)
            if residual > 1e-6 or control_error > 1e-8:
                raise RuntimeError(f'Historical replay failed equation/constraint audit: {episode}/{name}')
            checks.append(dict(episode=episode, origin=str(origin), policy=name,
                               direct_equation_residual=residual, max_control_error=control_error,
                               baseline_recursive_max_error=float(np.max(np.abs(baseline-model.paths.to_numpy()))),
                               instrument_baseline=spot_proxy,
                               instrument_baseline_definition='last_observed_quarter_average_not_meeting_spot',
                               control_quarters=control_quarters,
                               required_rule_shocks=str(diag.get('required_rule_shocks', {}))))
            for t,state in enumerate(states):
                paths.append(dict(episode=episode, origin=str(origin), period=str(origin+1+t),
                                  regime=regime(origin+1), policy=name, horizon=t+1,
                                  ir_pol=float(state[rate_ix]),
                                  inflation_yoy_pct=_exact_change(state[model.ix['d4l_cpi']]),
                                  gdp_gap_log_pct=float(state[model.ix['l_gdp_gap']]),
                                  gdp_log_level=float(state[model.ix['l_gdp']]),
                                  fx_uah_per_eur=float(np.exp(state[model.ix['l_er']]/100)),
                                  experiment_status='ex_post_conditional_not_causal_counterfactual'))
            by_policy[name] = np.asarray(states)
        actual, flat = by_policy['actual_rate_path_ex_post'], by_policy['flat_last_quarter_average']
        differences = []
        for t in range(horizon):
            differences.append(dict(episode=episode, origin=str(origin), period=str(origin+1+t), horizon=t+1,
                                    delta_ir_pol_pp=float(actual[t,rate_ix]-flat[t,rate_ix]),
                                    delta_inflation_yoy_pp=_exact_change(actual[t,model.ix['d4l_cpi']])-_exact_change(flat[t,model.ix['d4l_cpi']]),
                                    delta_gdp_gap_log_pct=float(actual[t,model.ix['l_gdp_gap']]-flat[t,model.ix['l_gdp_gap']]),
                                    comparison='actual_rate_path_ex_post_minus_flat',
                                    causal_success_claim=False))
        fidelity = self.shock_fidelity(episode, origin, model, initial, horizon, baseline)
        return paths, differences, checks, fidelity

    def shock_fidelity(self, episode, origin, model, initial, horizon, baseline):
        """Unit shock probes at episode states; NOT estimated event shocks."""
        rows = []
        shocks = [('demand_unit', 'res_l_gdp_gap', 1.),
                  ('cost_unit', 'res_dl_cpi', 1.),
                  ('policy_rule_unit', 'res_ir_tlr', 1.),
                  ('risk_unit', 'res_prem_uip_gap', 1.),
                  ('potential_level_loss_unit', 'res_l_gdp_tnd', -1.)]
        for label, shock_name, size in shocks:
            imposed = {(0,shock_name):size}
            states, diag = self.path_solver(model, initial, horizon, shocks=imposed)
            residual = direct_equation_residual(model, initial, states, diag, imposed)
            if residual > 1e-6:
                raise RuntimeError(f'Shock equation audit failed: {episode}/{label}')
            for t,delta in enumerate(states-baseline):
                rows.append(dict(episode=episode, origin=str(origin), regime=regime(pd.Period(origin,freq='Q')+1),
                                 probe=label, shock=shock_name, imposed_size=size, horizon=t+1,
                                 delta_ir_pol_pp=float(delta[model.ix['ir_pol']]),
                                 delta_inflation_yoy_log_pp=float(delta[model.ix['d4l_cpi']]),
                                 delta_gdp_gap_log_pct=float(delta[model.ix['l_gdp_gap']]),
                                 delta_gdp_log_level=float(delta[model.ix['l_gdp']]),
                                 delta_fx_log_pct=float(delta[model.ix['l_er']]),
                                 direct_equation_residual=residual,
                                 interpretation='unit_model_probe_not_identified_historical_event'))
        return rows


def summarize_metrics(forecasts):
    rows=[]
    for keys, group in forecasts.groupby(['regime','model','metric']):
        errors = group.loc[group.label_available,'error'].dropna()
        rows.append(dict(zip(('regime','model','metric'),keys), n=len(errors),
                         MAE=float(errors.abs().mean()) if len(errors) else None,
                         RMSE=float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
                         bias=float(errors.mean()) if len(errors) else None))
    return pd.DataFrame(rows)


def run_historical_validation(output, settings=None, store=None, origins=None,
                              episodes=None, horizon=8, control_quarters=2,
                              model_factory=None, path_solver=None):
    """Write forecasts/metrics, ex-post rate replays, probes and protocol.

    model_factory(settings) must expose the ResearchQPM public state/transition
    interface; path_solver uses anticipated_path's arguments and diagnostics.
    Supplying episodes={} disables replay while preserving predictive checks.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runner = HistoricalReplayRunner(settings,store,model_factory,path_solver)
    origins = list(pd.period_range('2014Q4','2023Q3',freq='Q') if origins is None else origins)
    if not origins:
        raise ValueError('At least one forecast origin is required')
    episodes = dict(DEFAULT_EPISODES if episodes is None else episodes)
    forecasts = runner.forecasts(origins)
    metrics = summarize_metrics(forecasts)
    forecasts.to_csv(output/'historical_forecasts.csv',index=False)
    metrics.to_csv(output/'historical_metrics.csv',index=False)
    paths, differences, checks, fidelity = [], [], [], []
    for episode,origin in episodes.items():
        p,d,c,f=runner.replay(episode,origin,horizon,control_quarters)
        paths.extend(p); differences.extend(d); checks.extend(c); fidelity.extend(f)
    tables = {'historical_policy_replay.csv':paths,
              'historical_policy_differences.csv':differences,
              'historical_replay_checks.csv':checks,
              'historical_shock_fidelity.csv':fidelity}
    for name,rows in tables.items():
        pd.DataFrame(rows).to_csv(output/name,index=False)
    inputs = [ROOT/'data/inputs/core_monthly.csv',ROOT/'data/model/quarterly_levels.csv',
              ROOT/'models/qpm.py', ROOT/'models/research_qpm.py',Path(__file__),
              ROOT/'nbu_challenge_model_2026/model/model.model',ROOT/'nbu_challenge_model_2026/model/param.m']
    inputs += [p for p in runner.model_source_paths if p not in inputs]
    solver_source=inspect.getsourcefile(runner.path_solver)
    if solver_source and Path(solver_source).resolve() not in inputs:
        inputs.append(Path(solver_source).resolve())
    protocol = dict(
        as_of='2026-10-22', settings=dict(settings or {}),
        forecast_origins=[str(p) for p in origins],
        training='prefix through origin; both GDP series lagged one quarter; seasonal adjustment refit on prefix',
        vintage='current revised source vintage; historical first-release dates unavailable; NOT strict real-time',
        prior_inspection='historical outcomes already inspected; NOT pristine holdout',
        calibration='no tuning by this module; supplied settings may have been selected on overlapping history; external selection record required',
        forecast_metrics_units={'rate_pp':'calendar-quarter-average rate percent per annum',
                               'inflation_yoy_pct':'exact annual percent change of quarterly average research-SA CPI',
                               'inflation_qoq_annualized_log_pp':'four times quarter change of 100*ln(research-SA CPI)',
                               'gdp_log_level':'100*ln(real GDP); error approximates percent level error',
                               'gdp_yoy_pct':'exact annual percent change of real GDP',
                               'fx_log_level':'100*ln(UAH/EUR); error approximates percent FX level error'},
        baseline='metric-wise last available value; not a coherent structural macro path',
        episodes=episodes, horizon_quarters=horizon, control_quarters=control_quarters,
        replay='actual FUTURE quarterly average rates imposed EX POST versus last historical quarterly average held flat',
        replay_information='future actual rate labels appear only in conditional replay, never forecast fitting',
        replay_policy_after_control='same endogenous rule in all alternatives',
        replay_nonpolicy_shocks='zero future nonpolicy shocks in all alternatives; no outcome-fitted shock reconstruction',
        replay_limits=['not an identified causal counterfactual',
                       'not proof that actual policy caused historical disinflation or success',
                       'flat baseline uses quarter-average rate, not decision-date spot',
                       'episode names are calendar labels, not identified disturbance classifications',
                       'unit shock probes are model fidelity checks, not estimated event magnitudes',
                       'war regime reported separately; fixed small-model transmission may not transfer'],
        model_factory=getattr(runner.model_factory,'__name__',str(runner.model_factory)),
        effective_model_configurations=runner.model_configurations,
        path_solver=getattr(runner.path_solver,'__name__',str(runner.path_solver)),
        rows={name:len(rows) for name,rows in tables.items()},
        forecast_rows=len(forecasts),
        sha256={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        numpy=np.__version__, pandas=pd.__version__)
    write_json(output/'historical_protocol.json',protocol)
    return dict(protocol=protocol,forecasts=forecasts,metrics=metrics,
                paths=pd.DataFrame(paths),differences=pd.DataFrame(differences),
                checks=pd.DataFrame(checks),shock_fidelity=pd.DataFrame(fidelity))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'results/2026-10-22-policy-decision/history')
    args = parser.parse_args()
    result = run_historical_validation(args.output)
    print(result['metrics'].to_string(index=False))
