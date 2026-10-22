"""Explicit policy-analysis variant of the teaching QPM; legacy models unchanged.

The historical filter uses a noisy expert observation of the 2022Q2 output gap.
This is a declared identifying judgment, not a measured or estimated fact.
The stacked solver handles anticipated paths with one endogenous shock per
imposed variable and period. Its outputs are conditional model counterfactuals,
not causally identified effects, optimized policy, or conditional intervals.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.stats import norm

from research_qpm import ResearchQPM


DEFAULT_GAP_JUDGMENTS = {'2022Q2': (-18.0, 1.0)}
DEFAULT_SHOCK_CONTROLS = {
    'ir_pol': 'res_ir_tlr',
    'ir_ea': 'res_ir_ea',
    'dl_cpi_ea': 'res_dl_cpi_ea',
    'l_gdp_ea_gap': 'res_l_gdp_ea_gap',
    'dl_er_tar': 'res_dl_er_tar',
}


def _normalize_judgments(judgments):
    """None selects the documented default; {} explicitly disables judgments."""
    raw = DEFAULT_GAP_JUDGMENTS if judgments is None else judgments
    if not isinstance(raw, Mapping):
        raise ValueError('gap_judgments must be a period-to-(value, std) mapping')
    result = {}
    for when, specification in raw.items():
        period = pd.Period(when, freq='Q')
        if isinstance(specification, Mapping):
            value = float(specification['value'])
            std = float(specification['std'])
        else:
            try:
                value, std = map(float, specification)
            except (TypeError, ValueError) as exc:
                raise ValueError('Each gap judgment needs (value, positive std)') from exc
        if not np.isfinite(value) or not np.isfinite(std) or std <= 0:
            raise ValueError('Gap judgment value must be finite and std strictly positive')
        if period in result:
            raise ValueError(f'Duplicate gap judgment: {period}')
        result[period] = (value, std)
    return result


class PolicyQPM(ResearchQPM):
    """A separately named filter and exact-target policy-analysis specification.

    settings retains ResearchQPM coefficient and shock-standard-deviation keys.
    With exact_targets=True, dl_cpi_tar and dl_cpi_ea_tar are set to log(1.05)
    and log(1.02), times 100. gap_judgments=None enables (-18, sigma=1) in
    2022Q2; pass {} to disable. Judgments outside observed history are ignored.
    """
    name = 'PolicyQPM'

    def __init__(self, settings=None, gap_judgments=None, exact_targets=True):
        if not isinstance(exact_targets, (bool, np.bool_)):
            raise ValueError('exact_targets must be boolean')
        self.exact_targets = bool(exact_targets)
        self.gap_judgments = _normalize_judgments(gap_judgments)
        selected = dict(settings or {})
        if self.exact_targets:
            selected['dl_cpi_tar'] = float(100*np.log1p(.05))
            selected['dl_cpi_ea_tar'] = float(100*np.log1p(.02))
        super().__init__(selected)
        # Fail at configuration, not after a nonfinite covariance reaches LAPACK.
        for key, value in self.parameters.items():
            if key.startswith('std_') or key == 'war_potential_std':
                if not np.isfinite(value) or value < 0:
                    raise ValueError(f'{key} must be finite and nonnegative')

    def observation_frame(self, frame):
        observed = super().observation_frame(frame)
        observed['l_gdp_gap'] = np.nan
        for period, (value, _) in self.gap_judgments.items():
            if period in observed.index:
                observed.loc[period, 'l_gdp_gap'] = value
        return observed

    def fit_forecast(self, frame, horizon=9):
        """One-sided Kalman filtering with a noisy gap judgment, then rule forecast.

        Returns the same (rate_frame, metadata) form as ResearchQPM. filtered
        contains filtered, not smoothed, states. paths is the unconstrained
        rule forecast, with Gaussian intervals conditional on the calibration.
        """
        if not isinstance(frame.index, pd.PeriodIndex) or frame.index.freqstr != 'Q-DEC':
            raise ValueError('frame must have a calendar-quarter PeriodIndex (Q-DEC)')
        if len(frame) < 3 or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError('frame needs at least three unique, increasing quarters')
        if not frame.index.equals(pd.period_range(frame.index[0], frame.index[-1], freq='Q')):
            raise ValueError('frame must include each quarter; use NaN for missing observations')
        if not isinstance(horizon, Integral) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError('horizon must be a positive integer')
        observed = self.observation_frame(frame)
        if np.isinf(observed.to_numpy(dtype=float)).any():
            raise ValueError('Infinite observation; use NaN for missing values')
        self.observed = observed
        bar, drift = self.balanced_growth()
        first = frame.iloc[0]
        state = bar.copy()
        for key in ['l_gdp', 'l_cpi', 'l_er', 'l_cpi_ea']:
            value = first[key]
            if not np.isfinite(value):
                raise ValueError(f'Missing initial {key}')
            state[self.ix[key]] = value-drift[self.ix[key]]
        state[self.ix['l_gdp_tnd']] = state[self.ix['l_gdp']]
        rer = state[self.ix['l_er']]-state[self.ix['l_cpi']]+state[self.ix['l_cpi_ea']]
        state[self.ix['l_rer']] = state[self.ix['l_rer_tnd']] = rer
        state[self.ix['l_er_tar']] = state[self.ix['l_er_uip']] = state[self.ix['l_er']]
        for lag in range(1, 4):
            state[self.ix[f'cpi_lag{lag}']] = state[self.ix['l_cpi']]-lag*self.parameters['dl_cpi_tar']/4
        covariance = np.eye(self.n)*100
        identity = np.eye(self.n)
        history, innovations, judgment_diagnostics = [], [], []
        shock_means, shock_stds = [], []
        for period, row in observed.iterrows():
            shock_std = self.shock_std(period)
            weighted_impact = self.impact*shock_std
            q = weighted_impact@weighted_impact.T
            state = self.transition@state+self.intercept
            covariance = self.transition@covariance@self.transition.T+q
            mask = row.notna()
            selected_names = list(row.index[mask])
            idx = [self.ix[name] for name in selected_names]
            h = identity[idx]
            innovation = row.loc[mask].to_numpy(float)-h@state
            # Preserve legacy observation noise for measured variables. The
            # judgment has its own explicitly declared variance, not epsilon.
            rdiag = np.full(len(idx), 1e-6)
            prior_gap = float(state[self.ix['l_gdp_gap']])
            if 'l_gdp_gap' in selected_names:
                rdiag[selected_names.index('l_gdp_gap')] = self.gap_judgments[period][1]**2
            r = np.diag(rdiag)
            # Current-period disturbances conditional on the historical prefix,
            # not full-sample smoothed or independently identified causal shocks.
            shock_mean = np.zeros(self.k)
            shock_covariance = np.diag(shock_std**2)
            if len(idx):
                s = h@covariance@h.T+r
                gain = solve(s, h@covariance, assume_a='pos').T
                shock_cross = (shock_std**2)[:, None]*self.impact.T@h.T
                shock_mean = shock_cross@solve(s, innovation, assume_a='pos')
                shock_covariance -= shock_cross@solve(s, shock_cross.T, assume_a='pos')
                state = state+gain@innovation
                a = identity-gain@h
                covariance = a@covariance@a.T+gain@r@gain.T
            covariance = (covariance+covariance.T)/2
            shock_means.append(shock_mean)
            shock_stds.append(np.sqrt(np.maximum(0., np.diag(shock_covariance))))
            history.append(state.copy())
            innovations.append(dict(period=str(period), n_observed=int(mask.sum()),
                                    max_innovation=float(np.max(np.abs(innovation))) if len(idx) else 0.))
            if 'l_gdp_gap' in selected_names:
                value, std = self.gap_judgments[period]
                judgment_diagnostics.append(dict(period=str(period), value=value,
                    measurement_std=std, measurement_variance=std**2,
                    prior_gap=prior_gap, filtered_gap=float(state[self.ix['l_gdp_gap']]),
                    measurement_minus_filtered_gap=float(value-state[self.ix['l_gdp_gap']]),
                    measurement_misfit_in_declared_std=float((value-state[self.ix['l_gdp_gap']])/std),
                    filtered_gap_std=float(np.sqrt(max(0., covariance[self.ix['l_gdp_gap'], self.ix['l_gdp_gap']])))))
        self.filtered = pd.DataFrame(history, index=frame.index, columns=self.variables)
        self.last_covariance = covariance.copy()
        self.innovations = innovations
        self.gap_judgment_diagnostics = judgment_diagnostics
        self.filtered_shocks = pd.DataFrame(shock_means, index=frame.index, columns=self.shocks)
        self.filtered_shock_std = pd.DataFrame(shock_stds, index=frame.index, columns=self.shocks)
        rows, paths = [], []
        index = pd.period_range(frame.index[-1]+1, periods=horizon, freq='Q')
        weighted_impact = self.impact*self.shock_std()
        q = weighted_impact@weighted_impact.T
        for step, period in enumerate(index, 1):
            state = self.transition@state+self.intercept
            covariance = self.transition@covariance@self.transition.T+q
            mean = state[self.ix['ir_pol']]
            std = np.sqrt(max(0., covariance[self.ix['ir_pol'], self.ix['ir_pol']]))
            rows.append(dict(period=str(period), mean=mean, lower80=mean-norm.ppf(.9)*std,
                             upper80=mean+norm.ppf(.9)*std, lower95=mean-norm.ppf(.975)*std,
                             upper95=mean+norm.ppf(.975)*std, horizon=step))
            paths.append(state.copy())
        self.paths = pd.DataFrame(paths, index=index, columns=self.variables)
        return pd.DataFrame(rows), dict(solver=self.solver_info, parameters=self.parameters,
            target='quarterly_calendar_day_average_policy_rate',
            method='linear_RE_solution_missing_observation_Kalman_filter_with_noisy_gap_judgment',
            structural_parameters='teaching template calibration, not causal identification',
            foreign_path='endogenous template AR equations; use anticipated_path for imposed assumptions',
            intervals='unconstrained Gaussian forecast conditional on fixed calibration and judgments',
            exact_targets=self.exact_targets,
            inflation_target_pct=float(100*np.expm1(self.parameters['dl_cpi_tar']/100)),
            foreign_inflation_target_pct=float(100*np.expm1(self.parameters['dl_cpi_ea_tar']/100)),
            gap_judgments=[dict(period=str(p), value=v, measurement_std=s)
                           for p, (v, s) in sorted(self.gap_judgments.items())],
            applied_gap_judgments=judgment_diagnostics,
            gap_judgment_status='expert assumption with measurement uncertainty, not observed truth',
            gap_judgment_availability='default originates in 2026 teaching material; disable for earlier predictive information sets',
            disturbance_diagnostics='one-step current disturbance posterior conditional on historical prefix and judgments; not smoothed or causally identified',
            filter_status='one-sided filtered states; numerical IRIS equivalence not established')


def anticipated_path(model, initial_state, horizon, constraints=None, shocks=None,
                     *, variable_paths=None, shock_controls=None):
    """Solve anticipated shock news and imposed paths in the structural system.

    constraints: {quarter_offset: ir_pol_target}, legacy-compatible shorthand.
    shocks: {(quarter_offset, shock_name): known_innovation} in model units.
    variable_paths: {(quarter_offset, variable_name): imposed_target}.
    shock_controls: {variable_name OR (offset, variable_name): shock_name};
      omitted mappings use DEFAULT_SHOCK_CONTROLS. One distinct same-quarter
      shock is endogenized per target. An imposed policy rate is not a fixed
      policy-shock size: its required shock is an output of the solution.

    Terminal behavior is the unchanged RE rule after the horizon. All supplied
    future shocks/constraints are anticipated at the first forecast quarter.
    Returns (state_array, diagnostics), not conditional uncertainty intervals.
    """
    if not isinstance(horizon, Integral) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError('horizon must be a positive integer')
    n = model.n
    initial = np.asarray(initial_state, dtype=float)
    if initial.shape != (n,) or not np.isfinite(initial).all():
        raise ValueError(f'initial_state must be a finite vector of length {n}')
    imposed = dict(variable_paths or {})
    for when, value in dict(constraints or {}).items():
        key = (when, 'ir_pol')
        if key in imposed and imposed[key] != value:
            raise ValueError(f'Conflicting rate target at offset {when}')
        imposed[key] = value
    known = dict(shocks or {})
    controls = dict(shock_controls or {})

    def validate_offset(when):
        if not isinstance(when, Integral) or isinstance(when, bool) or not 0 <= when < horizon:
            raise ValueError(f'Offset {when!r} outside integer horizon 0..{horizon-1}')

    targets = []
    used_controls = set()
    for key, target in imposed.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError('variable_paths keys must be (quarter_offset, variable_name)')
        when, name = key
        validate_offset(when)
        if name not in model.ix or not np.isfinite(target):
            raise ValueError(f'Unknown variable or nonfinite target: {key}')
        shock = controls.get(key, controls.get(name, DEFAULT_SHOCK_CONTROLS.get(name)))
        if shock not in model.shocks:
            raise ValueError(f'Explicit valid shock control required for {name}')
        shock_index = model.shocks.index(shock)
        if not np.any(model.e[:, shock_index]):
            raise ValueError(f'Control shock {shock} is inactive in this specification')
        control_key = (when, shock)
        if control_key in used_controls:
            raise ValueError(f'Two targets cannot endogenize the same control: {control_key}')
        used_controls.add(control_key)
        targets.append((int(when), name, float(target), shock))
    targets.sort(key=lambda target: (target[0], target[1]))
    known_array = np.zeros((horizon, model.k))
    for key, value in known.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError('shocks keys must be (quarter_offset, shock_name)')
        when, name = key
        validate_offset(when)
        if name not in model.shocks or not np.isfinite(value):
            raise ValueError(f'Unknown shock or nonfinite size: {key}')
        known_array[when, model.shocks.index(name)] += value

    size = n*horizon
    matrix = np.zeros((size, size))
    rhs = np.tile(model.c, horizon)
    for when in range(horizon):
        block = slice(when*n, (when+1)*n)
        matrix[block, block] = model.a0
        if when:
            matrix[block, slice((when-1)*n, when*n)] = model.am
        else:
            rhs[block] -= model.am@initial
        if when < horizon-1:
            matrix[block, slice((when+1)*n, (when+2)*n)] = model.ap
        else:
            matrix[block, block] += model.ap@model.transition
            rhs[block] -= model.ap@model.intercept
        rhs[block] += model.e@known_array[when]

    count = len(targets)
    augmented = np.zeros((size+count, size+count))
    augmented[:size, :size] = matrix
    for j, (when, name, target, shock) in enumerate(targets):
        augmented[when*n:(when+1)*n, size+j] = -model.e[:, model.shocks.index(shock)]
        augmented[size+j, when*n+model.ix[name]] = 1.
    full_rhs = np.r_[rhs, [target[2] for target in targets]]
    try:
        solution = solve(augmented, full_rhs, check_finite=True)
    except np.linalg.LinAlgError as exc:
        raise ValueError('Imposed paths and shock controls do not identify a unique solution') from exc
    residual = float(np.max(np.abs(augmented@solution-full_rhs)))
    if not np.isfinite(residual) or residual > 1e-6:
        raise RuntimeError(f'Stacked structural/constraint residual {residual}')
    states = solution[:size].reshape(horizon, n)
    required = []
    for j, (when, name, target, shock) in enumerate(targets):
        adjustment = float(solution[size+j])
        required.append(dict(quarter_offset=when, variable=name, target=target,
            control_shock=shock, adjustment=adjustment,
            total_shock=float(known_array[when, model.shocks.index(shock)]+adjustment)))
    return states, dict(structural_residual=residual,
        constraint_residual=max((abs(states[t, model.ix[name]]-value)
                                 for t, name, value, _ in targets), default=0.),
        required_shocks=required,
        required_rule_shocks={str(row['quarter_offset']): row['total_shock']
                              for row in required if row['control_shock'] == 'res_ir_tlr'},
        n_constraints=count,
        shock_timing='all supplied future news anticipated at forecast origin',
        terminal='unchanged stable RE law, not zero terminal levels',
        uncertainty='point path only; unconstrained filter intervals do not become conditional intervals')
