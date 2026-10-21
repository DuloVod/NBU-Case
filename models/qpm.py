"""Linear rational-expectations QPM port of the supplied NBU teaching template.

The equations are parsed, NOT executed. A first-order RE policy solution and a
missing-observation Kalman filter replace MATLAB/IRIS. This is not a claim of
numerical equivalence to IRIS's initialization, filters, or expert judgments.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.linalg import solve, eigvals, ordqz
from scipy.stats import norm
from statsmodels.tsa.filters.hp_filter import hpfilter


def section(text, name):
    return text.split('!' + name, 1)[1].split('!', 1)[0]


def names(text, name):
    return [line.split()[-1] for raw in section(text, name).splitlines()
            if (line := raw.split('%')[0].strip())]


@dataclass
class Linear:
    coef: np.ndarray
    constant: float = 0.0

    def __add__(self, other):
        if not isinstance(other, Linear):
            other = Linear(np.zeros_like(self.coef), float(other))
        return Linear(self.coef + other.coef, self.constant + other.constant)

    __radd__ = __add__

    def __neg__(self):
        return Linear(-self.coef, -self.constant)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, Linear):
            if np.any(other.coef):
                if np.any(self.coef):
                    raise ValueError('Nonlinear expression in a linear model')
                return other * self.constant
            other = other.constant
        return Linear(self.coef * other, self.constant * other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Linear):
            if np.any(other.coef):
                raise ValueError('Variable denominator')
            other = other.constant
        return self * (1 / other)


class QPM:
    name = 'QPM'

    def __init__(self, template: Path | None = None, overrides=None):
        template = Path(template or Path(__file__).resolve().parents[1] / 'nbu_challenge_model_2026' / 'model')
        text = (template / 'model.model').read_text(encoding='utf-8-sig')
        param_text = (template / 'param.m').read_text(encoding='utf-8-sig')
        # Numeric literal assignments only. In p.beta = 0.7; 0.8; MATLAB uses 0.7.
        self.parameters = {k: float(v) for k, v in re.findall(
            r'p\.(\w+)\s*=\s*([-+]?\d+(?:\.\d+)?)\s*;', param_text)}
        self.parameters.update(overrides or {})
        self.variables = names(text, 'transition_variables')
        self.original_variables = self.variables.copy()
        self.variables += ['cpi_lag1', 'cpi_lag2', 'cpi_lag3',
                           'd4_cpi_lead1', 'd4_cpi_lead2']
        self.shocks = names(text, 'transition_shocks')
        self.ix = {v: i for i, v in enumerate(self.variables)}
        self.n, self.k = len(self.variables), len(self.shocks)
        clean = '\n'.join(x.split('%')[0] for x in section(text, 'transition_equations').splitlines())
        self.equations = [x.strip() for x in clean.split(';') if x.strip()]
        self.equations += ['cpi_lag1 = l_cpi{-1}', 'cpi_lag2 = cpi_lag1{-1}',
                           'cpi_lag3 = cpi_lag2{-1}',
                           'd4_cpi_lead1 = d4l_cpi{+1}',
                           'd4_cpi_lead2 = d4_cpi_lead1{+1}']
        if len(self.equations) != self.n:
            raise ValueError(f'{len(self.equations)} equations for {self.n} variables')
        rows = []
        for equation in self.equations:
            lhs, rhs = equation.split('=')
            rows.append(self.expression(lhs) - self.expression(rhs))
        matrix = np.array([r.coef for r in rows])
        self.am = matrix[:, :self.n]
        self.a0 = matrix[:, self.n:2*self.n]
        self.ap = matrix[:, 2*self.n:3*self.n]
        self.e = -matrix[:, 3*self.n:]
        self.c = -np.array([r.constant for r in rows])
        self.solve()

    def expression(self, text):
        text = re.sub(r'(\w+)\{([+-]?\d+)\}',
                      lambda m: f'{m[1]}__{int(m[2]):+d}'.replace('+', 'p').replace('-', 'm'),
                      text.strip())

        def read(node):
            if isinstance(node, ast.Expression):
                return read(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return Linear(np.zeros(3*self.n+self.k), float(node.value))
            if isinstance(node, ast.Name):
                name, lag = node.id, 0
                m = re.fullmatch(r'(\w+)__([pm])(\d+)', name)
                if m:
                    name, lag = m[1], int(m[3]) * (1 if m[2] == 'p' else -1)
                if name == 'l_cpi' and lag == -4:
                    name, lag = 'cpi_lag3', -1
                if name == 'd4l_cpi' and lag == 3:
                    name, lag = 'd4_cpi_lead2', 1
                if name in self.parameters and lag == 0:
                    return Linear(np.zeros(3*self.n+self.k), self.parameters[name])
                result = Linear(np.zeros(3*self.n+self.k))
                if name in self.shocks and lag == 0:
                    result.coef[3*self.n+self.shocks.index(name)] = 1
                elif name in self.ix and lag in (-1, 0, 1):
                    result.coef[(lag+1)*self.n+self.ix[name]] = 1
                else:
                    raise ValueError(f'Unsupported variable/lag: {name}, {lag}')
                return result
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
                return -read(node.operand) if isinstance(node.op, ast.USub) else read(node.operand)
            if isinstance(node, ast.BinOp):
                a, b = read(node.left), read(node.right)
                if isinstance(node.op, ast.Add): return a + b
                if isinstance(node.op, ast.Sub): return a - b
                if isinstance(node.op, ast.Mult): return a * b
                if isinstance(node.op, ast.Div): return a / b
            raise ValueError(f'Expression not allowed: {ast.dump(node)}')

        return read(ast.parse(text, mode='eval'))

    def balanced_growth(self):
        p, ix = self.parameters, self.ix
        bar, drift = np.zeros(self.n), np.zeros(self.n)
        inflation, foreign_pi = p['dl_cpi_tar'], p['dl_cpi_ea_tar']
        drer, growth = p['dl_rer_tnd_ss'], p['dl_gdp_tnd_ss']
        neutral = p['prem_uip_tnd_ss'] + drer + p['rir_ea_tnd_ss']
        values = dict(dl_gdp_tnd=growth, dl_rer_tnd=drer,
                      prem_uip=p['prem_uip_tnd_ss'], prem_uip_tnd=p['prem_uip_tnd_ss'],
                      rir_ea_tnd=p['rir_ea_tnd_ss'], rir_tnd=neutral, rir_pol=neutral,
                      dl_cpi=inflation, d4l_cpi=inflation, d4_cpi_lead1=inflation,
                      d4_cpi_lead2=inflation, dl_cpi_ea=foreign_pi,
                      ir_ea=p['rir_ea_tnd_ss']+foreign_pi,
                      ir_pol=neutral+inflation, ir_tlr=neutral+inflation,
                      ir_uip=neutral+inflation, dl_er_tar=drer+inflation-foreign_pi,
                      cpi_lag1=-inflation/4, cpi_lag2=-inflation/2,
                      cpi_lag3=-3*inflation/4)
        for key, value in values.items(): bar[ix[key]] = value
        for key in ['l_gdp', 'l_gdp_tnd']: drift[ix[key]] = growth/4
        for key in ['l_cpi', 'cpi_lag1', 'cpi_lag2', 'cpi_lag3']: drift[ix[key]] = inflation/4
        drift[ix['l_cpi_ea']] = foreign_pi/4
        for key in ['l_er', 'l_er_tar', 'l_er_uip']:
            drift[ix[key]] = (drer+inflation-foreign_pi)/4
        for key in ['l_rer', 'l_rer_tnd']: drift[ix[key]] = drer/4
        return bar, drift

    def solve(self):
        # Linear time iteration chooses the stable minimal-state RE solution.
        transition = np.zeros((self.n, self.n))
        for iteration in range(20000):
            nxt = solve(self.a0 + self.ap @ transition, -self.am, check_finite=False)
            change = np.max(np.abs(nxt-transition))
            transition = nxt
            if change < 1e-11: break
        else:
            raise RuntimeError('QPM RE time iteration did not converge')
        self.transition = transition
        self.impact = solve(self.a0+self.ap@transition, self.e, check_finite=False)
        bar, drift = self.balanced_growth()
        self.intercept = bar+drift-transition@bar
        residual = np.max(np.abs(self.ap@transition@transition+self.a0@transition+self.am))
        affine = np.max(np.abs(self.a0@self.intercept + self.ap@(transition@self.intercept+self.intercept)-self.c))
        radius = float(np.max(np.abs(eigvals(transition))))
        # Generalized eigenvalues independently check the saddle-path root count.
        n = self.n
        m = np.block([[-self.a0, -self.am], [np.eye(n), np.zeros((n,n))]])
        g = np.block([[self.ap, np.zeros((n,n))], [np.zeros((n,n)), np.eye(n)]])
        roots = eigvals(m, g)
        stable_count = int(np.sum(np.isfinite(roots) & (np.abs(roots) <= 1+1e-7)))
        self.solver_info = dict(iterations=iteration+1, equation_residual=float(residual),
                                affine_residual=float(affine), spectral_radius=radius,
                                stable_generalized_roots=stable_count, states=n)
        if residual > 1e-7 or affine > 1e-7 or radius > 1+1e-7 or stable_count != n:
            raise RuntimeError(f'Invalid RE solution: {self.solver_info}')

    def shock_std(self, period=None):
        std = []
        for shock in self.shocks:
            key = 'std_' + shock
            # The supplied param.m spells this unused name with a trailing _c.
            if shock == 'res_dl_cpi':
                value = self.parameters.get(key, self.parameters['std_res_dl_cpi_c'])
            else:
                value = self.parameters.get(key, 0.0)
            if shock == 'res_l_gdp_tnd' and period is not None and str(period) in ('2022Q1','2022Q2'):
                # Explicit adaptation: allow permanent wartime output destruction.
                value = self.parameters.get('war_potential_std', 10.0)
            std.append(value)
        return np.array(std)

    def observation_frame(self, frame):
        observed = frame[['l_gdp','l_cpi','ir_pol','l_er','ir_ea','l_cpi_ea']].copy()
        foreign = frame.l_gdp_ea.dropna()
        cycle, _ = hpfilter(foreign, lamb=1600)
        observed['l_gdp_ea_gap'] = cycle.reindex(frame.index)
        return observed

    def fit_forecast(self, frame, horizon=9):
        observed = self.observation_frame(frame)
        self.observed = observed
        bar, drift = self.balanced_growth()
        first = frame.iloc[0]
        state = bar.copy()
        for key in ['l_gdp','l_cpi','l_er','l_cpi_ea']:
            value = first[key]
            if not np.isfinite(value): raise ValueError(f'Missing initial {key}')
            state[self.ix[key]] = value-drift[self.ix[key]]
        state[self.ix['l_gdp_tnd']] = state[self.ix['l_gdp']]
        rer = state[self.ix['l_er']]-state[self.ix['l_cpi']]+state[self.ix['l_cpi_ea']]
        state[self.ix['l_rer']] = state[self.ix['l_rer_tnd']] = rer
        state[self.ix['l_er_tar']] = state[self.ix['l_er_uip']] = state[self.ix['l_er']]
        for lag in range(1,4):
            state[self.ix[f'cpi_lag{lag}']] = state[self.ix['l_cpi']]-lag*self.parameters['dl_cpi_tar']/4
        covariance = np.eye(self.n)*100
        history, innovations, covariances = [], [], []
        identity = np.eye(self.n)
        for period, row in observed.iterrows():
            q = (self.impact*self.shock_std(period)) @ (self.impact*self.shock_std(period)).T
            state = self.transition@state+self.intercept
            covariance = self.transition@covariance@self.transition.T+q
            mask = row.notna()
            idx = [self.ix[x] for x in row.index[mask]]
            h = identity[idx]
            innovation = row.loc[mask].to_numpy(float)-h@state
            # Small explicit measurement noise prevents singular identities.
            r = np.eye(len(idx))*1e-6
            s = h@covariance@h.T+r
            gain = solve(s, h@covariance, assume_a='pos').T
            state = state+gain@innovation
            # Joseph form preserves covariance symmetry/positive semidefiniteness.
            a = identity-gain@h
            covariance = a@covariance@a.T+gain@r@gain.T
            covariance = (covariance+covariance.T)/2
            history.append(state.copy()); covariances.append(covariance.copy())
            innovations.append(dict(period=str(period), n_observed=int(mask.sum()),
                                    max_innovation=float(np.max(np.abs(innovation)))))
        self.filtered = pd.DataFrame(history,index=frame.index,columns=self.variables)
        self.last_covariance = covariance.copy()
        self.innovations = innovations
        rows, paths = [], []
        index = pd.period_range(frame.index[-1]+1, periods=horizon, freq='Q')
        q = (self.impact*self.shock_std())@(self.impact*self.shock_std()).T
        for step, period in enumerate(index, 1):
            state = self.transition@state+self.intercept
            covariance = self.transition@covariance@self.transition.T+q
            mean = state[self.ix['ir_pol']]
            std = np.sqrt(max(0,covariance[self.ix['ir_pol'],self.ix['ir_pol']]))
            rows.append(dict(period=str(period),mean=mean,lower80=mean-norm.ppf(.9)*std,
                             upper80=mean+norm.ppf(.9)*std,lower95=mean-norm.ppf(.975)*std,
                             upper95=mean+norm.ppf(.975)*std,horizon=step))
            paths.append(state.copy())
        self.paths = pd.DataFrame(paths,index=index,columns=self.variables)
        return pd.DataFrame(rows), dict(solver=self.solver_info, parameters=self.parameters,
                                        target='quarterly_calendar_day_average_policy_rate',
                                        method='linear_RE_solution_missing_observation_Kalman_filter',
                                        structural_parameters='NBU template calibration, not estimated',
                                        foreign_path='endogenous template AR equations; no future actuals',
                                        intervals='Gaussian conditional on fixed calibration')


if __name__ == '__main__':
    from common import standalone
    standalone('qpm')
