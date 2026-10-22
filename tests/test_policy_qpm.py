"""Mechanical identification/units tests; not empirical validation of policy effects."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'models'))
from common import DataStore
from policy_qpm import PolicyQPM, anticipated_path
from research_qpm import ResearchQPM, anticipated_path as legacy_anticipated_path


class PolicyQPMTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = DataStore()
        cls.frame = cls.store.frame('2026Q3', gdp_lag=1)
        cls.model = PolicyQPM()
        cls.model.fit_forecast(cls.frame, 8)
        cls.initial = cls.model.filtered.iloc[-1].to_numpy()

    def assert_structural_equations(self, states, diagnostics, known=None):
        model = self.model
        unknown = {(row['quarter_offset'], row['control_shock']): row['adjustment']
                   for row in diagnostics['required_shocks']}
        previous = self.initial
        for when, state in enumerate(states):
            following = (states[when+1] if when < len(states)-1
                         else model.transition@state+model.intercept)
            eps = np.zeros(model.k)
            for (offset, shock), value in dict(known or {}).items():
                if offset == when:
                    eps[model.shocks.index(shock)] += value
            for (offset, shock), value in unknown.items():
                if offset == when:
                    eps[model.shocks.index(shock)] += value
            residual = model.am@previous+model.a0@state+model.ap@following-model.c-model.e@eps
            np.testing.assert_allclose(residual, 0., atol=1e-7)
            previous = state

    def test_exact_targets_are_exact_percent_changes(self):
        self.assertAlmostEqual(100*np.expm1(self.model.parameters['dl_cpi_tar']/100), 5., places=12)
        self.assertAlmostEqual(100*np.expm1(self.model.parameters['dl_cpi_ea_tar']/100), 2., places=12)
        self.assertLess(self.model.solver_info['equation_residual'], 1e-7)
        self.assertLess(self.model.solver_info['affine_residual'], 1e-7)

    def test_balanced_growth_preserves_exact_targets_and_zero_gaps(self):
        bar, drift = self.model.balanced_growth()
        states, _ = anticipated_path(self.model, bar, 8)
        for t, state in enumerate(states, 1):
            np.testing.assert_allclose(state, bar+t*drift, atol=2e-7)
        np.testing.assert_allclose(states[:, self.model.ix['l_gdp_gap']], 0., atol=2e-7)
        np.testing.assert_allclose(100*np.expm1(states[:, self.model.ix['d4l_cpi']]/100), 5., atol=1e-7)

    def test_no_constraints_equals_recursive_re_forecast(self):
        states, diagnostic = anticipated_path(self.model, self.initial, 8)
        np.testing.assert_allclose(states, self.model.paths.to_numpy(), atol=2e-6)
        self.assertEqual(diagnostic['n_constraints'], 0)
        self.assert_structural_equations(states, diagnostic)

    def test_rate_control_and_fixed_policy_shock_are_distinct(self):
        target = 16.5
        states, diagnostic = anticipated_path(self.model, self.initial, 8, constraints={0: target})
        self.assertAlmostEqual(states[0, self.model.ix['ir_pol']], target, places=9)
        self.assert_structural_equations(states, diagnostic)
        shock_states, shock_diagnostic = anticipated_path(
            self.model, self.initial, 8, shocks={(0, 'res_ir_tlr'): 1.})
        self.assertEqual(shock_diagnostic['n_constraints'], 0)
        # Simultaneous endogenous terms mean a +1 innovation is not a +1 rate target.
        realized = shock_states[0, self.model.ix['ir_pol']]-self.model.paths.iloc[0].ir_pol
        self.assertGreater(abs(realized-1.), 1e-5)

    def test_joint_foreign_and_rate_paths_satisfy_original_equations(self):
        imposed = {(0, 'ir_ea'): 3.25, (1, 'ir_ea'): 3.,
                   (0, 'dl_cpi_ea'): 2.5, (2, 'l_gdp_ea_gap'): -.5}
        shocks = {(3, 'res_dl_cpi'): 2.}
        states, diagnostic = anticipated_path(self.model, self.initial, 8,
            constraints={0: 16.5, 1: 16.}, shocks=shocks, variable_paths=imposed)
        for (when, variable), target in imposed.items():
            self.assertAlmostEqual(states[when, self.model.ix[variable]], target, places=9)
        self.assert_structural_equations(states, diagnostic, shocks)
        self.assertLess(diagnostic['constraint_residual'], 1e-7)

    def test_future_anticipated_foreign_news_changes_earlier_endogenous_fx(self):
        target = self.model.paths.iloc[3].ir_ea+1.
        states, _ = anticipated_path(self.model, self.initial, 8,
                                     variable_paths={(3, 'ir_ea'): target})
        self.assertGreater(abs(states[0, self.model.ix['l_er']]-self.model.paths.iloc[0].l_er), 1e-6)

    def test_fixed_shock_and_endogenous_control_report_total_separately(self):
        known = {(0, 'res_ir_tlr'): .4}
        states, diagnostic = anticipated_path(self.model, self.initial, 8,
                                               constraints={0: 16.}, shocks=known)
        row = diagnostic['required_shocks'][0]
        self.assertAlmostEqual(row['total_shock'], row['adjustment']+.4, places=12)
        self.assert_structural_equations(states, diagnostic, known)

    def test_default_gap_is_noisy_expert_observation_not_exact_constraint(self):
        observed = self.model.observed
        self.assertEqual(observed.loc['2022Q2', 'l_gdp_gap'], -18.)
        self.assertEqual(observed.l_gdp_gap.notna().sum(), 1)
        diagnostic = self.model.gap_judgment_diagnostics[0]
        self.assertEqual(diagnostic['measurement_std'], 1.)
        self.assertAlmostEqual(diagnostic['measurement_minus_filtered_gap'],
                               -18.-diagnostic['filtered_gap'])
        self.assertAlmostEqual(diagnostic['measurement_misfit_in_declared_std'],
                               diagnostic['measurement_minus_filtered_gap']/diagnostic['measurement_std'])
        self.assertGreater(diagnostic['filtered_gap_std'], 0.)
        # Posterior is a weighted update, not an overwrite to an asserted truth.
        self.assertGreater(abs(diagnostic['filtered_gap']+18.), 1e-5)
        self.assertTrue(np.isfinite(self.model.last_covariance).all())
        self.assertGreaterEqual(np.linalg.eigvalsh(self.model.last_covariance).min(), -1e-7)

    def test_current_disturbance_posteriors_are_finite_with_bounded_uncertainty(self):
        self.assertTrue(self.model.filtered_shocks.index.equals(self.frame.index))
        self.assertEqual(list(self.model.filtered_shocks), self.model.shocks)
        self.assertTrue(np.isfinite(self.model.filtered_shocks.to_numpy()).all())
        self.assertTrue(np.isfinite(self.model.filtered_shock_std.to_numpy()).all())
        for period, row in self.model.filtered_shock_std.iterrows():
            self.assertTrue((row.to_numpy() >= 0.).all())
            self.assertTrue((row.to_numpy() <= self.model.shock_std(period)+1e-8).all())
        zero_innovation = self.model.shocks.index('res_l_rer_tnd')
        np.testing.assert_array_equal(self.model.filtered_shocks.iloc[:, zero_innovation], 0.)

    def test_sigma_limits_control_strength_of_judgment(self):
        weak = PolicyQPM(gap_judgments={'2022Q2': (-18., 1e6)})
        strong = PolicyQPM(gap_judgments={'2022Q2': {'value': -18., 'std': .01}})
        no_judgment = PolicyQPM(gap_judgments={})
        for model in (weak, strong, no_judgment):
            model.fit_forecast(self.frame, 1)
        weak_gap = weak.filtered.loc['2022Q2', 'l_gdp_gap']
        strong_gap = strong.filtered.loc['2022Q2', 'l_gdp_gap']
        no_gap = no_judgment.filtered.loc['2022Q2', 'l_gdp_gap']
        self.assertLess(abs(strong_gap+18.), abs(weak_gap+18.))
        self.assertAlmostEqual(weak_gap, no_gap, places=6)

    def test_future_gap_judgment_does_not_enter_earlier_origin(self):
        frame = self.store.frame('2021Q4', gdp_lag=1)
        with_default, without = PolicyQPM(), PolicyQPM(gap_judgments={})
        with_default.fit_forecast(frame, 1)
        without.fit_forecast(frame, 1)
        np.testing.assert_array_equal(with_default.filtered.to_numpy(), without.filtered.to_numpy())
        np.testing.assert_array_equal(with_default.paths.to_numpy(), without.paths.to_numpy())
        self.assertEqual(with_default.gap_judgment_diagnostics, [])

    def test_potential_std_variants_change_only_declared_war_dates(self):
        low = PolicyQPM(settings={'war_potential_std': 0.})
        high = PolicyQPM(settings={'war_potential_std': 30.})
        index = low.shocks.index('res_l_gdp_tnd')
        self.assertEqual(low.shock_std(pd.Period('2022Q2'))[index], 0.)
        self.assertEqual(high.shock_std(pd.Period('2022Q2'))[index], 30.)
        np.testing.assert_array_equal(low.shock_std(pd.Period('2022Q3')),
                                      high.shock_std(pd.Period('2022Q3')))
        for model in (low, high):
            model.fit_forecast(self.frame, 1)
            self.assertTrue(np.isfinite(model.filtered.to_numpy()).all())

    def test_explicit_legacy_mode_matches_parent_without_mutating_it(self):
        parent = ResearchQPM()
        adapted = PolicyQPM(exact_targets=False, gap_judgments={})
        parent.fit_forecast(self.frame, 2)
        adapted.fit_forecast(self.frame, 2)
        np.testing.assert_array_equal(parent.transition, adapted.transition)
        np.testing.assert_allclose(parent.filtered.to_numpy(), adapted.filtered.to_numpy(), atol=1e-10, rtol=0)
        np.testing.assert_allclose(parent.paths.to_numpy(), adapted.paths.to_numpy(), atol=1e-10, rtol=0)
        self.assertEqual(parent.parameters['dl_cpi_tar'], 5.)
        newer, _ = anticipated_path(parent, parent.filtered.iloc[-1].to_numpy(), 2,
                                    constraints={0: 16.})
        older, _ = legacy_anticipated_path(parent, parent.filtered.iloc[-1].to_numpy(), 2,
                                           constraints={0: 16.})
        np.testing.assert_allclose(newer, older, atol=1e-10)

    def test_bad_controls_offsets_and_uncertainty_fail_explicitly(self):
        for judgments in ({'2022Q2': (-18., 0.)}, {'2022Q2': (-18., -1.)}):
            with self.assertRaises(ValueError):
                PolicyQPM(gap_judgments=judgments)
        for kwargs in (
            {'constraints': {8: 16.}},
            {'shocks': {(-1, 'res_dl_cpi'): 1.}},
            {'variable_paths': {(0, 'not_a_variable'): 1.}},
            {'variable_paths': {(0, 'l_gdp'): 1.}},
            {'constraints': {0: 16.}, 'variable_paths': {(0, 'ir_pol'): 17.}},
            {'variable_paths': {(0, 'ir_ea'): 2., (0, 'dl_cpi_ea'): 2.},
             'shock_controls': {'dl_cpi_ea': 'res_ir_ea'}},
        ):
            with self.assertRaises(ValueError):
                anticipated_path(self.model, self.initial, 8, **kwargs)


if __name__ == '__main__':
    unittest.main()
