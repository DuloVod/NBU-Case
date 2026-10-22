"""Selection protocol tests without rerunning the full historical model grid."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'models'))
import policy_calibration as calibration


class PolicyCalibrationTest(unittest.TestCase):
    def score_rows(self):
        return pd.DataFrame([
            dict(origin='2020Q1', metric='inflation_yoy_pct', error=.5),
            dict(origin='2020Q1', metric='gdp_log_percent', error=4.),
            dict(origin='2020Q1', metric='fx_log_percent', error=12.),
            dict(origin='2020Q1', metric='rate_pp', error=-99999.),
        ])

    def test_score_is_exactly_three_outcomes_and_excludes_rate(self):
        rows = self.score_rows()
        scales = {'inflation_yoy_pct': 1., 'gdp_log_percent': 2., 'fx_log_percent': 3.}
        # Standardized magnitudes .5, 2, 4 -> Huber .125, 2, 6.
        expected = (.125+2.+6.)/3
        self.assertAlmostEqual(calibration.predictive_score(rows, scales), expected)
        for rate_error in (-1e100, 0., 1e100):
            changed = rows.copy()
            changed.loc[changed.metric.eq('rate_pp'), 'error'] = rate_error
            self.assertAlmostEqual(calibration.predictive_score(changed, scales), expected)
        no_rate = rows.loc[rows.metric.ne('rate_pp')]
        self.assertAlmostEqual(calibration.predictive_score(no_rate, scales), expected)
        with self.assertRaises(ValueError):
            calibration.predictive_score(rows.loc[rows.metric.ne('fx_log_percent')], scales)

    def test_grid_has_nine_effective_pairs_and_one_default(self):
        settings = calibration.candidate_settings()
        effective = [(row.get('beta', .7), row.get('inflation_memory', .4)) for row in settings]
        self.assertEqual(settings[0], {})
        self.assertEqual(len(settings), 9)
        self.assertEqual(len(set(effective)), 9)
        self.assertEqual(set(effective), {(b, m) for b in (.5, .7, .9) for m in (.3, .4, .6)})
        self.assertEqual(calibration.REQUIRED_VALIDATION_GAIN, .05)
        self.assertEqual(calibration.HUBER_CUTOFF, 2.)
        self.assertEqual(calibration.SCALE_FLOOR, 1.)

    def test_prediction_and_label_annual_lags_use_different_correct_offsets(self):
        class Model:
            paths = pd.DataFrame([dict(l_cpi=9., l_gdp=12., l_er=20., ir_pol=16.)])

        training = pd.DataFrame({'l_cpi': np.arange(1., 9.)})
        predicted = calibration.prediction_vector(Model(), training)
        # Future quarter9 minus historical quarter5 = 4 log points.
        self.assertAlmostEqual(predicted['inflation_yoy_pct'], 100*np.expm1(.04))

        class Labels:
            def frame(self, period, gdp_lag):
                self.last_call = (period, gdp_lag)
                return pd.DataFrame({'l_cpi': np.arange(1., 10.),
                                     'l_gdp': 12., 'l_er': 20., 'ir_pol': 16.})

        labels = Labels()
        target = pd.Period('2021Q1', freq='Q')
        actual = calibration.target_vector(labels, target)
        self.assertEqual(labels.last_call, (target, 0))
        # In target-inclusive labels, annual lag is the fifth row from the end.
        self.assertAlmostEqual(actual['inflation_yoy_pct'], predicted['inflation_yoy_pct'])

    def mocked_run(self, best_score=.951, audit_label_shift=0.):
        events, configurations = [], []

        class Store:
            def frame(self, origin, gdp_lag):
                if gdp_lag != 1:
                    raise AssertionError('Training construction must request GDP lag one')
                events.append(('training', str(origin)))
                return pd.DataFrame({'l_cpi': np.zeros(8)},
                    index=pd.period_range(origin-7, origin, freq='Q'))

        class Model:
            def __init__(self, settings, gap_judgments, exact_targets):
                configurations.append((dict(settings), gap_judgments, exact_targets))
                self.settings = settings

            def fit_forecast(self, frame, horizon):
                if horizon != 1:
                    raise AssertionError('Calibration must remain one-quarter predictive')
                origin = frame.index[-1]
                events.append(('prediction', str(origin)))
                effective = (self.settings.get('beta', .7), self.settings.get('inflation_memory', .4))
                if origin <= pd.Period('2019Q4'):
                    inflation_error, gdp_error, fx_error = .5, 2., 3.
                else:
                    # Candidate beta=.5,memory=.3 is the unique validation best.
                    score = best_score if effective == (.5, .3) else 1.
                    magnitude = np.sqrt(2*score)
                    inflation_error, gdp_error, fx_error = magnitude, 2*magnitude, 3*magnitude
                self.paths = pd.DataFrame([dict(
                    l_cpi=100*np.log1p(inflation_error/100),
                    l_gdp=gdp_error, l_er=fx_error, ir_pol=10.)])

        def labels(store, period):
            events.append(('label', str(period)))
            # First audit target is 2023Q2. Validation ends with target 2023Q1.
            shift = audit_label_shift if period >= pd.Period('2023Q2') else 0.
            return {metric: shift for metric in calibration.DIAGNOSTIC_METRICS}

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(calibration, 'PolicyQPM', Model), \
                 patch.object(calibration, 'target_vector', labels), \
                 contextlib.redirect_stdout(io.StringIO()):
                settings, protocol = calibration.calibrate(Store(), Path(directory))
            predictions = pd.read_csv(Path(directory)/'policy_calibration_predictions.csv')
            grid = pd.read_csv(Path(directory)/'policy_calibration_grid.csv')
            persisted = json.loads((Path(directory)/'policy_calibration_protocol.json').read_text())
            self.assertEqual(persisted['selected_settings'], settings)
        return settings, protocol, predictions, grid, events, configurations

    def test_development_scales_and_five_percent_threshold_are_used(self):
        retained, protocol, predictions, grid, _, configurations = self.mocked_run(best_score=.951)
        self.assertEqual(retained, {})
        self.assertAlmostEqual(protocol['validation_improvement_fraction'], .049, places=12)
        self.assertEqual(protocol['scales_from_corrected_template_development_rmse'],
                         {'inflation_yoy_pct': 1., 'gdp_log_percent': 2., 'fx_log_percent': 3.})
        adopted, protocol, *_ = self.mocked_run(best_score=.949)
        self.assertEqual(adopted, {'beta': .5, 'inflation_memory': .3})
        self.assertAlmostEqual(protocol['validation_improvement_fraction'], .051, places=12)
        boundary, protocol, *_ = self.mocked_run(best_score=.95)
        self.assertEqual(boundary, {'beta': .5, 'inflation_memory': .3})
        self.assertAlmostEqual(protocol['validation_improvement_fraction'], .05, places=12)
        self.assertTrue(all(judgments == {} and exact for _, judgments, exact in configurations))
        self.assertEqual(len(predictions), 9*(17+12+13)*4)
        self.assertEqual(len(grid), 9)

    def test_audit_labels_cannot_change_selected_settings_or_validation_scores(self):
        baseline = self.mocked_run(best_score=.949, audit_label_shift=0.)
        changed = self.mocked_run(best_score=.949, audit_label_shift=1e6)
        self.assertEqual(baseline[0], changed[0])
        self.assertEqual(baseline[1]['selected_candidate'], changed[1]['selected_candidate'])
        self.assertEqual(baseline[1]['scales_from_corrected_template_development_rmse'],
                         changed[1]['scales_from_corrected_template_development_rmse'])
        np.testing.assert_array_equal(baseline[3].validation_score, changed[3].validation_score)
        self.assertGreater(abs(changed[2].loc[changed[2].split.eq('audit'), 'actual'].iloc[0]
                               - baseline[2].loc[baseline[2].split.eq('audit'), 'actual'].iloc[0]), 1e5)

    def test_all_split_predictions_exist_before_its_labels_are_attached(self):
        _, protocol, _, _, events, _ = self.mocked_run()
        for split in ('validation', 'audit'):
            first, last = calibration.SPLITS[split]
            origins = set(map(str, pd.period_range(first, last, freq='Q')))
            targets = set(map(str, pd.period_range(pd.Period(first)+1, pd.Period(last)+1, freq='Q')))
            predicted_at = [i for i, (kind, value) in enumerate(events)
                            if kind == 'prediction' and value in origins]
            labeled_at = [i for i, (kind, value) in enumerate(events)
                          if kind == 'label' and value in targets]
            self.assertEqual(len(predicted_at), len(origins)*9)
            self.assertEqual(len(labeled_at), len(targets))
            self.assertLess(max(predicted_at), min(labeled_at))
        train_events = [event for event in events if event[0] == 'training']
        self.assertEqual(len(train_events), 17+12+13)
        self.assertEqual(protocol['splits']['audit']['last_target'], '2026Q2')
        self.assertFalse(protocol['rate_used_in_selection'])


if __name__ == '__main__':
    unittest.main()
