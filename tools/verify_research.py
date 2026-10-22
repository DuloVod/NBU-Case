"""Independent offline audit of saved research; does not rebuild or edit outputs."""
from pathlib import Path
from urllib.parse import urlparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/2026-10-22-research"
FAILURES = []
CHECKS = 0


def check(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(message)


def close(actual, expected, message, tolerance=1e-9):
    check(bool(np.isclose(actual, expected, atol=tolerance, rtol=tolerance, equal_nan=False)), message)


def load(name):
    return pd.read_csv(RESULT / name)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def monthly_audit():
    predictions = load("monthly_predictions.csv")
    metrics = load("monthly_metrics.csv")
    november = load("monthly_november_forecast.csv")
    protocol = read_json(RESULT / "monthly_protocol.json")
    check(len(predictions) == 92 * 12, "Monthly prediction count must be 92 x 12")
    check(predictions.model.nunique() == 12 and predictions.origin.nunique() == 92,
          "Monthly model/origin counts differ")
    check(len(metrics) == 24 and not metrics.duplicated(["split", "model"]).any(), "Monthly metric cases differ")
    check(not predictions.duplicated(["model", "origin"]).any(), "Duplicate monthly origin/model")
    check(predictions.groupby("origin").model.nunique().eq(12).all(), "Different model cases by origin")
    check(predictions.groupby("period").actual.nunique().eq(1).all(), "Actual targets differ across models")
    check(predictions.groupby("origin").baseline.nunique().eq(1).all(), "Baseline differs across models")
    origins = pd.PeriodIndex(predictions.origin, freq="M")
    check((pd.PeriodIndex(predictions.period, freq="M") == origins + 1).all(), "Monthly target must equal origin + 1")
    check((pd.PeriodIndex(predictions.training_label_end, freq="M") == origins - 1).all(),
          "Monthly labels must stop at origin - 1")
    check((predictions.as_of == predictions.origin + "-22").all(), "Historical origin day differs from 22")
    check(predictions.groupby("split").origin.nunique().to_dict() == {"test": 24, "validation": 68},
          "Chronological split size differs")
    expected_origins = pd.period_range("2019-01", "2026-08", freq="M").astype(str)
    check(sorted(predictions.origin.unique()) == list(expected_origins), "Monthly origin calendar differs")
    numeric = ["prediction", "actual", "baseline"]
    check(np.isfinite(predictions[numeric].to_numpy()).all(), "Missing/nonfinite historical predictions")

    daily = pd.read_csv(ROOT / "results/2026-10-22-monthly/policy_daily_as_of.csv", parse_dates=["date"])
    actuals = daily.groupby(daily.date.dt.to_period("M")).rate.mean()
    mapped = predictions.period.map(lambda period: actuals.loc[pd.Period(period, freq="M")])
    check(np.allclose(predictions.actual, mapped, atol=1e-10, rtol=1e-10), "Targets disagree with independent daily policy averages")
    rates = daily.set_index(daily.date.dt.strftime("%Y-%m-%d")).rate
    check(np.allclose(predictions.baseline, predictions.as_of.map(rates), atol=1e-10),
          "Baseline differs from observed effective origin rate")

    for (split, model), group in predictions.groupby(["split", "model"]):
        saved = metrics.loc[(metrics.split == split) & (metrics.model == model)]
        check(len(saved) == 1, f"Missing/duplicate metric {split}/{model}")
        if len(saved) != 1:
            continue
        row = saved.iloc[0]
        error = group.prediction - group.actual
        close(row.n, len(group), f"Monthly n {split}/{model}")
        for name, expected in {"MAE_pp": error.abs().mean(), "RMSE_pp": np.sqrt(np.mean(error ** 2)),
                               "bias_pp": error.mean()}.items():
            close(row[name], expected, f"Recomputed monthly {name} {split}/{model}")
        for level in (80, 95):
            check(group[f"lower{level}"].isna().equals(group[f"upper{level}"].isna()),
                  f"Only one interval bound missing {level} {split}/{model}")
            valid = group.dropna(subset=[f"lower{level}", f"upper{level}"])
            close(row[f"interval_n{level}"], len(valid), f"Interval n {level} {split}/{model}")
            coverage = ((valid.actual >= valid[f"lower{level}"]) & (valid.actual <= valid[f"upper{level}"])).mean()
            close(row[f"coverage{level}"], coverage, f"Coverage {level} {split}/{model}")
            close(row[f"mean_width{level}_pp"], (valid[f"upper{level}"] - valid[f"lower{level}"]).mean(),
                  f"Interval width {level} {split}/{model}")
    winner = metrics.loc[metrics.split == "validation"].sort_values(["MAE_pp", "RMSE_pp", "model"]).iloc[0].model
    check(winner == protocol["selected_model"], "Validation winner disagrees with protocol")
    check(len(november) == 12 and november.model.nunique() == 12, "Final model count differs")
    check(set(november.model) == set(protocol["candidates"]) == set(predictions.model), "Final/validated candidate sets differ")
    selected = november.loc[november.selected_by_validation.astype(str).str.lower() == "true"]
    check(len(selected) == 1 and selected.iloc[0].model == winner, "Final selected November model differs")
    check(november.period.eq("2026-11").all() and november.origin.eq("2026-10").all(), "Wrong final forecast month")
    check(november.training_label_end.eq("2026-09").all(), "Final forecast used unresolved October label")
    check(november.actual.isna().all(), "November actual must remain unknown")
    if winner == "NO_CHANGE":
        close(selected.iloc[0].prediction, 16, "Selected no-change November forecast differs")
    summary = read_json(RESULT / "research_summary.json")
    check(summary["forecast_model"] == winner, "Research summary winner differs")
    close(summary["november_average_forecast"], selected.iloc[0].prediction, "Summary forecast differs")


def qpm_audit():
    grid = load("qpm_calibration_grid.csv")
    protocol = read_json(RESULT / "qpm_calibration_protocol.json")
    predictions = load("qpm_calibration_predictions.csv")
    metrics = load("qpm_calibration_metrics.csv")
    check(len(grid) == 109 and set(grid.candidate) == set(range(109)), "QPM grid must contain 109 unique candidates")
    check(protocol["candidate_count"] == len(grid), "QPM candidate count disagrees with protocol")
    check(len(metrics) == 36 and not metrics.duplicated(["model", "split", "metric"]).any(), "QPM metric cases differ")
    default = grid.loc[grid.candidate == 0].iloc[0]
    best = grid.sort_values(["score", "candidate"]).iloc[0]
    expected_selected = int(best.candidate) if best.score < .95 * default.score else 0
    close(protocol["default_score"], default.score, "QPM default score differs")
    close(protocol["best_validation_score"], best.score, "QPM best validation score differs")
    check(protocol["selected_candidate"] == expected_selected, "QPM 5% selection gate violated")
    selected_settings = json.loads(grid.loc[grid.candidate == expected_selected, "settings"].iloc[0])
    check(protocol["selected_settings"] == selected_settings, "QPM selected settings differ")
    close(protocol["validation_improvement_fraction"], 1 - best.score / default.score, "QPM improvement differs")
    check((pd.PeriodIndex(predictions.period, freq="Q") == pd.PeriodIndex(predictions.origin, freq="Q") + 1).all(),
          "QPM target must equal origin + 1 quarter")
    check(np.allclose(predictions.error, predictions.prediction - predictions.actual, equal_nan=True), "QPM saved errors differ")
    check(not predictions.duplicated(["model", "split", "origin", "metric"]).any(), "Duplicate QPM prediction rows")
    for (model, split, metric), group in predictions.groupby(["model", "split", "metric"]):
        saved = metrics.loc[(metrics.model == model) & (metrics.split == split) & (metrics.metric == metric)]
        check(len(saved) == 1, f"Missing QPM metric {model}/{split}/{metric}")
        if len(saved) != 1:
            continue
        row = saved.iloc[0]
        error = group.error.dropna()
        close(row.n, len(error), f"QPM valid n {model}/{split}/{metric}")
        close(row.n_origins, group.origin.nunique(), f"QPM origin count {model}/{split}/{metric}")
        for name, value in {"MAE": error.abs().mean(), "RMSE": np.sqrt(np.mean(error ** 2)), "bias": error.mean()}.items():
            close(row[name], value, f"QPM recomputed {name} {model}/{split}/{metric}")
        if split == "audit":
            close(row.n_origins, 14, f"QPM audit origin count {model}/{metric}")
            close(row.n, 13 if metric == "gdp_log_percent" else 14, f"QPM audit valid count {model}/{metric}")
    unknown_gdp = predictions.loc[(predictions.metric == "gdp_log_percent") & (predictions.period == "2026Q3")]
    check(len(unknown_gdp) == 3 and unknown_gdp.actual.isna().all(), "GDP 2026Q3 must not be an observed target")

    paths = load("qpm_policy_paths.csv")
    first = paths.loc[paths.period == "2026Q4"]
    for policy, decision in {"cut_15_5": 15.5, "hold_16": 16., "hike_16_5": 16.5}.items():
        group = first.loc[first.policy == policy]
        check(len(group) > 0, f"Missing first-step policy {policy}")
        expected = (29 * 16 + 63 * decision) / 92
        check(np.allclose(group.ir_pol, expected, atol=1e-9), f"Q4 calendar-day policy mean {policy}")
    check(np.allclose(paths.inflation_yoy_pct, 100 * np.expm1(paths.d4l_cpi / 100), atol=1e-9),
          "QPM inflation ordinary-percent conversion differs")
    check(np.allclose(paths.fx_uah_per_eur, np.exp(paths.l_er / 100), atol=1e-9), "QPM EUR exchange-rate conversion differs")
    losses = load('qpm_policy_losses.csv')
    winners = load('qpm_loss_sensitivity.csv')
    check(set(np.round(losses.inflation_target_pct, 6)) == {5., round(100*np.expm1(.05),6)},
          'Missing logarithmic-price-target loss sensitivity')
    check('exact_price_targets' in set(paths.variant), 'Missing exact price-target model variant')
    information = load('qpm_research_information.csv')
    last_rate = float(information.ir_pol.iloc[-1])
    path_groups = {key:g.sort_values('period') for key,g in paths.groupby(['variant','shock_case','policy'])}
    for row in losses.itertuples():
        group = path_groups[(row.variant,row.shock_case,row.policy)]
        changes = np.diff(np.r_[last_rate,group.ir_pol])
        value = (group.inflation_yoy_pct-row.inflation_target_pct)**2 + row.output_weight*group.l_gdp_gap**2 + row.smoothing_weight*changes**2
        expected = float(np.average(value,weights=.99**np.arange(len(group))))
        close(row.discounted_loss,expected,f'Policy loss {row.variant}/{row.shock_case}/{row.policy}')
    keys = ['variant','shock_case','output_weight','smoothing_weight','inflation_target_pct']
    decisions = losses.loc[losses.policy!='endogenous_rule']
    optimal = decisions.loc[decisions.groupby(keys).discounted_loss.idxmin()].sort_values(keys).reset_index(drop=True)
    saved = winners.sort_values(keys).reset_index(drop=True)
    check(optimal.policy.tolist()==saved.policy.tolist() and len(optimal)==len(saved), 'Loss-grid winners differ')


def evidence_audit():
    folder = ROOT / "data/research/2026-10-22"
    evidence = pd.read_csv(folder / "evidence_long.csv", keep_default_na=False)
    latest = pd.read_csv(folder / "latest_evidence.csv", keep_default_na=False)
    manifest = read_json(folder / "manifest.json")
    required = {"date", "variable", "value", "unit", "source_url", "published_at", "status", "frequency", "source_file", "locator", "availability", "note"}
    check(set(evidence.columns) == required, "Evidence schema differs")
    check(len(evidence) == manifest["rows"] and evidence.variable.nunique() == manifest["variables"], "Evidence manifest counts differ")
    check(len(latest) == evidence.variable.nunique() and latest.variable.is_unique, "Latest evidence variable count differs")
    check(evidence.published_at.ne("").sum() == manifest["publication_date_known_rows"], "Known publication count differs")
    check(np.isfinite(pd.to_numeric(evidence.value)).all(), "Evidence contains nonfinite numeric values")
    check(not evidence.duplicated(["date", "variable", "source_file"]).any(), "Evidence duplicate observations")
    check(evidence.source_url.map(lambda u: urlparse(u).scheme in {"https", "http"} and bool(urlparse(u).netloc)).all(), "Evidence contains invalid source URLs")
    for _, row in evidence.iterrows():
        check(not row.published_at or row.published_at[:10] <= manifest["as_of"], f"Future evidence publication {row.variable}/{row.date}")
        freq = "Q" if "Q" in row.date else "D" if len(row.date) == 10 else "M"
        end = pd.Period(row.date, freq=freq).end_time.date().isoformat()
        check(end <= manifest["as_of"], f"Future/incomplete evidence period {row.variable}/{row.date}")
    for name in ("evidence_long", "latest_evidence"):
        path = folder / "excel_exports" / (name + "_excel.csv")
        check(path.read_bytes().startswith(b"\xef\xbb\xbf"), f"Excel export lacks UTF-8 BOM: {name}")
        excel = pd.read_csv(path, encoding="utf-8-sig", sep=";", keep_default_na=False, decimal=",")
        machine = evidence if name == "evidence_long" else latest
        check(len(excel) == len(machine) and list(excel.columns) == list(machine.columns), f"Excel export cells/count differ: {name}")
        check(np.allclose(pd.to_numeric(excel.value.astype(str).str.replace(",", ".", regex=False)), pd.to_numeric(machine.value)), f"Excel decimal values differ: {name}")


def hash_audit():
    manifests = [(ROOT / "data/research/2026-10-22/manifest.json", "inputs_sha256"),
                 (RESULT / "monthly_protocol.json", "hashes"),
                 (RESULT / "qpm_research_manifest.json", "sha256"),
                 (RESULT / 'research_run_manifest.json', 'input_sha256'),
                 (RESULT / 'research_run_manifest.json', 'output_sha256')]
    for path, key in manifests:
        entries = read_json(path)[key]
        check(bool(entries), f"Empty input hashes in {path.name}")
        for relative, expected in entries.items():
            source = ROOT / relative
            check(source.is_file(), f"Missing hashed input {relative}")
            if source.is_file():
                actual = hashlib.sha256(source.read_bytes()).hexdigest()
                check(actual == expected, f"SHA-256 mismatch: {relative} ({path.name}); reproduce outputs after source changes")


def main():
    for audit in (monthly_audit, qpm_audit, evidence_audit, hash_audit):
        audit()
    if FAILURES:
        print(f"Research audit FAILED: {len(FAILURES)} failures / {CHECKS} checks", file=sys.stderr)
        for message in FAILURES:
            print("- " + message, file=sys.stderr)
        return 1
    print(f"Research audit PASSED: {CHECKS} checks; 92 x 12 forecasts, metrics, QPM calibration, calendar means, evidence and SHA-256.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
