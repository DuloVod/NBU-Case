"""Shared information set, preprocessing, and command-line entry points."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

ROOT = Path(__file__).resolve().parents[1]


def read_periods(path, frequency):
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.PeriodIndex(frame.index, freq=frequency)
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError(f'Duplicate/unordered index: {path}')
    return frame


def quarter_mean(series):
    grouped = series.groupby(series.index.asfreq('Q'))
    return grouped.mean().where(grouped.count() == 3)


class DataStore:
    def __init__(self, path=None):
        self.path = Path(path or ROOT/'data')
        self.monthly = read_periods(self.path/'inputs/core_monthly.csv', 'M')
        self.levels = read_periods(self.path/'model/quarterly_levels.csv', 'Q')
        self.aux_monthly = read_periods(self.path/'model/auxiliary_monthly.csv','M')
        self.aux_quarterly = read_periods(self.path/'model/auxiliary_quarterly.csv','Q')
        self.target = self.levels.ir_pol.loc['2010Q1':].dropna()
        self.cache = {}

    @property
    def latest(self):
        return self.target.index[-1]

    def frame(self, origin, gdp_lag=1):
        """As-of origin quarter end +21d; GDP available through origin-1.

        Current-vintage GDP/FX and raw CPI are truncated BEFORE adjustment.
        No future normalization base, seasonal factor, GDP, or policy decision
        enters an earlier fold. Revised source vintages remain a limitation.
        """
        origin = pd.Period(origin, freq='Q')
        key = (origin, gdp_lag)
        if key in self.cache: return self.cache[key].copy()
        month_end = origin.asfreq('M', how='end')
        month = self.monthly.loc[:month_end].copy()
        q = self.levels.loc[:origin].copy()
        # A fixed first observation anchor, never a base year in a fold's future.
        for source, target in [('CPI_UA_INDEX_NSA_2021_100','cpi'),('HICP_EA_NSA','cpi_ea')]:
            observed = month[source].dropna()
            if not observed.index.equals(pd.period_range(observed.index[0],observed.index[-1],freq='M')):
                raise ValueError(f'Price history gap: {source}')
            logged = np.log(observed.to_numpy(float))
            adjustment = STL(logged,period=12,seasonal=13,trend=21,low_pass=13,robust=True).fit()
            sa = pd.Series(np.exp(logged-adjustment.seasonal),index=observed.index)
            sa = sa/sa.iloc[0]*100
            q[target] = quarter_mean(sa).reindex(q.index)
        # Rebuild average policy/external rates & FX from unadjusted monthly input.
        for source,target in [('IR_POLICY_CALENDAR_AVG','ir_pol'),('IR_ECB_DFR_CALENDAR_AVG','ir_ea')]:
            days = pd.Series(month.index.days_in_month,index=month.index,dtype=float)
            valid_days = days.where(month[source].notna())
            weights = valid_days.groupby(month.index.asfreq('Q')).sum()
            totals = (month[source]*days).groupby(month.index.asfreq('Q')).sum(min_count=3)
            recomputed = (totals/weights).reindex(q.index)
            available = q[target].notna()&recomputed.notna()
            if not np.allclose(recomputed[available],q.loc[available,target],atol=1e-8,rtol=0):
                raise ValueError(f'Rate aggregation mismatch: {source}')
            q[target] = recomputed
        q['er'] = quarter_mean(month.ER_UAH_EUR_AVG).reindex(q.index)
        q.loc[q.index > origin-gdp_lag,['gdp','gdp_ea']] = np.nan
        frame = pd.DataFrame(index=q.index)
        for variable in ['gdp','cpi','er','gdp_ea','cpi_ea']:
            frame['l_'+variable] = 100*np.log(q[variable])
        frame['ir_pol'],frame['ir_ea'] = q.ir_pol,q.ir_ea
        frame = frame.loc['2010Q1':origin]
        if frame.ir_pol.isna().any(): raise ValueError('Missing policy-rate target in history')
        self.cache[key] = frame.copy()
        return frame

    def inventory(self):
        """All prepared tables are inspected, but forecasts stay out of training."""
        rows = []
        roles = {
            'quarterly_training.csv':'reference only; SA rebuilt inside every fold',
            'quarterly_levels.csv':'GDP, policy rate, FX and foreign rate source',
            'monthly.csv':'reference only; raw monthly inputs preferred',
            'auxiliary_monthly.csv':'context and optional extensions, not forced into 7-variable core',
            'auxiliary_quarterly.csv':'context and optional extensions',
            'nbu_benchmarks.csv':'comparison only; excluded from model fitting',
            'nbu_external_assumptions.csv':'optional scenario reference; excluded from automatic fitting',
            'external_path_template.csv':'unfilled future assumptions; not treated as observations',
            'surveys_monthly.csv':'expectations context, not realized inflation',
            'surveys_quarterly.csv':'expectations context',
            'business_monthly.csv':'activity context',
            'business_quarterly_latest.csv':'latest survey context, no invented long history',
        }
        for path in sorted((self.path/'model').glob('*.csv')):
            table = pd.read_csv(path)
            rows.append(dict(file=path.name,rows=len(table),columns=len(table.columns),
                             numeric_cells=int(table.select_dtypes('number').count().sum()),
                             role=roles.get(path.name,'supporting prepared data / audit'),
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        return pd.DataFrame(rows)

    def context(self):
        rows = []
        for frequency,frame in [('monthly',self.aux_monthly),('quarterly',self.aux_quarterly)]:
            for column in frame:
                series = frame[column].dropna()
                if len(series):
                    rows.append(dict(series=column,frequency=frequency,first=str(series.index[0]),
                                     last=str(series.index[-1]),latest_value=float(series.iloc[-1]),n=len(series),
                                     role='auxiliary context; not an automatically estimated QPM equation'))
        return pd.DataFrame(rows)


def fit_model(name, frame, horizon):
    module = importlib.import_module(name)
    model = getattr(module, {'qpm':'QPM','arima':'ARIMAModel','sarima':'SARIMAModel','var':'VARModel'}[name])()
    forecast, metadata = model.fit_forecast(frame,horizon)
    if len(forecast) != horizon or not np.isfinite(forecast[['mean','lower80','upper80','lower95','upper95']].to_numpy()).all():
        raise ValueError(f'{name}: invalid forecast')
    if not ((forecast.lower95 <= forecast.lower80)&(forecast.lower80 <= forecast['mean'])&
            (forecast['mean'] <= forecast.upper80)&(forecast.upper80 <= forecast.upper95)).all():
        raise ValueError(f'{name}: crossed prediction bands')
    return model,forecast,metadata


def write_json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str,allow_nan=False))


def standalone(name):
    parser = argparse.ArgumentParser(description=f'Forecast quarterly average NBU policy rate with {name.upper()}')
    parser.add_argument('--data',type=Path,default=ROOT/'data')
    parser.add_argument('--output',type=Path,default=ROOT/'results'/'2026-10-21'/name)
    parser.add_argument('--origin',default=None)
    parser.add_argument('--horizon',type=int,default=9)
    args = parser.parse_args()
    store = DataStore(args.data)
    origin = pd.Period(args.origin or store.latest,freq='Q')
    args.output.mkdir(parents=True,exist_ok=True)
    model,forecast,meta = fit_model(name,store.frame(origin),args.horizon)
    forecast.to_csv(args.output/'forecast.csv',index=False)
    meta.update(origin=str(origin),information_date=str(origin.end_time.date()+pd.Timedelta(days=21)),
                units='percent_per_annum',target='quarterly_calendar_day_average_policy_rate')
    write_json(args.output/'metadata.json',meta)
    if name == 'qpm':
        model.filtered.to_csv(args.output/'filtered_states.csv',index_label='period')
        model.paths.to_csv(args.output/'forecast_states.csv',index_label='period')
    print(forecast.to_string(index=False))
