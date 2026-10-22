"""Conditional shock-response experiments, distinct from predicting the NBU rate.

No loss weight or shock is estimated to obtain a preferred rate. The quarterly
model cannot identify the timing of two within-quarter announcements beyond
their day-weighted rate average. All policy rankings are conditional on this
approximation, the teaching-model transmission and explicitly saved judgments.
"""
from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from common import ROOT, DataStore, quarter_mean, read_periods, write_json
from policy_qpm import PolicyQPM, anticipated_path
from research_qpm import enrich_path

OUT = ROOT/'results/2026-10-22-policy-decision'
CURRENT_RATE = 16.


def calendar_average(october_rate, december_rate):
    """Hypothetical effects from Oct 30 / Dec 18; not observed future decisions."""
    return (29*CURRENT_RATE+49*float(october_rate)+14*float(december_rate))/92


def inflation_bridge(store, forecast):
    """NSA monthly satellite -> Q4 mean SA log-price, frozen past seasonality.

    Use the same prefix STL/first-observation normalization as DataStore. Future
    seasonal factors repeat the preceding year's factors, a declared bridge
    assumption. Do not refit STL on forecast observations or future actuals.
    """
    observed=store.monthly.CPI_UA_INDEX_NSA_2021_100.loc[:'2026-09'].dropna()
    logged=np.log(observed)
    fitted=STL(logged.to_numpy(),period=12,seasonal=13,trend=21,low_pass=13,robust=True).fit()
    seasonal=pd.Series(fitted.seasonal,index=observed.index)
    base=logged.iloc[0]-seasonal.iloc[0]
    historical_sa=np.exp(logged-seasonal-base)*100
    historical_q3=100*np.log(quarter_mean(historical_sa).loc['2026Q3'])
    expected=float(store.frame('2026Q3').loc['2026Q3','l_cpi'])
    if not np.isclose(historical_q3,expected,atol=1e-9,rtol=0):
        raise ValueError('Bridge and QPM historical normalization disagree')
    rows=[];previous=float(logged.iloc[-1])
    selected=forecast.loc[forecast.selected_by_validation].set_index('period')
    for period in pd.period_range('2026-10','2026-12',freq='M'):
        row=selected.loc[str(period)]
        previous+=float(np.log1p(row.headline_mom_pct/100))
        factor=float(seasonal.loc[period-12])
        sa=float(100*np.exp(previous-factor-base))
        rows.append(dict(period=str(period),model=row.model,headline_mom_pct=float(row.headline_mom_pct),
                         headline_yoy_pct=float(row.headline_yoy_pct),frozen_seasonal_log=factor,sa_index=sa))
    target=float(100*np.log(np.mean([r['sa_index'] for r in rows])))
    return target,dict(months=rows,q4_l_cpi_target=target,historical_q3_l_cpi=expected,
        seasonal_assumption='last-year monthly STL seasonal factors; historical prefix ends 2026-09',
        interpretation='conditional price-level bridge; inferred cost shock is not causally identified')


def fixed_bridge_shock(model, initial, target, horizon=16):
    _,diagnostic=anticipated_path(model,initial,horizon,
        variable_paths={(0,'l_cpi'):float(target)},shock_controls={'l_cpi':'res_dl_cpi'})
    value=float(diagnostic['required_shocks'][0]['total_shock'])
    return {(0,'res_dl_cpi'):value},diagnostic


def shock_scale(model, name):
    values=model.filtered_shocks.loc['2015Q1':'2019Q4',name].to_numpy()
    # Observed dispersion of one-step conditional residual means, not true
    # structural shock volatility or a calibrated probability distribution.
    return float(np.std(values,ddof=1))


def scenario_shocks(model, bridge):
    cost=float(bridge[(0,'res_dl_cpi')])
    demand=shock_scale(model,'res_l_gdp_gap')
    premium=shock_scale(model,'res_prem_uip_gap')
    persistent=dict(bridge)
    persistent.update({(t,'res_dl_cpi'):cost*.5**t for t in range(1,4)})
    return {
        'inherited_only':{},
        'satellite_bridge':dict(bridge),
        'persistent_cost':persistent,
        'negative_demand':dict(bridge)|{(0,'res_l_gdp_gap'):-demand},
        'risk_premium_stress':dict(bridge)|{(0,'res_prem_uip_gap'):premium},
    },dict(demand_conditional_sd=demand,risk_premium_conditional_sd=premium,
           estimation_window='2015Q1–2019Q4',cost_persistence=.5,
           meaning='stress sizes, not event probabilities or causal historical identifications')


def policy_loss(path, october_rate, december_rate, output_weight, smoothing_weight, horizon):
    p=path.iloc[:horizon]
    discount=.99**np.arange(len(p))
    inflation=float(np.sum(discount*(p.inflation_yoy_pct.to_numpy()-5.)**2))
    output=float(np.sum(discount*p.l_gdp_gap.to_numpy()**2))
    # Quarterly average dynamics do not reveal daily transmission. Penalize
    # the two hypothetical announcement changes explicitly, then rule changes.
    changes=np.r_[october_rate-CURRENT_RATE,december_rate-october_rate,
                  p.ir_pol.iloc[1:].to_numpy()-np.r_[december_rate,p.ir_pol.iloc[1:-1].to_numpy()]]
    smooth=float(np.sum((.99**np.arange(len(changes)))*changes**2))
    return dict(inflation_loss=inflation,output_loss=output,smoothing_loss=smooth,
                total_loss=inflation+output_weight*output+smoothing_weight*smooth)


def model_variants(settings):
    return [
        ('selected',dict(settings),None),
        ('corrected_template',{},None),
        ('no_gap_judgment',dict(settings),{}),
        ('strong_gap_judgment',dict(settings),{'2022Q2':(-18.,.05)}),
        ('war_potential_std_0',dict(settings,war_potential_std=0.),None),
        ('war_potential_std_30',dict(settings,war_potential_std=30.),None),
        ('beta_0.5',dict(settings,beta=.5),None),
        ('beta_0.9',dict(settings,beta=.9),None),
        ('potential_growth_4',dict(settings,dl_gdp_tnd_ss=4.),None),
    ]


def run_decisions(output=OUT, settings=None):
    output=Path(output);folder=output/'decision';folder.mkdir(parents=True,exist_ok=True)
    store=DataStore();frame=store.frame('2026Q3',gdp_lag=1)
    forecasts=pd.read_csv(output/'nowcast/cpi_forecast.csv')
    target,bridge_info=inflation_bridge(store,forecasts)
    metrics=pd.read_csv(output/'nowcast/cpi_metrics.csv')
    bridge_bounds={}
    for direction,sign in [('low',-1.),('high',1.)]:
        levels=[]
        for month in bridge_info['months']:
            h=pd.Period(month['period'],'M').month-9
            error=float(metrics.loc[(metrics.split=='audit')&(metrics.model==month['model'])&
                                    (metrics.horizon==h),'RMSE_pp'].iloc[0])
            levels.append(month['sa_index']*(100+month['headline_yoy_pct']+sign*error)/(100+month['headline_yoy_pct']))
        bridge_bounds[direction]=float(100*np.log(np.mean(levels)))
    bridge_info['uncertainty_targets']=bridge_bounds
    bridge_info['uncertainty_meaning']='exploratory +/- one audit RMSE of annual inflation per month, same-sign errors; not a confidence interval or event probability'
    write_json(folder/'inflation_bridge.json',bridge_info)
    grid=np.arange(14.,20.01,.5)
    candidates=[(float(a),float(b)) for a,b in product(grid,repeat=2)]
    path_rows=[];loss_rows=[];state_rows=[];judgments=[];checks=[];shock_rows=[]
    bridge_rows=[];diagnostic_rows=[];scale_rows=[]
    for variant,coefficients,gap_judgments in model_variants(settings or {}):
        model=PolicyQPM(coefficients,gap_judgments=gap_judgments)
        model.fit_forecast(frame,16);initial=model.filtered.iloc[-1].to_numpy()
        model.filtered.assign(variant=variant).rename_axis('period').to_csv(folder/f'filtered_{variant}.csv')
        if variant=='selected':
            model.filtered_shocks.rename_axis('period').to_csv(folder/'filtered_shocks.csv')
            model.filtered_shock_std.rename_axis('period').to_csv(folder/'filtered_shock_std.csv')
        state_rows.append(dict(variant=variant,period='2026Q3',gap_log_pp=float(initial[model.ix['l_gdp_gap']]),
            gap_percent=float(100*np.expm1(initial[model.ix['l_gdp_gap']]/100)),
            gap_conditional_std=float(np.sqrt(model.last_covariance[model.ix['l_gdp_gap'],model.ix['l_gdp_gap']])),
            settings=json.dumps(coefficients,sort_keys=True),gap_judgments=json.dumps({str(k):v for k,v in model.gap_judgments.items()})))
        judgments.extend([dict(variant=variant,**r) for r in model.gap_judgment_diagnostics])
        bridge,bridge_diag=fixed_bridge_shock(model,initial,target)
        bridge_rows.append(dict(variant=variant,cost_shock=bridge[(0,'res_dl_cpi')],
                                **{k:bridge_diag[k] for k in ['structural_residual','constraint_residual']}))
        cases,scales=scenario_shocks(model,bridge);scale_rows.append(dict(variant=variant,**scales))
        for direction,bound in bridge_bounds.items():
            uncertain,_=fixed_bridge_shock(model,initial,bound)
            cases[f'bridge_{direction}']=uncertain
        for scenario,shocks in cases.items():
            shock_rows.extend([dict(variant=variant,scenario=scenario,quarter_offset=t,shock=s,value=v)
                               for (t,s),v in shocks.items()])
            # Two exact solves define an affine control response. Every candidate
            # has the same initial state, same nonpolicy innovations and same rule
            # after Q4. No satellite outcome is re-imposed under changed rates.
            p0,d0=anticipated_path(model,initial,16,constraints={0:16.},shocks=shocks)
            p1,d1=anticipated_path(model,initial,16,constraints={0:17.},shocks=shocks)
            base=enrich_path(model,frame,p0)
            for a,b in candidates:
                average=calendar_average(a,b)
                states=p0+(average-16.)*(p1-p0)
                path=enrich_path(model,frame,states)
                path['delta_inflation_pp_vs_hold16']=path.inflation_yoy_pct-base.inflation_yoy_pct
                path['delta_gdp_percent_vs_hold16']=100*np.expm1((path.l_gdp-base.l_gdp)/100)
                path['delta_fx_percent_vs_hold16']=100*np.expm1((path.l_er-base.l_er)/100)
                # Store full trajectories for the central nearby alternatives;
                # all grid candidates retain their loss vectors and rankings.
                if (a==b and a in [14.,15.5,16.,16.5,18.,20.]) or (a,b) in [(16.,16.5),(16.5,16.)]:
                    keep=['ir_pol','inflation_yoy_pct','l_gdp_gap','gdp_yoy_pct','fx_uah_per_eur',
                          'delta_inflation_pp_vs_hold16','delta_gdp_percent_vs_hold16','delta_fx_percent_vs_hold16']
                    path_rows.extend(path[keep].reset_index(names='period').assign(
                        variant=variant,scenario=scenario,october_rate=a,december_rate=b,
                        q4_average_rate=average).to_dict('records'))
                for h,wg,ws in product([8,12],[.25,1.,2.],[0.,.1,1.]):
                    loss_rows.append(dict(variant=variant,scenario=scenario,october_rate=a,december_rate=b,
                        q4_average_rate=average,horizon_quarters=h,output_weight=wg,smoothing_weight=ws,
                        **policy_loss(path,a,b,wg,ws,h)))
            for average in [14.,20.]:
                direct,diag=anticipated_path(model,initial,16,constraints={0:average},shocks=shocks)
                checks.append(dict(variant=variant,scenario=scenario,rate_target=average,
                    affine_path_max_error=float(np.max(np.abs(direct-(p0+(average-16.)*(p1-p0))))),
                    structural_residual=float(diag['structural_residual']),constraint_residual=float(diag['constraint_residual'])))
            diagnostic_rows.append(dict(variant=variant,scenario=scenario,
                base_rule_shock=d0['required_rule_shocks']['0'],unit_rule_shock=d1['required_rule_shocks']['0']-d0['required_rule_shocks']['0']))
        print('Policy scenarios complete:',variant,flush=True)
    paths=pd.DataFrame(path_rows);losses=pd.DataFrame(loss_rows)
    keys=['variant','scenario','horizon_quarters','output_weight','smoothing_weight']
    losses['rank']=losses.groupby(keys).total_loss.rank(method='min')
    winners=losses.sort_values(keys+['total_loss','october_rate','december_rate']).groupby(keys,as_index=False).first()
    winners['grid_boundary']=(winners.october_rate.isin([14.,20.])|winners.december_rate.isin([14.,20.]))
    losses['regret']=losses.total_loss-losses.groupby(keys).total_loss.transform('min')
    regret=losses.groupby(['october_rate','december_rate']).agg(worst_regret=('regret','max'),
        average_regret=('regret','mean')).reset_index().sort_values('worst_regret')
    for name,data in [('policy_paths',paths),('policy_losses',losses),('conditional_winners',winners),
        ('regret_diagnostics',regret),('initial_state_sensitivity',pd.DataFrame(state_rows)),
        ('judgment_diagnostics',pd.DataFrame(judgments)),('scenario_innovations',pd.DataFrame(shock_rows)),
        ('bridge_innovations',pd.DataFrame(bridge_rows)),('shock_scale_assumptions',pd.DataFrame(scale_rows)),
        ('numerical_checks',pd.DataFrame(checks)),('required_policy_shocks',pd.DataFrame(diagnostic_rows))]:
        data.to_csv(folder/f'{name}.csv',index=False)
    recent=read_periods(ROOT/'data/policy/2026-10-22/monthly.csv','M')
    diagnostic=pd.DataFrame(index=recent.index)
    diagnostic['fx_sales_usd_mn']=recent.fx_sales_usd_mn
    diagnostic['gross_reserves_usd_mn']=recent.gross_reserves_usd_mn
    diagnostic['sales_pct_previous_month_reserves']=100*recent.fx_sales_usd_mn/recent.gross_reserves_usd_mn.shift(1)
    diagnostic.loc['2026-01':].to_csv(folder/'fx_pressure_context.csv')
    numerical=pd.DataFrame(checks)
    if numerical[['affine_path_max_error','structural_residual','constraint_residual']].to_numpy().max()>1e-7:
        raise RuntimeError('Policy equation/affine checks failed')
    protocol=dict(as_of='2026-10-22',model_origin='2026Q3',gdp_available_through='2026Q2',
        decision_window=['2026-10-29','2026-12-17'],effective_dates_assumption=['2026-10-30','2026-12-18'],
        calendar_formula='(29*16 + 49*october_rate + 14*december_rate)/92',
        calendar_source='https://bank.gov.ua/ua/monetary/schedule',
        calendar_verification='schedule dates from saved project context; official live page returned HTTP403 on 2026-10-22; effective-next-day is a scenario convention',
        controlled_period='Q4 2026 only; no commitment over Q1 2027 or an exact January spot rate',
        timing_limitation='Quarterly transmission sees only day-weighted Q4 average. Different within-quarter paths with the same average have the same macro outcome in this approximation.',
        solve_horizon_quarters=16,loss_horizons_quarters=[8,12],candidate_count=len(candidates),
        candidate_rates_pct=grid.tolist(),
        future_nonpolicy_shocks='identical across rate candidates within each variant/scenario; bridge shock inferred once under endogenous rule, then frozen',
        bridge_model='validation-selected headline CPI satellite; repeat last-year seasonal factor to bridge NSA monthly to SA quarterly mean',
        foreign_path='same endogenous teaching-template foreign AR block for all candidate policies',
        scenarios={
            'inherited_only':'filtered inherited state, zero additional future innovations',
            'satellite_bridge':'Q4 cost innovation inferred to match statistical CPI bridge under endogenous policy rule; frozen across controls',
            'persistent_cost':'bridge cost innovation repeated with assumed .5 decay in next three quarters',
            'negative_demand':'bridge plus negative one prewar SD of filtered demand-innovation means in Q4',
            'risk_premium_stress':'bridge plus positive one prewar SD of filtered premium-innovation means in Q4; no estimated flow-to-premium mapping',
            'bridge_low':'satellite annual CPI lowered by one audit RMSE per month, coherent same-sign bridge; exploratory, not a probability interval',
            'bridge_high':'satellite annual CPI raised by one audit RMSE per month, coherent same-sign bridge; exploratory, not a probability interval'},
        inflation_measure='exact yoy percent change of seasonally adjusted quarterly-average price level; not monthly NSA CPI or Dec/Dec',
        output_gap_measure='log percentage-point gap, latent and sensitive to potential/judgments, not an observable',
        criterion='sum .99^t [(inflation_pct-5)^2 + output_weight*gap_log_pp^2] + smoothing_weight*announcement/rule changes squared',
        weights='inflation=1; output=.25,1,2; smoothing=0,.1,1; normative assumptions, never estimated from actual rates',
        rankings='conditional grid comparisons; a boundary winner is not an established optimum; variant/scenario counts are not probabilities',
        causal_status='teaching-model counterfactuals, not empirically identified policy effects or proof that a historical decision caused success',
        numerical_max_residual=float(numerical[['affine_path_max_error','structural_residual','constraint_residual']].to_numpy().max()),
        path_rows=len(paths),loss_rows=len(losses),conditional_winners=len(winners))
    write_json(folder/'protocol.json',protocol)
    inputs=[ROOT/'models/policy_decision.py',ROOT/'models/policy_qpm.py',
        ROOT/'models/research_qpm.py',ROOT/'models/common.py',ROOT/'data/inputs/core_monthly.csv',
        ROOT/'data/model/quarterly_levels.csv',ROOT/'data/policy/2026-10-22/monthly.csv',
        output/'nowcast/cpi_forecast.csv',output/'calibration/policy_calibration_protocol.json']
    write_json(folder/'input_hashes.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in inputs if p.exists()})
    print('Decision experiments saved:',dict(paths=len(paths),losses=len(losses),winners=len(winners)),flush=True)
    return paths,losses,winners


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT)
    args=p.parse_args();calibration=args.output/'calibration/policy_calibration_protocol.json'
    settings=json.loads(calibration.read_text()).get('selected_settings',{}) if calibration.exists() else {}
    run_decisions(args.output,settings)
