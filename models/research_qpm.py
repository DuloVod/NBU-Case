"""Bounded QPM calibration audit and rational-expectations policy experiments.

This is a pseudo-out-of-sample, current-vintage research exercise. Historical
outcomes have already been inspected in this project; the last block is a
chronological audit, not a pristine unseen test. Calibration is predictive,
not identification of causal parameters. Policy loss weights are assumptions.
"""
from __future__ import annotations
from itertools import product
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from scipy.linalg import solve
from common import ROOT, DataStore, write_json
from qpm import QPM

OUT = ROOT/'results/2026-10-22-research'
AS_OF = '2026-10-22'
METRICS = ['rate_pp', 'inflation_log_pp', 'fx_log_percent', 'gdp_log_percent']


class ResearchQPM(QPM):
    """Two explicitly named coefficient substitutions; original template intact."""
    def __init__(self, settings=None):
        self.settings = dict(settings or {})
        super().__init__(overrides={k:v for k,v in self.settings.items()
                                   if k not in ('policy_inertia','inflation_memory')})

    def expression(self, text):
        text = text.strip()
        if '0.8*ir_pol{-1}' in text:
            inertia = float(self.settings.get('policy_inertia', .8))
            text = text.replace('0.8*ir_pol{-1}',f'{inertia}*ir_pol{{-1}}')
            text = text.replace('(1 - 0.8)',f'(1 - {inertia})')
        if '0.4*dl_cpi{-1}' in text:
            memory = float(self.settings.get('inflation_memory', .4))
            text = text.replace('0.4*dl_cpi{-1}',f'{memory}*dl_cpi{{-1}}')
            text = text.replace('(1 - 0.4)',f'(1 - {memory})')
        return super().expression(text)


def model_vector(model, frame):
    state = model.paths.iloc[0]
    cpi_lag = frame.l_cpi.iloc[-4]
    return np.array([state.ir_pol, state.l_cpi-cpi_lag,
                     state.l_er, state.l_gdp],float)


def target_vector(store, period):
    # Evaluation labels only. Never passed into a training information set.
    actual = store.frame(period,gdp_lag=0)
    row = actual.iloc[-1]
    return np.array([row.ir_pol,row.l_cpi-actual.l_cpi.iloc[-5],
                     row.l_er,row.l_gdp],float)


def evaluate_candidate(store, settings, origins):
    model=ResearchQPM(settings); rows=[]
    for origin in origins:
        frame=store.frame(origin,gdp_lag=1)
        model.fit_forecast(frame,1)
        prediction=model_vector(model,frame)
        actual=target_vector(store,origin+1)
        for j,metric in enumerate(METRICS):
            rows.append(dict(origin=str(origin),period=str(origin+1),metric=metric,
                             prediction=float(prediction[j]),actual=float(actual[j]),
                             error=float(prediction[j]-actual[j])))
    return pd.DataFrame(rows)


def robust_score(errors, scales):
    value=np.abs(errors/scales)
    # Huber loss: war jumps remain in evaluation but do not dominate calibration.
    return float(np.mean(np.where(value<=2,.5*value**2,2*(value-1))))


def calibrate(store, output):
    development=pd.period_range('2015Q4','2019Q4',freq='Q')
    validation=pd.period_range('2020Q1','2022Q4',freq='Q')
    audit=pd.period_range('2023Q1','2026Q2',freq='Q')
    default={}
    dev=evaluate_candidate(store,default,development)
    scale=dev.groupby('metric').error.apply(lambda s:float(np.sqrt(np.mean(s**2))))
    scale=scale.clip(lower=1.0).reindex(METRICS)
    settings=[{}]+[dict(beta=b,coef=c,policy_inertia=r,inflation_memory=m,
                       std_res_dl_cpi_c=s) for b,c,r,m,s in
                    product([.5,.7,.9],[1.,1.5],[.6,.8,.9],[.3,.4,.6],[1.,3.])]
    scores=[]; predictions=[]
    for i,setting in enumerate(settings):
        frame=evaluate_candidate(store,setting,validation)
        errors=frame.pivot(index='origin',columns='metric',values='error').reindex(columns=METRICS)
        score=robust_score(errors.to_numpy(),scale.to_numpy())
        scores.append(dict(candidate=i,score=score,settings=json.dumps(setting,sort_keys=True)))
        if i%20==0:print('QPM calibration',i,'/',len(settings),flush=True)
    scores=pd.DataFrame(scores).sort_values(['score','candidate'])
    default_score=float(scores.loc[scores.candidate==0,'score'].iloc[0])
    best=scores.iloc[0]
    # Predefined conservative rule: retain template unless validation improves 5%.
    selected=int(best.candidate) if best.score<.95*default_score else 0
    chosen=settings[selected]
    scores.to_csv(output/'qpm_calibration_grid.csv',index=False)
    for name,setting in [('template',default),('selected',chosen),('best_validation',settings[int(best.candidate)])]:
        for split,origins in [('development',development),('validation',validation),('audit',audit)]:
            frame=evaluate_candidate(store,setting,origins)
            predictions.append(frame.assign(model=name,split=split))
    predictions=pd.concat(predictions,ignore_index=True)
    predictions.to_csv(output/'qpm_calibration_predictions.csv',index=False)
    rows=[]
    for (name,split,metric),group in predictions.groupby(['model','split','metric']):
        rows.append(dict(model=name,split=split,metric=metric,n=int(group.error.notna().sum()),n_origins=len(group),
                         MAE=float(group.error.abs().mean()),RMSE=float(np.sqrt(np.mean(group.error**2))),
                         bias=float(group.error.mean())))
    pd.DataFrame(rows).to_csv(output/'qpm_calibration_metrics.csv',index=False)
    protocol=dict(as_of=AS_OF,development_origins=[str(development[0]),str(development[-1])],
                  validation_origins=[str(validation[0]),str(validation[-1])],
                  audit_origins=[str(audit[0]),str(audit[-1])],candidate_count=len(settings),
                  objective='equal-weight standardized Huber loss across four one-quarter predictions',
                  scales_from_development=scale.to_dict(),default_score=default_score,
                  best_validation_score=float(best.score),selected_candidate=selected,selected_settings=chosen,
                  best_validation_settings=settings[int(best.candidate)],
                  validation_improvement_fraction=1-float(best.score)/default_score,
                  selection='5% validation improvement required; otherwise retain template; never select on audit',
                  structural_estimation='bounded predictive calibration, not causal identification or full Bayesian estimation',
                  information='quarter end +21d proxy; both GDP series lagged one quarter; current revised vintages',
                  caveat='historical outcomes already inspected; chronological audit is not pristine holdout')
    write_json(output/'qpm_calibration_protocol.json',protocol)
    return chosen,protocol


def anticipated_path(model, initial_state, horizon, constraints=None, shocks=None):
    """Solve stacked structural equations with known future shocks and targets.

    Terminal condition is the stable RE law, not zero terminal levels. Each
    rate constraint endogenizes the Taylor-rule shock in the same quarter.
    This retains the original model's expectation equations; adding a reduced-
    form impact vector every quarter would miss anticipation effects.
    """
    n=model.n;constraints=dict(constraints or {});shocks=dict(shocks or {})
    matrix=np.zeros((n*horizon,n*horizon)); rhs=np.tile(model.c,horizon)
    for t in range(horizon):
        sl=slice(t*n,(t+1)*n);matrix[sl,sl]=model.a0
        if t:matrix[sl,slice((t-1)*n,t*n)]=model.am
        else:rhs[sl]-=model.am@initial_state
        if t<horizon-1:matrix[sl,slice((t+1)*n,(t+2)*n)]=model.ap
        else:
            matrix[sl,sl]+=model.ap@model.transition
            rhs[sl]-=model.ap@model.intercept
        eps=np.zeros(model.k)
        for (when,name),value in shocks.items():
            if when==t:eps[model.shocks.index(name)]+=value
        rhs[sl]+=model.e@eps
    ordered=sorted(constraints);k=len(ordered)
    augment=np.zeros((n*horizon+k,n*horizon+k));augment[:n*horizon,:n*horizon]=matrix
    for j,t in enumerate(ordered):
        if not 0<=t<horizon:raise ValueError('Constraint outside horizon')
        augment[t*n:(t+1)*n,n*horizon+j]=-model.e[:,model.shocks.index('res_ir_tlr')]
        augment[n*horizon+j,t*n+model.ix['ir_pol']]=1
    fullrhs=np.r_[rhs,[constraints[t] for t in ordered]]
    solution=solve(augment,fullrhs,check_finite=True)
    residual=float(np.max(np.abs(augment@solution-fullrhs)))
    if residual>1e-6:raise RuntimeError(f'Stacked structural residual {residual}')
    states=solution[:n*horizon].reshape(horizon,n)
    return states,dict(structural_residual=residual,
                       required_rule_shocks={str(t):float(solution[n*horizon+j]) for j,t in enumerate(ordered)})


def enrich_path(model, frame, states):
    index=pd.period_range(frame.index[-1]+1,periods=len(states),freq='Q')
    path=pd.DataFrame(states,index=index,columns=model.variables)
    # Exact percent change of quarterly average model price levels, NOT Dec/Dec.
    path['inflation_yoy_pct']=100*np.expm1(path.d4l_cpi/100)
    path['fx_uah_per_eur']=np.exp(path.l_er/100)
    history=model.filtered[['l_gdp','l_cpi']]
    joined=pd.concat([history,path[['l_gdp','l_cpi']]])
    path['gdp_yoy_pct']=100*np.expm1(joined.l_gdp.diff(4).reindex(index)/100)
    path['gdp_qoq_pct']=100*np.expm1(joined.l_gdp.diff().reindex(index)/100)
    return path


def policy_experiments(store, settings, output, validation_best=None):
    frame=store.frame('2026Q3',gdp_lag=1)
    final=ResearchQPM(settings);final.fit_forecast(frame,12)
    final.filtered.to_csv(output/'qpm_research_filtered.csv',index_label='period')
    frame.to_csv(output/'qpm_research_information.csv',index_label='period')
    horizon=12;rows=[];diagnostics=[];losses=[]
    # Not estimated probabilities: these are illustrative sensitivity shocks.
    shock_cases={'no_additional_shock':{},'cost_pressure':{(0,'res_dl_cpi'):2.},
                 'risk_premium':{(0,'res_prem_uip_gap'):2.}}
    variants=[('selected',settings),('template',{}),
              ('validation_best',dict(validation_best or settings)),
              ('less_fx_smoothing',dict(settings,beta=.5)),
              ('more_fx_smoothing',dict(settings,beta=.9)),
              ('higher_risk_trend',dict(settings,prem_uip_tnd_ss=6.)),
              ('higher_real_trend',dict(settings,prem_uip_tnd_ss=8.)),
              ('exact_price_targets',dict(settings,dl_cpi_tar=float(100*np.log1p(.05)),
                                           dl_cpi_ea_tar=float(100*np.log1p(.02))))]
    seen=set()
    for variant,setting in variants:
        signature=json.dumps(setting,sort_keys=True)
        if signature in seen:continue
        seen.add(signature)
        model=ResearchQPM(setting);model.fit_forecast(frame,horizon)
        initial=model.filtered.iloc[-1].to_numpy()
        recursive,_=anticipated_path(model,initial,horizon)
        np.testing.assert_allclose(recursive,model.paths.to_numpy(),atol=2e-6)
        for shock_case,shock in shock_cases.items():
            for label,decision in [('cut_15_5',15.5),('hold_16',16.),('hike_16_5',16.5),('endogenous_rule',None)]:
                # Explicit policy experiment: 16 through Oct29, chosen rate from
                # Oct30 to Dec31. No additional Q4 decision; endogenous from Q1.
                q4_rate=None if decision is None else (29*16.+63*decision)/92
                constraints={} if decision is None else {0:q4_rate}
                states,diagnostic=anticipated_path(model,initial,horizon,constraints,shock)
                path=enrich_path(model,frame,states)
                diagnostic.update(variant=variant,shock_case=shock_case,policy=label,
                                  decision_rate=decision,q4_average=q4_rate)
                diagnostics.append(diagnostic)
                saved=path.reset_index(names='period').assign(variant=variant,shock_case=shock_case,policy=label)
                saved['period']=saved.period.astype(str);rows.append(saved)
                rate_changes=np.diff(np.r_[float(frame.ir_pol.iloc[-1]),path.ir_pol])
                discounts=.99**np.arange(horizon)
                for output_weight in [0.,.25,1.,2.]:
                    for smoothing_weight in [0.,.1,1.]:
                        # Weights are explicit normative assumptions, NOT fitted
                        # welfare weights or a central-bank objective estimate.
                        for target in [5.,float(100*np.expm1(.05))]:
                            value=(path.inflation_yoy_pct.to_numpy()-target)**2+output_weight*path.l_gdp_gap.to_numpy()**2+smoothing_weight*rate_changes**2
                            losses.append(dict(variant=variant,shock_case=shock_case,policy=label,
                                               output_weight=output_weight,smoothing_weight=smoothing_weight,
                                               inflation_target_pct=target,
                                               discounted_loss=float(np.average(value,weights=discounts))))
    allpaths=pd.concat(rows,ignore_index=True)
    allpaths.to_csv(output/'qpm_policy_paths.csv',index=False)
    losses=pd.DataFrame(losses)
    # Compare explicit rate decisions only, not the unrestricted rule's policy set.
    decisions=losses.loc[losses.policy!='endogenous_rule']
    winners=decisions.loc[decisions.groupby(['variant','shock_case','output_weight','smoothing_weight','inflation_target_pct']).discounted_loss.idxmin()]
    winners.to_csv(output/'qpm_loss_sensitivity.csv',index=False)
    losses.to_csv(output/'qpm_policy_losses.csv',index=False)
    selected=allpaths.loc[(allpaths.variant=='selected')&(allpaths.shock_case=='no_additional_shock')]
    selected.to_csv(output/'qpm_policy_comparison.csv',index=False)
    differences=[]
    keys=['variant','shock_case','period']
    hold=allpaths.loc[allpaths.policy=='hold_16'].set_index(keys)
    for policy in ['cut_15_5','hike_16_5']:
        alternative=allpaths.loc[allpaths.policy==policy].set_index(keys)
        delta=(alternative[['ir_pol','inflation_yoy_pct','l_gdp_gap','fx_uah_per_eur']]-hold[['ir_pol','inflation_yoy_pct','l_gdp_gap','fx_uah_per_eur']]).reset_index()
        delta['fx_percent_vs_hold']=100*np.expm1((alternative.l_er-hold.l_er).to_numpy()/100)
        delta=delta.rename(columns={k:'delta_'+k for k in ['ir_pol','inflation_yoy_pct','l_gdp_gap','fx_uah_per_eur']})
        differences.append(delta.assign(policy=policy))
    pd.concat(differences).to_csv(output/'qpm_policy_differences.csv',index=False)
    write_json(output/'qpm_policy_diagnostics.json',dict(as_of=AS_OF,origin='2026Q3',horizon_quarters=horizon,
               gdp_available_through='2026Q2',settings=settings,solver=final.solver_info,
               scenario_timing='16% through Oct29; alternative from Oct30 through Dec31, then endogenous rule; hypothetical Q4 commitment',
               no_december_change='assumption of experiment; not a prediction or recommendation to precommit at Dec17 meeting',
               shock_assumptions={'cost_pressure':'res_dl_cpi +2 annualized log percentage points in Q4 only',
                                  'risk_premium':'res_prem_uip_gap +2 percentage points in Q4; then template persistence'},
               inflation='exact yoy change in seasonally adjusted quarterly-average price level; not monthly or Dec/Dec CPI',
               fx='UAH/EUR quarterly-average model level; not UAH/USD or an optimal intervention target',
               terminal='stable RE law after 12 quarters; quarterly inflation target template uses log-percent approximation 5',
               limitations=['fixed-form small model','predictive calibration is not structural identification',
                            'no explicit reserves, FX controls, core inflation or fiscal block',
                            'no probability estimates for shocks or parameter variants',
                            'loss weights are assumptions; no claim of welfare optimality'],experiments=diagnostics))
    return allpaths,winners


def run(output=OUT):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    store=DataStore();settings,protocol=calibrate(store,output)
    paths,winners=policy_experiments(store,settings,output,protocol['best_validation_settings'])
    files=[Path(__file__),ROOT/'models/qpm.py',ROOT/'models/common.py',ROOT/'data/inputs/core_monthly.csv',ROOT/'data/model/quarterly_levels.csv']
    files+=list((ROOT/'nbu_challenge_model_2026/model').glob('*'))
    write_json(output/'qpm_research_manifest.json',dict(as_of=AS_OF,selected_settings=settings,
               sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()},
               numpy=np.__version__,pandas=pd.__version__,original_data_unchanged=True))
    print('Selected calibration:',settings)
    print('Policy loss winners by assumption:',winners.policy.value_counts().to_dict())
    return protocol,paths,winners


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();began=time.perf_counter();run(args.output)
    print(f'QPM research completed in {time.perf_counter()-began:.1f}s')
