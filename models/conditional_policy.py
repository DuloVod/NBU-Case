"""Fixed-state shock scenarios and explicitly normative robust decision review.

No causal energy/fuel elasticity is inferred. All nonpolicy shocks are identical
across rate alternatives. Diagnostic pure shocks do not enter the policy rank.
"""
from pathlib import Path
import hashlib,json
from itertools import product
import numpy as np
import pandas as pd
from common import ROOT,DataStore,write_json
from policy_qpm import PolicyQPM,anticipated_path
from policy_decision import calendar_average,policy_loss
from research_qpm import enrich_path

OUT=ROOT/'results/2026-10-23-conditional-decision'
OLD=ROOT/'results/2026-10-22-policy-decision/decision'
RATES=np.arange(14.,20.01,.5)


def rank_robust(losses):
    """Worst relative regret vs the same-context hold loss (floor 1).

    Baseline normalization is declared before solving. Not a probability or
    estimated welfare function. Range normalization is reported as sensitivity.
    """
    keys=['initial_state','beta','scenario','output_weight','smoothing_weight','horizon']
    data=losses.loc[losses.tier=='decision'].copy()
    hold=data.loc[data.rate==16.,keys+['total_loss']].rename(columns={'total_loss':'hold_loss'})
    data=data.merge(hold,on=keys,validate='many_to_one')
    groups=data.groupby(keys).total_loss
    data['best_loss']=groups.transform('min');data['worst_loss']=groups.transform('max')
    data['regret']=data.total_loss-data.best_loss
    data['regret_vs_hold_scale']=data.regret/np.maximum(data.hold_loss,1.)
    data['regret_vs_range_scale']=data.regret/np.maximum(data.worst_loss-data.best_loss,1e-9)
    summary=data.groupby('rate').agg(worst_regret_vs_hold=('regret_vs_hold_scale','max'),
        worst_regret_vs_range=('regret_vs_range_scale','max'),contexts=('total_loss','size')).reset_index()
    return data,summary


def scenario_definitions():
    records=pd.read_csv(OLD/'scenario_innovations.csv')
    def bridge(name):
        return float(records.loc[(records.variant=='selected')&(records.scenario==name)&
                                 (records.shock=='res_dl_cpi')&(records.quarter_offset==0),'value'].iloc[0])
    anchor=bridge('satellite_bridge')
    b={(0,'res_dl_cpi'):anchor}
    # The bridge residual is not renamed a measured fuel or energy innovation.
    decision={
        'bridge_low':{(0,'res_dl_cpi'):bridge('bridge_low')},
        'bridge_central':dict(b),
        'bridge_high':{(0,'res_dl_cpi'):bridge('bridge_high')},
        'winter_cost_tail':b|{(1,'res_dl_cpi'):2.,(2,'res_dl_cpi'):1.},
        'weak_demand':b|{(0,'res_l_gdp_gap'):-1.},
        'fx_risk':b|{(0,'res_prem_uip_gap'):1.,(1,'res_prem_uip_gap'):.5},
        'energy_supply_mix':b|{(0,'res_l_gdp_tnd'):-1.,(0,'res_dl_cpi'):anchor+1.,
                               (1,'res_dl_cpi'):.5,(0,'res_l_gdp_gap'):-.5},
        'persistent_price_risk':b|{(t,'res_dl_cpi'):1. for t in [1,2,3]},
    }
    diagnostic={'pure_cost':{(0,'res_dl_cpi'):1.},'pure_demand':{(0,'res_l_gdp_gap'):-1.},
                'pure_potential':{(0,'res_l_gdp_tnd'):-1.},
                'pure_premium':{(0,'res_prem_uip_gap'):1.},'no_new_innovation':{}}
    return decision,diagnostic


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    decision,diagnostic=scenario_definitions()
    protocol=dict(research_date='2026-10-23',information_cutoff='2026-10-22 end of day; recommendation drafted 2026-10-23',
        decision_window='Oct29/Dec17 2026; next-day effectiveness is an assumption',
        candidates='same announced rate at both Q4 meetings, 14..20 by .5; common endogenous rule afterwards',
        calendar_formula='(29*16 + 63*rate)/92',solve_horizon=16,loss_horizons=[8,12],
        initial_states={'low_gap':'prior beta_0.9 filtered vector; gap -0.610, not a confidence bound',
                        'central':'prior selected filtered vector; gap +0.462',
                        'high_gap':'prior war_potential_std_0 vector; gap +1.711'},
        beta=[.5,.7,.9],beta_meaning='weight on managed exchange-rate target vs UIP, not intervention volume',
        crossing='same full initial vector across beta cases; initial-state cases vary other latent states too, not an isolated gap intervention',
        shocks_fixed='same innovations across rate candidates AND beta/state stresses; central bridge matched only original selected endogenous rule',
        primary_criterion='minimize maximum regret / max(hold16 loss in same context,1); no probabilities or averaging scenario winners',
        secondary_criterion='normalize regret by within-context candidate loss range; explicitly report any change in choice',
        normative_weights=dict(inflation=1,output=[.25,1,2],smoothing=[.1,1],discount=.99),
        scenario_meaning='stress assumptions, not measured structural energy/fuel shocks or event probabilities',
        bridge_limits='low/high use previously inspected audit RMSE; exploratory stress, not independent confidence bounds',
        physical_mapping='No elasticity from fuel prices, power deficit or survey expectations to structural shocks is estimated',
        expectation_proxy='persistent_price_risk is residual price pressure, NOT an endogenous credibility or estimated expectation block',
        diagnostic_exclusion='pure shocks used to explain mechanisms only; excluded from the decision minimax',
        status='conditional research recommendation; no global optimum or causal identification')
    write_json(OUT/'protocol.json',protocol)
    shock_rows=[dict(tier=tier,scenario=name,quarter_offset=t,shock=s,value=v)
        for tier,cases in [('decision',decision),('diagnostic',diagnostic)]
        for name,shocks in cases.items() for (t,s),v in shocks.items()]
    pd.DataFrame(shock_rows).to_csv(OUT/'scenario_assumptions.csv',index=False)
    state_sources={'low_gap':'beta_0.9','central':'selected','high_gap':'war_potential_std_0'}
    store=DataStore();frame=store.frame('2026Q3',gdp_lag=1)
    paths=[];losses=[];checks=[];state_rows=[]
    for beta in [.5,.7,.9]:
        model=PolicyQPM({'beta':beta});model.fit_forecast(frame,16)
        for state_name,source in state_sources.items():
            saved=pd.read_csv(OLD/f'filtered_{source}.csv',index_col=0)
            initial=saved.iloc[-1].reindex(model.variables).to_numpy(float)
            if not np.isfinite(initial).all():raise ValueError('Initial vector mismatch')
            state_rows.append(dict(beta=beta,initial_state=state_name,
                gap=float(initial[model.ix['l_gdp_gap']]),source=f'filtered_{source}.csv'))
            cases=[('decision',n,s) for n,s in decision.items()]
            if state_name=='central' and beta==.7:
                cases += [('diagnostic',n,s) for n,s in diagnostic.items()]
            for tier,name,shocks in cases:
                base,d0=anticipated_path(model,initial,16,constraints={0:16.},shocks=shocks)
                one,d1=anticipated_path(model,initial,16,constraints={0:17.},shocks=shocks)
                baseline=enrich_path(model,frame,base)
                for rate in RATES:
                    avg=calendar_average(rate,rate)
                    states=base+(avg-16.)*(one-base)
                    path=enrich_path(model,frame,states)
                    if rate in [15.5,16.,16.5,17.,17.5]:
                        table=path[['ir_pol','inflation_yoy_pct','l_gdp_gap','l_gdp','l_gdp_tnd','l_er']].copy()
                        table['delta_inflation_pp']=path.inflation_yoy_pct-baseline.inflation_yoy_pct
                        table['delta_gdp_pct']=100*np.expm1((path.l_gdp-baseline.l_gdp)/100)
                        table['delta_fx_pct']=100*np.expm1((path.l_er-baseline.l_er)/100)
                        paths.extend(table.reset_index(names='period').assign(beta=beta,initial_state=state_name,
                                     tier=tier,scenario=name,rate=rate,q4_average=avg).to_dict('records'))
                    if tier=='decision':
                        for h,wg,ws in product([8,12],[.25,1.,2.],[.1,1.]):
                            losses.append(dict(beta=beta,initial_state=state_name,tier=tier,scenario=name,
                                rate=rate,horizon=h,output_weight=wg,smoothing_weight=ws,
                                **policy_loss(path,rate,rate,wg,ws,h)))
                # Independent solve for the discussed +50bp, not just affine interpolation.
                direct,dd=anticipated_path(model,initial,16,
                    constraints={0:calendar_average(16.5,16.5)},shocks=shocks)
                checks.append(dict(beta=beta,initial_state=state_name,scenario=name,
                    structural_residual=dd['structural_residual'],constraint_residual=dd['constraint_residual'],
                    affine_max_error=float(np.max(np.abs(direct-(base+(calendar_average(16.5,16.5)-16.)*(one-base)))))))
        print('Fixed-state policy stresses complete: beta',beta,flush=True)
    pd.DataFrame(paths).to_csv(OUT/'paths.csv',index=False)
    pd.DataFrame(state_rows).to_csv(OUT/'initial_states.csv',index=False)
    checks=pd.DataFrame(checks);checks.to_csv(OUT/'numerical_checks.csv',index=False)
    if checks[['structural_residual','constraint_residual','affine_max_error']].to_numpy().max()>1e-7:
        raise AssertionError('Structural or interpolation residual too large')
    detailed,summary=rank_robust(pd.DataFrame(losses))
    detailed.to_csv(OUT/'losses.csv',index=False);summary.to_csv(OUT/'robustness.csv',index=False)
    primary=summary.sort_values(['worst_regret_vs_hold','rate']).iloc[0]
    secondary=summary.sort_values(['worst_regret_vs_range','rate']).iloc[0]
    result=dict(primary_rate=float(primary.rate),primary_worst_regret=float(primary.worst_regret_vs_hold),
        alternative_normalization_rate=float(secondary.rate),
        decision_contexts=int(primary.contexts),loss_rows=len(detailed),path_rows=len(paths),
        primary_criterion=protocol['primary_criterion'],status='scenario-dependent diagnostic, not estimated optimal rate')
    write_json(OUT/'summary.json',result)
    inputs=[Path(__file__),ROOT/'models/policy_qpm.py',ROOT/'models/policy_decision.py',
        ROOT/'models/research_qpm.py',ROOT/'models/qpm.py',ROOT/'models/common.py',
        ROOT/'data/model/quarterly_levels.csv',ROOT/'data/inputs/core_monthly.csv',
        ROOT/'data/model/auxiliary_monthly.csv',ROOT/'data/model/auxiliary_quarterly.csv',
        OLD/'scenario_innovations.csv',*[OLD/f'filtered_{s}.csv' for s in state_sources.values()],
        *list((ROOT/'nbu_challenge_model_2026/model').glob('*'))]
    write_json(OUT/'manifest.json',dict(inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        outputs={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in
            ['protocol.json','scenario_assumptions.csv','paths.csv','initial_states.csv',
             'numerical_checks.csv','losses.csv','robustness.csv','summary.json']}))
    print(json.dumps(result,indent=2))

if __name__=='__main__':run()
