"""Research figures for the conditional recommendation; no fitted coefficients."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT,write_json
OUT=ROOT/'results/2026-10-23-conditional-decision'
REPORT=ROOT/'reports/policy-decision'

def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    paths=pd.read_csv(OUT/'paths.csv');robust=pd.read_csv(OUT/'robustness.csv')
    central=paths.loc[(paths.initial_state=='central')&(paths.beta==.7)&
                      (paths.scenario=='bridge_central')].copy()
    periods=sorted(central.period.unique())[:8];x=np.arange(8)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for rate,color in [(15.5,'#936324'),(16.5,'#236e70'),(17.,'#755090')]:
        g=central.loc[central.rate==rate].set_index('period').loc[periods]
        axes[0].plot(x,g.delta_inflation_pp,'o-',label=f'{rate:g}%',color=color)
        axes[1].plot(x,g.delta_gdp_pct,'o-',label=f'{rate:g}%',color=color)
    for ax,title,unit in zip(axes,['Зміна інфляції проти збереження 16%','Зміна рівня ВВП проти збереження 16%'],
                             ['Відсоткові пункти, CPI SA р/р','Відсотки рівня реального ВВП']):
        ax.axhline(0,color='#333333',lw=.8);ax.set_xticks(x,periods,rotation=40);ax.set_xlabel('Квартал наслідків рішення');ax.set_ylabel(unit);ax.set_title(title);ax.legend();ax.grid(alpha=.16)
    fig.suptitle('Однаковий шок і початковий стан: виграш інфляції має ціну для випуску')
    fig.text(.5,-.04,'Джерело: власний умовний QPM-експеримент; дані до 22.10.2026. Рішення Q4, спільне правило надалі; не причинна оцінка.',ha='center',fontsize=8)
    fig.savefig(REPORT/'policy_tradeoff.png',dpi=170,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    ax.plot(robust.rate,robust.worst_regret_vs_hold*100,'o-',color='#236e70')
    primary=robust.loc[robust.worst_regret_vs_hold.idxmin()]
    ax.scatter([primary.rate],[primary.worst_regret_vs_hold*100],s=80,color='#b13e53',zorder=3)
    ax.set_xticks(robust.rate);ax.set_xlabel('Однакова оголошена ставка на двох рішеннях Q4, %');ax.set_ylabel('Найгірший нормований regret, % втрат hold16')
    ax.set_title('Стійкість рішення залежить від нормативного критерію');ax.grid(alpha=.16)
    secondary=robust.loc[robust.worst_regret_vs_range.idxmin(),'rate']
    fig.text(.5,-.04,f'Первинна нормалізація: мінімум {primary.rate:g}%; альтернативна за діапазоном втрат: {secondary:g}%. Сценарії не мають ймовірностей.',ha='center',fontsize=9)
    fig.savefig(REPORT/'decision_robustness.png',dpi=170,bbox_inches='tight');plt.close(fig)
    diagnostics=paths.loc[(paths.tier=='diagnostic')&(paths.rate==16.)]
    base=diagnostics.loc[diagnostics.scenario=='no_new_innovation'].set_index('period')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    labels={'pure_cost':'Витрати +1','pure_demand':'Попит −1','pure_potential':'Потенціал −1','pure_premium':'Премія ризику +1'}
    for name,label in labels.items():
        g=diagnostics.loc[diagnostics.scenario==name].set_index('period')
        axes[0].plot(x,(g.inflation_yoy_pct-base.inflation_yoy_pct).loc[periods],label=label)
        axes[1].plot(x,100*np.expm1((g.l_gdp-base.l_gdp).loc[periods]/100),label=label)
    for ax,title,unit in zip(axes,['Різниця річної інфляції','Різниця рівня реального ВВП'],['Відсоткові пункти','Відсотки']):
        ax.axhline(0,color='#333333',lw=.8);ax.set_xticks(x,periods,rotation=40);ax.set_xlabel('Квартал');ax.set_ylabel(unit);ax.set_title(title);ax.legend(fontsize=8);ax.grid(alpha=.16)
    fig.suptitle('Чисті модельні шоки: механізми, а не виміряні зимові події')
    fig.text(.5,-.04,'Потенційний шок у цій специфікації не створює інфляції сам по собі. Одиниці шоків різні; не порівнюйте їх як однакові за розміром події.',ha='center',fontsize=8)
    fig.savefig(REPORT/'shock_mechanisms.png',dpi=170,bbox_inches='tight');plt.close(fig)
    central.to_csv(OUT/'central_paths_for_presentation.csv',index=False)
    # Separate rendering manifest: the original numerical run remains immutable.
    write_json(OUT/'report_manifest.json',dict(inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'paths.csv',OUT/'robustness.csv']},
        output_images={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in REPORT.glob('*.png')}))
    print('Three figures saved:',REPORT)
if __name__=='__main__':main()
