#!/usr/bin/env python3
"""Summarize actual saved policy runs; do not fit models or invent outcomes."""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'results/2026-10-22-policy-decision'
COLORS={'ARIMA':'#ca8622','SARIMA':'#487a58','VAR':'#153f75','VAR_POLICY':'#985c95','SEASONAL_NAIVE':'#888888'}

def _load_json(path):
 return json.loads(path.read_text()) if path.exists() else {}
def _fmt(x,n=2):return f'{float(x):.{n}f}'.replace('.',',')
def _truth(series):return series.astype(str).str.lower().isin(['true','1'])
def _bullet_table(frame,cols):
 # A compact reusable document fragment; all values come from saved CSVs.
 return '\n'.join('- '+', '.join(f'{label}: {_fmt(row[key]) if isinstance(row[key],(float,np.floating)) else row[key]}' for key,label in cols) for _,row in frame.iterrows())
def summarize(output=DEFAULT):
 output=Path(output);now=output/'nowcast';decision=output/'decision'
 fc=pd.read_csv(now/'cpi_forecast.csv');metrics=pd.read_csv(now/'cpi_metrics.csv')
 paths=pd.read_csv(decision/'policy_paths.csv');losses=pd.read_csv(decision/'policy_losses.csv')
 winners=pd.read_csv(decision/'conditional_winners.csv');gaps=pd.read_csv(decision/'initial_state_sensitivity.csv')
 judgments=pd.read_csv(decision/'judgment_diagnostics.csv');num=pd.read_csv(decision/'numerical_checks.csv')
 protocol=_load_json(decision/'protocol.json');np_= _load_json(now/'cpi_protocol.json')
 bridge=_load_json(decision/'inflation_bridge.json');cal=_load_json(output/'calibration/policy_calibration_protocol.json')
 bridge_innovations=pd.read_csv(decision/'bridge_innovations.csv')
 bridge_cost=float(bridge_innovations.loc[bridge_innovations.variant=='selected','cost_shock'].iloc[0])
 data=_load_json(ROOT/'data/policy/2026-10-22/checks.json')
 selected=fc[_truth(fc.selected_by_validation)].sort_values('horizon')
 assert not selected.empty,'No validation-selected forecast in actual results'
 assert not selected.duplicated('horizon').any(),'Ambiguous selected forecast'
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white','axes.grid':True,'grid.alpha':0.2})
 wanted=list(dict.fromkeys([*selected.model.tolist(),'ARIMA','SARIMA','VAR_POLICY']))
 fig,axes=plt.subplots(1,2,figsize=(13.4,5.5),gridspec_kw={'width_ratios':[1.15,1]})
 periods=sorted(fc.period.unique());x=np.arange(len(periods));lookup={p:i for i,p in enumerate(periods)}
 for model in wanted:
  rows=fc[fc.model==model].sort_values('period')
  if rows.empty:continue
  label=model+(' — вибір за validation' if model in selected.model.values else '')
  axes[0].plot([lookup[p] for p in rows.period],rows.headline_yoy_pct,marker='o',color=COLORS.get(model),linewidth=2.7 if model in selected.model.values else 1.7,label=label)
  mm=metrics[(metrics.split=='audit')&(metrics.model==model)].sort_values('horizon')
  axes[1].plot(mm.horizon,mm.MAE_pp,marker='o',color=COLORS.get(model),linewidth=2.7 if model in selected.model.values else 1.7,label=model)
 axes[0].set(xticks=x,xticklabels=periods,ylabel='ІСЦ, % р/р, місячний NSA',title='Умовний статистичний прогноз')
 axes[0].legend(fontsize=8.3,loc='best')
 axes[1].set(xticks=sorted(metrics.horizon.unique()),xlabel='Горизонт прогнозу, місяців',ylabel='MAE річної інфляції, в. п.',title='Помилка на хронологічному аудиті')
 axes[1].legend(fontsize=8.5)
 ns=metrics[metrics.split=='audit'].groupby('horizon').n.first()
 fig.suptitle('Прогноз інфляції та його історична похибка',fontsize=15,fontweight='bold',y=.98)
 fig.text(.055,.035,'Джерело: nowcast/cpi_forecast.csv та cpi_metrics.csv. Навчання до 2026-09; зріз 22.10.2026.\nВибір за validation, не за аудитом. Переглянуті вінтажі; аудит не є справжнім real-time тестом.\nПомилки — контекст точності, не довірчий інтервал. Число оцінок за горизонтом: '+', '.join(f'{h}: {n}' for h,n in ns.items())+'.',fontsize=8.5,ha='left',va='bottom')
 fig.subplots_adjust(left=.07,right=.975,top=.85,bottom=.25,wspace=.28)
 fig.savefig(output/'satellite_forecast_audit.png',dpi=180);plt.close(fig)
 # Controlled scenario, identical shocks and policy regime; rate held only at Q4 announcements.
 scenario='satellite_bridge';variant='selected'
 chosen=paths[(paths.variant==variant)&(paths.scenario==scenario)].copy()
 assert not chosen.empty,'Missing requested selected satellite_bridge path'
 fig,axes=plt.subplots(1,2,figsize=(13.4,5.4))
 rates=[15.5,16.0,16.5];policy_colors={15.5:'#ca8622',16.0:'#64748b',16.5:'#153f75'};policy_rows=[]
 for rate in rates:
  q=chosen[np.isclose(chosen.october_rate,rate)&np.isclose(chosen.december_rate,rate)].sort_values('period').head(8)
  assert len(q)==8,f'Missing first eight quarters for {rate}'
  assert not q.period.duplicated().any()
  label=f'{_fmt(rate,1)}% у двох рішеннях Q4'
  axes[0].plot(range(8),q.delta_inflation_pp_vs_hold16,marker='o',color=policy_colors[rate],linewidth=2,label=label)
  axes[1].plot(range(8),q.delta_gdp_percent_vs_hold16,marker='o',color=policy_colors[rate],linewidth=2,label=label)
  mi=q.loc[q.delta_inflation_pp_vs_hold16.abs().idxmax()]
  mg=q.loc[q.delta_gdp_percent_vs_hold16.abs().idxmax()]
  policy_rows.append({'rate':rate,'q4_average_rate':q.q4_average_rate.iloc[0],'max_abs_delta_cpi_pp':abs(mi.delta_inflation_pp_vs_hold16),'cpi_peak_period':mi.period,'max_abs_delta_gdp_pct':abs(mg.delta_gdp_percent_vs_hold16),'gdp_peak_period':mg.period})
 for ax in axes:
  ax.axhline(0,color='#374151',linewidth=.8);ax.set_xticks(range(8));ax.set_xticklabels(q.period,rotation=30,ha='right');ax.legend(fontsize=8.3)
 axes[0].set(title='Зміна інфляції відносно рішення 16%',ylabel='Різниця річної інфляції, в. п.\nсередній квартальний CPI, SA')
 axes[1].set(title='Зміна рівня реального ВВП відносно 16%',ylabel='Різниця рівня ВВП, %\nне зміна темпу зростання р/р')
 fig.suptitle('Однаковий шок → різні ставки → умовний компроміс',fontsize=15,fontweight='bold',y=.98)
 fig.text(.055,.025,'Джерело: decision/policy_paths.csv; selected / satellite_bridge. Зріз 22.10.2026.\nКонтроль стосується двох рішень Q4 2026; далі працює спільне ендогенне правило.\nШоки однакові між ставками; ефекти навчальної QPM не є емпірично ідентифікованими причинними оцінками.',fontsize=8.5,ha='left',va='bottom')
 fig.subplots_adjust(left=.08,right=.975,top=.85,bottom=.27,wspace=.3)
 fig.savefig(output/'conditional_policy_tradeoff.png',dpi=180);plt.close(fig)
 # Current criterion winner is reported as conditional, never promoted to a recommendation.
 base=winners[(winners.variant=='selected')&(winners.scenario=='satellite_bridge')&(winners.horizon_quarters==8)&np.isclose(winners.output_weight,1)&np.isclose(winners.smoothing_weight,.1)]
 assert len(base)==1,'Missing or ambiguous base criterion winner'
 b=base.iloc[0];boundary=_truth(winners.grid_boundary)
 gap=gaps[gaps.variant=='selected'].iloc[0];j=judgments[judgments.variant=='selected'].iloc[0]
 audit_selected=metrics[(metrics.split=='audit')&metrics.model.isin(selected.model)]
 h3=metrics[(metrics.split=='audit')&(metrics.horizon==3)&metrics.model.isin(['VAR','SARIMA','VAR_POLICY'])].sort_values('MAE_pp')
 variants=sorted(paths.variant.unique());scenarios=sorted(paths.scenario.unique())
 maxres=max(float(num[c].abs().max()) for c in ['affine_path_max_error','structural_residual','constraint_residual'])
 history_files=list((output/'history').rglob('*.csv')) if (output/'history').exists() else []
 history_summary=[]
 for name in ['historical_variant_comparison.csv','historical_metrics.csv','historical_policy_replay.csv']:
  matching=[p for p in history_files if p.name==name]
  if matching:
   count=sum(len(pd.read_csv(p)) for p in matching);history_summary.append(f'{name}: {count} рядків у {len(matching)} файлах')
 fclines=[]
 for _,r in selected.iterrows():fclines.append(f"- {r.period}: {_fmt(r.headline_mom_pct)}% м/м; {_fmt(r.headline_yoy_pct)}% р/р; {r.model}.")
 policy_lines=[]
 for r in policy_rows:
  policy_lines.append(f"- {_fmt(r['rate'],1)}% в обох рішеннях Q4: середня ставка Q4 {_fmt(r['q4_average_rate'],4)}%; найбільша абсолютна різниця CPI за 8 кварталів {_fmt(r['max_abs_delta_cpi_pp'],3)} в. п. ({r['cpi_peak_period']}), рівня ВВП {_fmt(r['max_abs_delta_gdp_pct'],3)}% ({r['gdp_peak_period']}).")
 inputs=[now/'cpi_forecast.csv',now/'cpi_metrics.csv',decision/'policy_paths.csv',decision/'policy_losses.csv',decision/'conditional_winners.csv',decision/'initial_state_sensitivity.csv',decision/'judgment_diagnostics.csv',decision/'protocol.json']
 hashes={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
 bridge_uncertainty=''
 if any(s in scenarios for s in ['bridge_low','bridge_high']):
  bridge_uncertainty='Додатково пораховано bridge_low/high. Їхні межі задані через похибку сателіта; це сценарна перевірка, не довірчий інтервал і не ймовірність події. Усі сценарії враховано в кількостях нижче.'
 gain=100*float(cal.get('validation_improvement_fraction',float('nan')))
 md=f'''# Фактичний запуск: дані, прогнози та умовний вибір політики

Зріз: **22.10.2026**. Цю записку створено зі збережених результатів запуску, без повторного підбору моделі за зручним висновком. Це дослідницький результат; остаточне рішення команди щодо ставки ще не затверджено.

## Що реально виконано

Розширення містить {data.get('observations','невідомо')} значень {data.get('series','невідомо')} рядів: {data.get('new_official_historical_series_observations','невідомо')} значення нових офіційних історій резервів/інтервенцій, {data.get('newly_extracted_existing_raw_observations','невідомо')} нові витяги компонентів CPI зі старої книги та {data.get('reused_existing_observations','невідомо')} явно повторно використаних якорів. Нові первинні ряди звірено з [графіком резервів НБУ](https://bank.gov.ua/ua/markets/international-reserves-allinfo/dynamics) та [сторінкою інтервенцій НБУ](https://bank.gov.ua/ua/markets/currency-interventions). Походження кожного значення, одиниці, клітинки/API й SHA-256 записано в data/policy/2026-10-22/long.csv та маніфесті. Невідомі історичні дати публікації не вигадано.

Виконано статистичні сателіти {', '.join(fc.model.unique())}; прогноз CPI відокремлено від рекомендації ставки. Перевірено {cal.get('candidate_count','невідомо')} обмежених калібрувань QPM. Найкраще поліпшення validation становило {_fmt(gain)}%, за необхідних 5%; збережено виправлений початковий шаблон. Ставка не входила в ціль вибору калібрування.

Політичний запуск містить **{len(paths):,} рядків траєкторій**, **{len(losses):,} оцінок втрат**, **{len(winners):,} умовних переможців**, {len(variants)} варіантів моделі та {len(scenarios)} сценаріїв. Варіанти: {', '.join(variants)}. Сценарії: {', '.join(scenarios)}. Кількості сценаріїв і переможців не є ймовірностями. Максимальний перевірений числовий залишок: {maxres:.2e}; це перевірка розв'язання рівнянь, не економічної правдивості.

## Власний прогноз загальної інфляції

Модель обрана за validation, окремо за горизонтом. Поточне навчання закінчується вереснем 2026. Усі числа нижче — прогнози місячного CPI NSA, а не майбутні факти:

{chr(10).join(fclines)}

![Прогноз і помилки аудиту](satellite_forecast_audit.png)

Аудит не використано для зміни переможця validation. На горизонті 3 місяці фактичний MAE річної інфляції становить:

{_bullet_table(h3,[('model','модель'),('MAE_pp','MAE, в. п.'),('RMSE_pp','RMSE, в. п.'),('n','оцінок')])}

SARIMA на цьому аудиті може бути точнішою за вибраний VAR. Додавання валютного сигналу у VAR_POLICY не довело прогнозної переваги. Наявність нового показника не є підставою автоматично збільшити його вагу або оголосити його причинним ефектом ставки. Дані переглянуті; лаги доступності приблизні, дати аудиту раніше частково переглядалися. Це хронологічна діагностика, не незайманий real-time тест. RMSE не перетворюється автоматично на довірчий інтервал.

## Що показав контрольований експеримент

Рішення моделюються на {', '.join(protocol.get('decision_window',[]))}; набрання чинності наступного дня — сценарна конвенція. Квартальна формула: `{protocol.get('calendar_formula','див. protocol.json')}`. Через квартальну частоту два різні внутрішньоквартальні шляхи з однаковим середнім дають однаковий макроефект. Повний календар не верифіковано наживо через HTTP 403 — це обмеження протоколу.

У selected / satellite_bridge майбутні ціни прив'язано до validation-вибраного статистичного прогнозу. Для узгодження NSA-місяців із SA-квартальним рівнем повторено торішні сезонні фактори. Виведений шок витрат — залишок, потрібний для цього умовного мосту, а не доведений чистий енергетичний шок. Після його обчислення всі немонетарні шоки, включно із цим шоком витрат, зафіксовано однаково між варіантами ставки. {bridge_uncertainty}

Ставки 15,5 / 16 / 16,5% нижче стосуються **двох оголошень Q4 2026**; далі працює спільне ендогенне правило. Це не обіцянка тримати інструмент незмінним 8 кварталів.

{chr(10).join(policy_lines)}

![Умовна різниця інфляції та рівня ВВП](conditional_policy_tradeoff.png)

Лівий графік — різниця річної інфляції середнього квартального CPI SA, в. п.; правий — різниця **рівня** реального ВВП у %, а не різниця його темпів р/р. Обидва порівнюються з 16% за того самого шоку. Місячний NSA прогноз вище та квартальний SA прогноз QPM — різні показники.

## Чому це ще не остаточна ставка

За базовими нормативними вагами інфляція=1, gap=1, зміна ставки=0,1 і горизонтом 8 кварталів мінімум на перевіреній сітці для selected / satellite_bridge дає **{_fmt(b.october_rate,1)}% у жовтневому та {_fmt(b.december_rate,1)}% у грудневому рішенні**. Середня Q4 — {_fmt(b.q4_average_rate,4)}%. Це умовний результат заданих ваг, діагнозу та трансмісії; **його не прийнято як рекомендацію команди**. Число не слід приховувати чи замінювати бажаними 16–16,5%, але й не можна називати встановленим оптимумом НБУ.

Серед усіх перевірених комбінацій **{int(boundary.sum())} із {len(winners)}** переможців лежать на межі сітки. Межовий результат вказує на обмеження пошуку. За інших ваг, gap або тривалості шоку переможці змінюються; множина сценаріїв не оцінює імовірність кожної ставки. Ваги функції втрат нормативні, не оцінені з функції добробуту НБУ. Прогнозна калібровка не встановлює причинної сили процентного каналу.

Поточний gap у вибраному варіанті — **{_fmt(gap.gap_log_pp,3)} логарифмічного в. п.** на 2026Q3. За перевіреними припущеннями діапазон **{_fmt(gaps.gap_log_pp.min(),3)}…{_fmt(gaps.gap_log_pp.max(),3)}**. Це сценарна чутливість, не довірчий інтервал; факт ВВП доступний лише до 2026Q2.

Істотна неузгодженість експертного припущення: заданий gap 2022Q2 {_fmt(j.value,1)}, фільтр дає {_fmt(j.filtered_gap,3)}, розбіжність {_fmt(j.measurement_minus_filtered_gap,3)} логарифмічного в. п., або {_fmt(abs(j.measurement_misfit_in_declared_std),2)} задекларованих стандартних відхилень вимірювання. Результат вимірює конфлікт припущень, а не точність знання воєнного потенціалу. Перед остаточним рішенням слід перевірити вагу цього judgment та альтернативні сценарії потенціалу.

Міст до статистичного прогнозу потребує шоку витрат **{_fmt(bridge_cost)} аннуалізованого квартального логарифмічного в. п.** за шаблонного стандартного відхилення 1. Це велика розбіжність між структурним і статистичним прогнозами. Її не можна автоматично назвати виміряним шоком пропозиції або використати як доказ необхідності різкого підвищення ставки.

## Історична перевірка й межі доказу

Збережені історичні результати: {'; '.join(history_summary) if history_summary else 'на момент підсумку придатні CSV ще не знайдені'}. Наявність replay або малих залишків не доводить, що історичне рішення спричинило успіх. Треба розрізняти справжні вінтажі, поточну переглянуту історію, задані фактичні зовнішні умови та модельні альтернативи. Подальше падіння інфляції могло залежати від пропозиції, енергії, курсу, допомоги та очікувань.

Навчальна QPM не ремонтує потужності; шок потенціалу, шок попиту й шок витрат слід розділити. У резервів та інтервенцій різна дата обліку; залишок зміни резервів не використано як вигаданий фіскальний імпульс. Історичні дати публікації не реконструйовано повністю. Тому підсумок запуску — **відтворюване порівняння умовних наслідків**, а не доведена причинна або остаточна політична рекомендація.

## Відтворення та файли

Числа зчитано безпосередньо з nowcast/cpi_forecast.csv, cpi_metrics.csv; decision/policy_paths.csv, policy_losses.csv, conditional_winners.csv, initial_state_sensitivity.csv, judgment_diagnostics.csv, numerical_checks.csv; протоколів калібрування, сателіта й policy-run. Підсумок створює `tools/summarize_policy_run.py`; функція `summarize(output)` не змінює моделі чи дані. Вхідні хеші записано в SUMMARY_INPUT_HASHES.json.
'''
 (output/'RUN_SUMMARY_UA.md').write_text(md)
 (output/'SUMMARY_INPUT_HASHES.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2)+'\n')
 return {'summary':str(output/'RUN_SUMMARY_UA.md'),'figures':[str(output/'satellite_forecast_audit.png'),str(output/'conditional_policy_tradeoff.png')],'path_rows':len(paths),'loss_rows':len(losses),'scenario_count':len(scenarios)}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=DEFAULT);args=ap.parse_args();print(json.dumps(summarize(args.output),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
