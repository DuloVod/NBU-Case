"""Readable report and figures for the next-month NBU rate forecast."""
from pathlib import Path
import html,json,os,tempfile,re
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'nbu-monthly-plots'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import ROOT

LABELS={'QPM_BRIDGE':'QPM + місячний міст','ARIMA':'ARIMA','SARIMA':'SARIMA','VAR':'VAR'}
COLORS={'QPM_BRIDGE':'#16635b','ARIMA':'#275b96','SARIMA':'#b66c24','VAR':'#75518c'}


def generate():
    results=ROOT/'results/2026-10-22-monthly'
    output=ROOT/'reports/2026-10-22-monthly';output.mkdir(parents=True,exist_ok=True)
    target=pd.read_csv(results/'next_month_forecast.csv')
    metrics=pd.read_csv(results/'metrics.csv')
    backtest=pd.read_csv(results/'backtest_predictions.csv')
    manifest=json.loads((results/'run_manifest.json').read_text())
    baseline=manifest['current_effective_rate']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(11,5.7),layout='constrained')
    positions=np.arange(len(target))
    for position,row in zip(positions,target.itertuples()):
        ax.errorbar(row.mean,position,xerr=[[row.mean-row.lower95],[row.upper95-row.mean]],fmt='o',
                    color=COLORS[row.model],capsize=5,lw=2,markersize=7)
        ax.text(row.mean+.08,position-.13,f'{row.mean:.2f}%',color=COLORS[row.model],fontsize=11)
    ax.axvline(baseline,color='#626a73',ls='--',label=f'Незмінна чинна ставка: {baseline:.2f}%')
    ax.set(yticks=positions,yticklabels=[LABELS[m] for m in target.model],xlabel='Середня ставка за листопад, % річних',
           title='Листопад 2026: місячні прогнози станом на 22 жовтня')
    ax.set_ylim(len(target)-.5,-.5)
    ax.grid(axis='x',alpha=.2);ax.legend(frameon=False,loc='upper right')
    ax.text(0,-.2,'Крапки — умовні середні, відрізки — модельні 95%-і інтервали. Джерела: НБУ та датасет команди УКУ.',
            transform=ax.transAxes,fontsize=9,color='#535b62')
    fig.savefig(output/'november_forecast.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for ax,metric in zip(axes,['MAE_pp','RMSE_pp']):
        values=[float(metrics.loc[metrics.model==model,metric].iloc[0]) for model in LABELS]
        benchmark=float(metrics['baseline_'+metric].iloc[0])
        ax.bar(np.arange(5),values+[benchmark],color=[COLORS[m] for m in LABELS]+['#626a73'])
        ax.set(xticks=np.arange(5),xticklabels=['QPM + міст','ARIMA','SARIMA','VAR','Незмінна'],
               ylabel='Помилка, відсоткових пунктів',title=metric.replace('_pp',''))
        ax.tick_params(axis='x',rotation=25);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Прогноз наступного місяця з інформацією на 22-ге число: 92 однакові зрізи')
    fig.savefig(output/'monthly_accuracy.png',dpi=180);plt.close(fig)
    means={row.model:row.mean for row in target.itertuples()}
    text=f'''# Прогноз середньої облікової ставки за листопад 2026 року

Інформаційна дата: 22 жовтня 2026. Ціль: середня ставка за всі 30 календарних днів листопада, % річних.
Це не середня ставка за IV квартал і не ставка на кінець місяця. Ціль уточнена користувачем 22.10.2026.

## Результат

- QPM + місячний міст: **{means['QPM_BRIDGE']:.2f}%**.
- ARIMA: **{means['ARIMA']:.2f}%**.
- SARIMA: **{means['SARIMA']:.2f}%**.
- VAR: **{means['VAR']:.2f}%**.
- Орієнтир «чинна ставка не зміниться»: **{baseline:.2f}%**.

Ці неперервні числа є умовними математичними очікуваннями середньої ставки; вони не означають, що НБУ оголосить рішення з точністю до сотих відсотка.

## Яка інформація відома

На офіційній сторінці архіву НБУ під час перевірки 22 жовтня остання зміна — 16% з 18 вересня 2026 року.
Датований знімок: data/snapshots/2026-10-22/policy_archive.json. Джерело: https://bank.gov.ua/ua/monetary/archive-rish.
Завершена місячна макроісторія закінчується вереснем 2026. Відома ставка за 1–22 жовтня та остання чинна ставка використані окремо.
Середня ставка за ВЕСЬ жовтень ще невідома. Жодного майбутнього значення за 23–31 жовтня або за листопад не оголошено фактом.

Моделі спочатку будують спільний прогноз жовтня та листопада з історії до вересня. Потім прогноз умовлюється на відомих жовтневих даних.
Жовтневий ряд має статус current_month_nowcast, листопадовий — next_month_forecast. Для роботи з ціллю використовуйте next_month_forecast.csv.

## Чому QPM тепер має місячний міст

Початковий шаблон НБУ квартальний. Його коефіцієнти не можна зробити місячними простим перейменуванням дат або діленням ставки на три.
Структурна QPM залишається квартальною; до неї додано явний стохастичний міст для розподілу квартального прогнозу на місяці.
Міст має випадкове блукання як попередню модель внутрішньоквартальної динаміки, а квартальне середнє — розподіл із QPM.
Ваги місяців пропорційні календарним дням. До додавання нових місячних спостережень зберігаються середнє й коваріація квартального прогнозу.
Після врахування відомих місячних ставок і поточного неповного місяця розподіл оновлюється. Це QPM + bridge, не нова оцінена місячна структурна QPM.
Макроісторія QPM закінчується попереднім повним кварталом; ВВП консервативно доступний ще на квартал раніше.

## Інші моделі

ARIMA оцінюється на місячній середній ставці. SARIMA має річний сезонний період 12 місяців, а не 4 квартали.
VAR використовує місячні зміни ставки НБУ, місячну інфляцію, зміни курсу та ставки ЄЦБ.
Квартальний ВВП не інтерполюється в удавані місячні спостереження. Ціни сезонно коригуються заново всередині кожного інформаційного зрізу.
Параметри статистичних моделей обираються лише за навчальною історією. Усі три моделі враховують неповний поточний місяць через спільний вимірювальний міст.

## Як враховано неповний жовтень

Два відомі показники — середня ставка за перші 22 дні й чинна ставка 22-го числа — є наближеними вимірюваннями ще невідомого повного місячного середнього.
Їхній історичний зсув, дисперсія помилки й взаємна кореляція оцінюються за завершеними попередніми місяцями. Для поточного прогнозу це 201 місяць.
Обидва показники застосовуються спільно через умовний нормальний розподіл, зі збереженням міжмісячної коваріації прогнозу.
Це статистична вимірювальна апроксимація. Вона не стверджує, що ставка залишиться незмінною до кінця жовтня, і не враховує повністю дискретний процес рішень НБУ.
Незалежність вимірювальних помилок від прогнозних інновацій є припущенням цього мосту; його якість перевіряється історично.

## Історичне порівняння

92 інформаційні дати: від 22 січня 2019 до 22 серпня 2026. Кожного разу прогнозується середня ставка НАСТУПНОГО місяця.
Усього 368 прогнозів, однакові дати для чотирьох моделей. Завершена навчальна історія закінчується попереднім місяцем; використано лише поточні дні до 22-го включно.
Цілі історичної перевірки: лютий 2019 — вересень 2026.

Незмінна ставка має MAE **{metrics.baseline_MAE_pp.iloc[0]:.3f} в. п.** і RMSE **{metrics.baseline_RMSE_pp.iloc[0]:.3f} в. п.**.
Вона краща за всі чотири моделі за обома метриками на загальній вибірці. Рейтинг попереднього квартального тесту не переноситься на місячну задачу.
Тому цей запуск не дає переконливого кількісного обґрунтування відхилення листопадового прогнозу від орієнтира близько 16%.
Модельні прогнози варто показувати як порівняння сценаріїв, а не обирати одну цифру лише через складність назви моделі.

## Межі результату

Залишаються поточні переглянуті вінтажі даних та наближені дати доступності макростатистики; це не повний архівний real-time backtest.
Рівняння й калібровка QPM — із шаблону 2026 року, застосовані ретроспективно. Чутливість мосту й параметрів потребує подальшої перевірки.
Модельні інтервали є умовними Gaussian-інтервалами; їхня ширина не є гарантованою оцінкою ризиків війни або майбутніх рішень НБУ.
Календар засідань і дискретні кроки зміни ставки явно не моделюються.
Це прогноз очікуваної ставки, не висновок про оптимальне рішення монетарної політики.

## Запуск

```sh
python models/run_monthly.py
python models/make_monthly_report.py
python -m unittest discover -s tests -v
```

Окремо: models/monthly_qpm.py, monthly_arima.py, monthly_sarima.py, monthly_var.py.
Архівний квартальний запуск збережено без перезапису в results/2026-10-21/ та reports/2026-10-21/.
'''
    (output/'REPORT_UA.md').write_text(text)
    rendered=target[['model','period','mean','lower80','upper80','lower95','upper95']].copy()
    rendered['model']=rendered.model.map(LABELS)
    rendered.columns=['Модель','Місяць','Середня ставка, %','Нижня 80%','Верхня 80%','Нижня 95%','Верхня 95%']
    score=metrics[['model','n','MAE_pp','RMSE_pp','coverage80','coverage95']].copy()
    score['model']=score.model.map(LABELS)
    score.columns=['Модель','Прогнозів','MAE, в. п.','RMSE, в. п.','Покриття 80%, частка','Покриття 95%, частка']
    blocks=[]
    for block in text.split('\n\n'):
        if block.startswith('# '):blocks.append('<h1>'+html.escape(block[2:])+'</h1>')
        elif block.startswith('## '):blocks.append('<h2>'+html.escape(block[3:])+'</h2>')
        else:
            paragraph=html.escape(block).replace('\n','<br>')
            paragraph=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',paragraph)
            blocks.append('<p>'+paragraph+'</p>')
    document='''<!doctype html><html lang="uk"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Середня ставка за листопад 2026</title><style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;line-height:1.65;color:#202c35}
    h1{font-size:30px}h2{margin-top:32px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:9px;text-align:right;border-bottom:1px solid #dbe0e3}
    th{background:#edf2f2}td:first-child,th:first-child{text-align:left}img{width:100%;height:auto}.scroll{overflow-x:auto}</style><body>'''
    document+=''.join(blocks[:2])+'<h2>Саме листопад: прогнози та модельні інтервали</h2><div class="scroll">'
    document+=rendered.to_html(index=False,border=0,float_format=lambda value:f'{value:.2f}')+'</div><img src="november_forecast.png">'
    document+='<h2>Точність на історичних місяцях</h2><div class="scroll">'+score.to_html(index=False,border=0,float_format=lambda value:f'{value:.3f}')+'</div>'
    document+='<p>Орієнтир незмінної ставки: MAE 0,459 в. п.; RMSE 1,626 в. п. Він точніший за всі чотири моделі.</p><img src="monthly_accuracy.png">'
    document+=''.join(blocks[2:])+'</body></html>'
    (output/'report.html').write_text(document)
    print(output/'report.html')


if __name__=='__main__':generate()
