"""Create a readable Ukrainian report and scientific plots from saved results."""
from pathlib import Path
import html
import json
import os
import re
import tempfile
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'nbu-rate-models-matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT,DataStore,write_json

COLORS={'QPM':'#16635b','ARIMA':'#275b96','SARIMA':'#b66c24','VAR':'#75518c','No change':'#626a73'}


def table(frame):
    return frame.to_html(index=False,border=0,float_format=lambda x:f'{x:.2f}',classes='data-table',escape=True)


def generate():
    results=ROOT/'results'/'2026-10-21'; report=ROOT/'reports'/'2026-10-21';report.mkdir(parents=True,exist_ok=True)
    forecasts=pd.read_csv(results/'forecast_comparison.csv')
    metrics=pd.read_csv(results/'metrics.csv')
    backtest=pd.read_csv(results/'backtest_predictions.csv')
    regimes=pd.read_csv(results/'metrics_by_regime.csv')
    fold=json.loads((results/'fold_metadata.json').read_text())
    store=DataStore();target=store.target
    means=forecasts.pivot(index='period',columns='model',values='mean')[['QPM','ARIMA','SARIMA','VAR']]
    means.to_csv(results/'forecast_means.csv')
    comparison=metrics[['model','horizon','n','MAE_pp','RMSE_pp','coverage80','coverage95']].copy()
    # Latest forecast ranges based on past absolute OOS errors. Descriptive only:
    # policy regimes and overlapping forecasts violate exchangeability.
    empirical=[]
    for _,row in metrics.iterrows():
        sample=backtest[(backtest.model==row.model)&(backtest.horizon==row.horizon)]
        errors=(sample.prediction-sample.actual).abs().to_numpy()
        latest=forecasts[(forecasts.model==row.model)&(forecasts.horizon==row.horizon)].iloc[0]
        entry=dict(model=row.model,period=latest.period,horizon=int(row.horizon),n=len(errors),mean=latest['mean'])
        for coverage in [.8,.95]:
            rank=min(len(errors),int(np.ceil((len(errors)+1)*coverage)))
            width=float(np.sort(errors)[rank-1])
            entry[f'lower{int(coverage*100)}']=latest['mean']-width
            entry[f'upper{int(coverage*100)}']=latest['mean']+width
        empirical.append(entry)
    pd.DataFrame(empirical).to_csv(results/'empirical_error_ranges.csv',index=False)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.facecolor':'white','axes.titleweight':'bold'})
    fig,ax=plt.subplots(figsize=(11.5,5.5),layout='constrained')
    historical=target.loc['2023Q1':]
    forecast_index=pd.PeriodIndex(means.index,freq='Q')
    all_index=historical.index.append(forecast_index)
    ax.plot(np.arange(len(historical)),historical,label='Факт: середня ставка за квартал',color='#262b30',lw=2.4)
    for model in means:
        x=np.arange(len(historical)-1,len(all_index))
        y=np.r_[historical.iloc[-1],means[model].to_numpy()]
        ax.plot(x,y,label=model,color=COLORS[model],lw=2,marker='o',markersize=3)
    ax.axvline(len(historical)-.5,color='#8b939b',ls=':',lw=1)
    ax.set_xticks(np.arange(len(all_index))[::2],[str(x) for x in all_index[::2]],rotation=35,ha='right')
    ax.set(xlabel='Квартал',ylabel='Середня облікова ставка, % річних',
           title='Прогноз середньої ставки НБУ: 2026Q4–2028Q4')
    ax.grid(axis='y',alpha=.2);ax.legend(ncol=3,frameon=False,loc='upper right')
    ax.text(0,-.30,'Точкові умовні прогнози. Історія до 2026Q3. Джерела: НБУ та підготовлений датасет команди УКУ.',
            transform=ax.transAxes,fontsize=9,color='#535b62')
    fig.savefig(report/'forecast.png',dpi=180,bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11.5,4.8),layout='constrained')
    for ax,metric,title in zip(axes,['RMSE_pp','MAE_pp'],['RMSE: більша вага великих помилок','MAE: середня абсолютна помилка']):
        for model in ['QPM','ARIMA','SARIMA','VAR']:
            group=metrics[metrics.model==model].sort_values('horizon')
            ax.plot(group.horizon,group[metric],marker='o',label=model,color=COLORS[model],lw=2)
        baseline=metrics[metrics.model=='QPM'].sort_values('horizon')
        ax.plot(baseline.horizon,baseline['baseline_'+metric],marker='s',label='Ставка незмінна',color=COLORS['No change'],ls='--')
        ax.set(xlabel='Горизонт прогнозу, кварталів',ylabel='Помилка ставки, відсоткових пунктів',title=title,xticks=[1,2,4])
        ax.grid(axis='y',alpha=.2)
    axes[0].legend(frameon=False,fontsize=9)
    fig.savefig(report/'accuracy.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(11.5,5),layout='constrained')
    observed=backtest[(backtest.model=='QPM')&(backtest.horizon==1)]
    ax.plot(np.arange(len(observed)),observed.actual,label='Фактична ставка',color='#262b30',lw=2.5)
    for model in ['QPM','ARIMA','SARIMA','VAR']:
        group=backtest[(backtest.model==model)&(backtest.horizon==1)]
        ax.plot(np.arange(len(group)),group.prediction,label=model,color=COLORS[model],alpha=.85,lw=1.5)
    ax.set_xticks(np.arange(len(observed))[::4],observed.period.iloc[::4],rotation=30)
    ax.set(xlabel='Квартал, який прогнозували',ylabel='Середня ставка, % річних',title='Історичні прогнози на один квартал уперед')
    ax.legend(ncol=5,frameon=False,fontsize=9);ax.grid(axis='y',alpha=.2)
    fig.savefig(report/'backtest.png',dpi=180);plt.close(fig)

    qmeta=json.loads((results/'qpm/metadata.json').read_text())
    ameta=json.loads((results/'arima/metadata.json').read_text())
    smeta=json.loads((results/'sarima/metadata.json').read_text())
    vmeta=json.loads((results/'var/metadata.json').read_text())
    unstable=[x['origin'] for x in fold if x['model']=='var' and not x['metadata']['stable']]
    convergence_rejections=sum(1 for x in fold if x['model'] in ['arima','sarima']
                               for c in x['metadata']['candidates'] if not c.get('converged',False))
    write_json(results/'fit_diagnostics.json',dict(var_unstable_folds=unstable,
               rejected_unconverged_or_failed_arima_sarima_candidates=convergence_rejections,
               qpm_solver=qmeta['solver']))
    cols={'model':'Модель','horizon':'Кварталів уперед','n':'Кількість прогнозів',
          'MAE_pp':'MAE, в. п.','RMSE_pp':'RMSE, в. п.','coverage80':'Покриття 80%, частка',
          'coverage95':'Покриття 95%, частка'}
    primary=comparison.rename(columns=cols)
    first=means.iloc[0]
    text=f'''# Прогноз облікової ставки НБУ: QPM, ARIMA, SARIMA, VAR

Версія 21.10.2026. Історія показників обрізана вереснем 2026; жовтневі рішення або дані автоматично не додавалися.
Ціль — **середня за календарні дні кварталу облікова ставка, % річних**. Це не прогноз рішення на найближчому засіданні і не ставка на кінець кварталу.

## Отриманий прогноз

На IV квартал 2026: QPM **{first.QPM:.2f}%**, ARIMA **{first.ARIMA:.2f}%**, SARIMA **{first.SARIMA:.2f}%**, VAR **{first.VAR:.2f}%**.
Останній факт, III квартал 2026: **{target.iloc[-1]:.4f}%**. Повні дев'ять кварталів — у results/2026-10-21/forecast_means.csv та HTML-звіті.

## Як перевіряли

31 старт прогнозу: 2018Q4–2026Q2. На кожному старті доступна історія від 2010Q1.
Горизонти 1, 2, 4 квартали мають відповідно 31, 30, 28 результатів для КОЖНОЇ моделі.
На кожному старті повторно розраховано robust log-STL лише на відомій місячній історії.
ВВП України та єврозони доступний лише до попереднього кварталу. Інформаційний зріз умовно відповідає 21-му дню після завершення кварталу.
Поточні курси, інфляція й ставки доступні за завершений квартал. Це явне наближення до календаря публікацій, а не перевірка кожної історичної дати релізу.
Зовнішні майбутні факти та прогнози НБУ не використовуються як навчальні спостереження.

Показники якості: MAE — середня абсолютна помилка; RMSE сильніше карає великі помилки.
Одиниці обох — відсоткові пункти. Додано орієнтир «майбутня ставка дорівнює останній відомій середній ставці».

## Висновки порівняння

QPM має найменшу RMSE на загальній вибірці на всіх трьох горизонтах: 2.28, 4.12, 7.23 в. п.
Її перевага над незмінною ставкою за RMSE — приблизно 13.4%, 12.3%, 9.2%.
На один квартал орієнтир незмінної ставки має MAE 1.50 в. п., що краще за кожну з чотирьох моделей.
Отже, називати QPM безумовним переможцем за всіма критеріями не можна.

На цілях починаючи з 2022 року QPM має найменшу RMSE на 1 і 2 квартали; на 4 квартали ARIMA має RMSE 8.30 проти 8.38 у QPM.
SARIMA не покращила загальний результат проти ARIMA. Підстав вважати, що ставка має корисну для прогнозування сталу річну сезонність, цей тест не дав.
VAR поступається іншим на загальній вибірці; використовувати її як головну модель цей запуск не обґрунтовує.
Це описові результати невеликої вибірки з великими змінами режиму. Статистичної значущості переваги не встановлено.

## Невизначеність

Модельні Gaussian-інтервали — умовні за фіксованими параметрами. Для QPM 80%-й інтервал на один квартал покрив лише 48.4% фактів.
На чотири квартали — лише 32.1%. 95%-і інтервали також недопокривають. Отже, вони істотно занижують фактичну невизначеність.
У results/2026-10-21/empirical_error_ranges.csv окремо подано описові діапазони за минулими абсолютними помилками для горизонтів 1, 2, 4.
Вони не мають гарантованого майбутнього покриття через зміну режимів та залежність перекривних прогнозів.
Гаусівські довгі інтервали статистичних моделей можуть заходити нижче нуля; межі штучно не обрізалися.

## Що саме являє собою QPM

Прочитано 32 початкові лінійні рівняння та параметри з наданого model.model і param.m.
Для лагу інфляційного рівня -4 і очікувань +3 введено 5 допоміжних змінних: 37 станів.
Розв'язок раціональних очікувань отримано лінійною часовою ітерацією; перевірено матричні рівняння, шоки, афінну частину й число стійких узагальнених коренів.
Потім застосовано фільтр Калмана з пропущеними спостереженнями. Прогноз ставки формується всередині структурної системи.
Сталу траєкторію, вплив ставки на попит, ціновий і валютний канали взято з шаблону.

Відмінності від MATLAB/IRIS задокументовані, тому бітової тотожності результатів не заявлено:
- ціни мають дослідницьке STL-коригування, а не X-12/X-13;
- ставка — календарне середнє, дозволене описом джерел шаблону;
- alpha=1, beta=0.7, інфляційна ціль 5, зовнішня 2, інші структурні коефіцієнти — як у шаблоні;
- у param.m назва std_res_dl_cpi_c не відповідає res_dl_cpi; її значення 1 явно зіставлено з цим шоком;
- для потенційного ВВП у 2022Q1–Q2 явно дозволено структурний шок зі стандартним відхиленням 10 логарифмічних пунктів; це припущення нашої адаптації;
- ручні налаштування розриву ВВП 2015/2022 з демонстрації не оголошені фактами й не накладалися;
- початкова коваріація фільтра 100*I, вимірювальний шум 1e-6; це власна ініціалізація;
- розрив ВВП єврозони оцінено HP(1600) на доступній кожному зрізу історії, а не взято як спостережуваний факт;
- зовнішній прогноз створюють AR-рівняння шаблону; заповнених командою зовнішніх траєкторій ще немає.

Структурні коефіцієнти QPM НЕ оцінено економетрично. Шаблон 2026 року застосовано до минулих зрізів ретроспективно.
Фільтр оцінює приховані стани за цією калібровкою; це слід відрізняти від оцінювання коефіцієнтів моделі.

## Інші моделі

ARIMA: на кожному навчальному зрізі перебрано p=0..2, d=0..1, q=0..1; відкинуто незбіжні оцінки, вибір за AICc.
Поточний вибір: ARIMA{tuple(ameta['order'])}.
SARIMA: 4 неcезонні та 3 сезонні специфікації, річний період 4; вибір за AICc лише на навчальній історії.
Поточний вибір: {tuple(smeta['order'])} × {tuple(smeta['seasonal_order'])}.
VAR: 4 показники — зміна ставки, аннуалізована квартальна інфляція, зростання реального ВВП, зміна курсу.
Ставка прогнозується в змінах і повертається в рівень із накопиченням усієї міжперіодної коваріації.
Лаги 1 або 2, не менше 5 спостережень на коефіцієнт рівняння; BIC серед стійких варіантів, за їх відсутності нестійкість явно записується.
Невідоме поточне зростання ВВП оцінюється умовним VAR-прогнозом з урахуванням уже відомих показників, а не заповнюється майбутнім фактом.
Поточний вибір: VAR({vmeta['lags']}); нестійкі зрізи у фінальному тесті: {unstable or 'немає'}.

## Як задіяно всі наші дані

Повний знімок підготовленої бази збережено в data/. Програма читає й каталогізує всі 18 підготовлених модельних CSV.
Сім базових спостережень використовує QPM; ARIMA/SARIMA використовують історію ставки; малий VAR — чотири обґрунтовані показники.
Сировина, FAO, зарплата, безробіття, UONIA, кредитні ставки, торгівля й опитування залишаються доступними для контексту та подальших розширень.
Автоматичне додавання всіх цих колонок до малої вибірки спричинило б втрату історії та надмірну кількість параметрів.
results/2026-10-21/data_usage.csv фіксує роль кожної таблиці; results/2026-10-21/auxiliary_context.csv — останні значення й покриття додаткових рядів.
Прогнози НБУ збережені як орієнтир, не підміняють факти та не визначають автоматично наше прогнозне значення ставки.

## Межі перевірки

Історичні ВВП і ціни взято з поточного переглянутого вінтажу. Повного архіву «що саме знали в ту дату» немає.
Майбутні значення й сезонні фактори за межами зрізу не використовувалися; перевірено зміною майбутніх даних.
Затримки ВВП задано правилом, а не відновлено за всіма фактичними календарями публікацій.
Фіксована калібровка QPM походить із навчального шаблону 2026; це теж обмежує трактування результатів як справжнього історичного прогнозування.
Тривалий горизонт до 2028Q4 є умовним сценарієм, особливо за відсутності зовнішніх припущень команди.
Результат прогнозує ставку за моделлю, а не доводить оптимальність цієї ставки для рекомендації НБУ.

## Методичні посилання

- ARIMA: https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html
- SARIMA: https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
- VAR: https://www.statsmodels.org/stable/vector_ar.html
- Первинні статистичні джерела та їх точні адреси: data/documentation/sources.csv.
'''
    (report/'REPORT_UA.md').write_text(text)
    paragraphs=[]
    for block in text.split('\n\n'):
        if block.startswith('# '): paragraphs.append('<h1>'+html.escape(block[2:])+'</h1>')
        elif block.startswith('## '): paragraphs.append('<h2>'+html.escape(block[3:])+'</h2>')
        else:
            paragraph=html.escape(block).replace('\n','<br>')
            paragraph=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',paragraph)
            paragraphs.append('<p>'+paragraph+'</p>')
    mean_html=table(means.reset_index().rename(columns={'period':'Квартал'}))
    baseline=metrics[metrics.model=='QPM'][['horizon','baseline_MAE_pp','baseline_RMSE_pp']].rename(
        columns={'horizon':'Кварталів уперед','baseline_MAE_pp':'MAE незмінної ставки, в. п.',
                 'baseline_RMSE_pp':'RMSE незмінної ставки, в. п.'})
    intervals=forecasts[forecasts.horizon==1][['model','mean','lower80','upper80','lower95','upper95']]
    intervals=intervals.rename(columns={'model':'Модель','mean':'Прогноз, %','lower80':'Нижня 80%','upper80':'Верхня 80%',
                                      'lower95':'Нижня 95%','upper95':'Верхня 95%'})
    images='''<h2>Траєкторії та точність</h2><img src="forecast.png" alt="Порівняння прогнозів середньої квартальної ставки НБУ">
    <img src="accuracy.png" alt="RMSE і MAE на горизонтах один, два та чотири квартали">
    <img src="backtest.png" alt="Історичні прогнози на один квартал і фактична ставка">'''
    page='''<!doctype html><html lang="uk"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Прогноз ставки НБУ — порівняння чотирьох моделей</title><style>
    body{font-family:system-ui,sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#202c35;line-height:1.65}
    h1{font-size:30px;line-height:1.25}h2{margin-top:36px;font-size:23px}p{max-width:1000px}img{width:100%;height:auto;margin:12px 0}
    .data-table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}th,td{padding:9px 12px;text-align:right;border-bottom:1px solid #dbe0e3}
    th{background:#edf2f2}th:first-child,td:first-child{text-align:left}.scroll{overflow-x:auto}strong{color:#125d55}
    </style><body>'''
    page+=''.join(paragraphs[:2])+'<h2>Прогнози, % річних</h2><div class="scroll">'+mean_html+'</div>'
    page+='<h2>Модельні інтервали на 2026Q4</h2><p>Це умовні інтервали. Для QPM історичне покриття нижче заявленого.</p><div class="scroll">'+table(intervals)+'</div>'
    page+=images+'<h2>Однакові історичні тести</h2><div class="scroll">'+table(primary)+'</div>'
    page+='<h2>Орієнтир: ставка не змінюється</h2>'+table(baseline)
    page+=''.join(paragraphs[2:])+'</body></html>'
    (report/'report.html').write_text(page)
    print(f'Report: {report / "report.html"}')


if __name__=='__main__':generate()
