"""Three-month figures and plain Ukrainian explanation of actual ensemble weights."""
from pathlib import Path
import html,json,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT,write_json

RESULT=ROOT/'results/2026-10-22-short-term'
REPORT=ROOT/'reports/final'
MONTHS=['2026-11','2026-12','2027-01']
LABELS=['Листопад 2026','Грудень 2026','Січень 2027']
NAMES={'NO_CHANGE':'Незмінна ставка','ARIMA':'ARIMA','SARIMA':'SARIMA','VAR':'VAR',
       'QPM_BRIDGE':'QPM + місячний міст','ENSEMBLE':'Ансамбль'}
COLORS={'NO_CHANGE':'#263238','ARIMA':'#2e64a4','SARIMA':'#b27527','VAR':'#755090',
        'QPM_BRIDGE':'#278274','ENSEMBLE':'#b13e53'}


def mdtable(frame):
    rows=[list(frame.columns)]+frame.astype(str).values.tolist()
    return '\n'.join(['| '+' | '.join(rows[0])+' |','| '+' | '.join(['---']*len(rows[0]))+' |']+
                     ['| '+' | '.join(row)+' |' for row in rows[1:]])


def build():
    REPORT.mkdir(parents=True,exist_ok=True)
    forecasts=pd.read_csv(RESULT/'forecast.csv');weights=pd.read_csv(RESULT/'ensemble_weights.csv')
    metrics=pd.read_csv(RESULT/'metrics.csv');protocol=json.loads((RESULT/'protocol.json').read_text())
    selected=forecasts.loc[forecasts.selected_by_validation].sort_values('horizon')
    fig,axes=plt.subplots(1,2,figsize=(13,5.4),layout='constrained')
    x=np.arange(3)
    for name in ['NO_CHANGE','ARIMA','SARIMA','VAR','QPM_BRIDGE','ENSEMBLE']:
        g=forecasts.loc[forecasts.model==name].sort_values('horizon')
        axes[0].plot(x,g.prediction,'o-',label=NAMES[name],color=COLORS[name],
                     linewidth=2.5 if name=='ENSEMBLE' else 1.4,alpha=1 if name=='ENSEMBLE' else .8)
    axes[0].set_title('Що прогнозує кожна модель');axes[0].legend(fontsize=8)
    axes[1].plot(x,selected.prediction,'o-',color=COLORS['NO_CHANGE'],label='Вибрано за validation')
    axes[1].fill_between(x,selected.lower80,selected.upper80,color='#278274',alpha=.16,label='Історичний 80% інтервал')
    for i,row in enumerate(selected.itertuples()):axes[1].annotate(f'{row.prediction:.2f}%',(i,row.prediction),xytext=(0,9),textcoords='offset points',ha='center')
    axes[1].set_title('Основний прогноз і його невизначеність');axes[1].legend(fontsize=8,loc='lower left')
    for ax in axes:
        ax.set_xticks(x,LABELS,fontsize=8);ax.set_xlabel('Цільовий місяць');ax.set_ylabel('Середня облікова ставка, % річних');ax.grid(alpha=.18)
    fig.suptitle('Три місяці: листопад 2026 — січень 2027. Інформація на 22.10.2026')
    fig.text(.5,-.025,'Джерело: власні розрахунки з даних НБУ. Середні за календарні дні; інтервали не гарантують покриття воєнних шоків.',ha='center',fontsize=9)
    fig.savefig(REPORT/'forecast_three_months.png',dpi=170,bbox_inches='tight');plt.close(fig)

    fig,ax=plt.subplots(figsize=(11,5.2),layout='constrained');left=np.zeros(3)
    for name in ['NO_CHANGE','ARIMA','SARIMA','VAR','QPM_BRIDGE']:
        values=weights.loc[weights.model==name].sort_values('horizon').weight.to_numpy()*100
        ax.barh(LABELS,values,left=left,color=COLORS[name],label=NAMES[name])
        for i,(start,v) in enumerate(zip(left,values)):
            if v>=6:ax.text(start+v/2,i,f'{v:.1f}%',ha='center',va='center',color='white',fontsize=9)
        left+=values
    ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Ефективна вага в ансамблі, %');ax.set_ylabel('Цільовий місяць')
    ax.set_title('З чого насправді складається ансамбль');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=3,fontsize=9)
    fig.text(.5,-.08,'Джерело: останні 36 завершених історичних цілей окремо для кожного горизонту. Не рівні ваги і не голосування моделей.',ha='center',fontsize=9)
    fig.savefig(REPORT/'ensemble_weights.png',dpi=170,bbox_inches='tight');plt.close(fig)

    audit=metrics.loc[metrics.split=='audit']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,col,title in zip(axes,['MAE_pp','RMSE_pp'],['Середня абсолютна помилка','RMSE: більша вага великих помилок']):
        for offset,name in [(-.17,'NO_CHANGE'),(.17,'ENSEMBLE')]:
            g=audit.loc[audit.model==name].sort_values('horizon')
            ax.bar(x+offset,g[col],width=.32,color=COLORS[name],label=NAMES[name])
        ax.set_xticks(x,['1 місяць','2 місяці','3 місяці']);ax.set_xlabel('Горизонт прогнозу');ax.set_ylabel('Помилка, в. п.');ax.set_title(title);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    fig.suptitle('Чи допомагає ансамбль: однакові 24 цільові місяці на кожному горизонті')
    fig.text(.5,-.03,'Джерело: хронологічний аудит, жовтень 2024 — вересень 2026. Ці дати вже були частково видимі в попередніх дослідженнях.',ha='center',fontsize=9)
    fig.savefig(REPORT/'monthly_audit.png',dpi=170,bbox_inches='tight');plt.close(fig)

    # Mechanical paths, not model-estimated effects on inflation or GDP.
    scenario=pd.DataFrame({'period':MONTHS,'hold_16':[16.,16.,16.],
        'hike_and_hold':[16.5,16.5,16.5],
        'hike_then_reverse_dec18':[16.5,(17*16.5+14*16)/31,16.],
        'hold_then_hike_dec18':[16.,(17*16+14*16.5)/31,16.5]})
    scenario.to_csv(RESULT/'calendar_scenarios.csv',index=False)
    fig,ax=plt.subplots(figsize=(11,4.5),layout='constrained')
    for col,label,color in [('hold_16','Збереження 16%','#263238'),('hike_and_hold','Підвищення до 16,5% і утримання','#278274'),
                             ('hike_then_reverse_dec18','16,5%, повернення до 16% з 18.12','#b27527'),
                             ('hold_then_hike_dec18','16%, підвищення до 16,5% з 18.12','#755090')]:
        ax.plot(x,scenario[col],'o-',label=label,color=color)
    ax.set_xticks(x,LABELS);ax.set_xlabel('Цільовий місяць');ax.set_ylabel('Середня ставка, % річних');ax.set_title('Короткострокові умовні сценарії рішень');ax.grid(alpha=.16);ax.legend(fontsize=8,loc='center left',bbox_to_anchor=(1,.5))
    fig.text(.5,-.035,'Механічні середні за днями: це припущення, а не прогноз грудневого рішення. У сценаріях немає січневої зміни ставки. Джерело: власний розрахунок.',ha='center',fontsize=8)
    fig.savefig(REPORT/'policy_tradeoffs.png',dpi=170,bbox_inches='tight');plt.close(fig)

    pivot=forecasts.pivot(index='model',columns='period',values='prediction').reindex(list(NAMES))
    pivot.index=[NAMES[x] for x in pivot.index];pivot=pivot.rename(columns=dict(zip(MONTHS,LABELS))).round(3).reset_index(names='Модель')
    weight_table=weights.pivot(index='period',columns='model',values='weight').reindex(MONTHS)
    weight_table=weight_table[['NO_CHANGE','ARIMA','SARIMA','VAR','QPM_BRIDGE']].mul(100).round(2).rename(columns=NAMES).reset_index(names='Місяць')
    interval_table=selected[['period','model','prediction','lower80','upper80']].round(3)
    audit_table=audit[['horizon','model','n','MAE_pp','RMSE_pp']].round(4)
    text=f'''# Короткострокове дослідження: три місяці

Інформаційна дата — **22 жовтня 2026**. Єдина поточна прогнозна область: **листопад 2026, грудень 2026, січень 2027**. Показник — середня облікова ставка за всі календарні дні кожного місяця, % річних. Далекі траєкторії не входять до висновків цього звіту.

## Прогноз і наглядні графіки

{mdtable(pivot)}

![Тримісячний прогноз](forecast_three_months.png)

Основний прогноз обирається окремо для кожного горизонту за помилкою на validation, а не за назвою моделі. Ось відібрані прогнози та історично калібровані інтервали:

{mdtable(interval_table)}

Інтервали побудовані з уже завершених минулих помилок відповідного горизонту. Це не гарантія ймовірності за воєнного структурного зламу.

## Які моделі є в ансамблі та як вони працюють

**ARIMA** дивиться на історію самої ставки: її попередні рівні/зміни й те, як модель помилялася. Вона продовжує оцінену часову закономірність. Не знає сама по собі про новий обстріл чи рішення Правління. Специфікація обирається за AICc тільки на навчальному минулому.

**SARIMA** додає до ARIMA можливість річного повторення через 12 місячних спостережень. Для облікової ставки сезонність не очевидна: це перевірювана гіпотеза, а не причина автоматично довіряти моделі більше.

**VAR** розглядає взаємозв'язки чотирьох рядів: змін ставки НБУ, місячної інфляції, змін UAH/EUR та змін ставки ЄЦБ. Прогнозує їх разом із лагами. Кількість лагів обирається за BIC, з перевагою стабільних і достатньо компактних моделей. Наявність зв'язку не доводить причинність.

**QPM + місячний міст** використовує квартальні економічні рівняння: попит, інфляцію, курс і правило ставки. Міст переводить розподіл квартального середнього в місячні значення, зберігаючи квартальне обмеження й враховуючи відомі ставки. Квартальну модель не перетворено на місячну простим перейменуванням дат; місячний ВВП не вигадується. Публікуємо тільки результати трьох цільових місяців.

**Незмінна ставка** — додатковий стабілізатор: чинні 16% зберігаються в прогнозному місяці. За рішеннями, які часто не змінюються, цей простий орієнтир може бути точним.

Саме **чотири моделі плюс стабілізатор** утворюють цей ансамбль. Ridge й локальні тренди з попереднього пошуку були конкурентами, але **не входили до чотиримодельного ансамблю**. Не змішуємо кількість перевірених кандидатів зі складом суміші.

## Як визначаються ваги

1. Для кожного горизонту окремо беремо останні 36 історичних цілей, які вже завершилися на дату прогнозу. Потрібно щонайменше 12.
2. Чотирьом моделям надаємо ваги, обернені до їхньої минулої MAE. Нижня межа помилки 0,05 в. п. запобігає надмірній вазі майже нульової оцінки.
3. Історичні помилки визначають, яку частину залишити цій суміші, а яку — незмінній ставці. Частка моделей обмежена 50%, щоб не переносити слабкий сигнал у надмірну зміну прогнозу. Це явний параметр методики.
4. Остаточний прогноз = вага незмінної ставки × 16% + зважені прогнози чотирьох моделей. Усі ваги невід'ємні й разом дорівнюють 100%. Вони не є ймовірностями майбутніх рішень.

Фактичні ефективні ваги поточного запуску, %:

{mdtable(weight_table)}

![Склад ансамблю](ensemble_weights.png)

Нульова або мала вага означає, що минулі дані не дають достатнього сигналу довіряти цій складовій сильніше. Вона не означає, що модель пояснює нуль відсотків економіки.

## Чи ансамбль кращий

Перевірені шість кандидатів: чотири моделі, незмінна ставка й ансамбль. Для кожної історичної дати моделі заново навчалися лише на доступному минулому; поточний місяць враховано як неповний, не як відому місячну ціль. Ваги та інтервали також використовують лише завершені цілі.

Validation закінчується вереснем 2024. На пізнішому аудиті — однакові 24 цільові місяці від жовтня 2024 до вересня 2026 для кожного горизонту. Кількість усіх історичних цілей різниться за горизонтом: найдальші ще не завершилися. Усередині одного горизонту всі шість моделей порівнюються на однакових датах.

{mdtable(audit_table)}

![Перевірка точності ансамблю](monthly_audit.png)

Правило вибору — validation MAE, потім RMSE. Не переобираємо модель за зручнішим audit-показником. Ці історичні дати вже були частково видимі в попередній роботі, а дані містять перегляди й приблизні лаги публікації: це хронологічний аудит, не незайманий real-time тест. Краща складність не гарантує кращої точності.

## Ставка, рекомендація і календар рішень

Прогноз фактичної ставки та бажане рішення — різні питання. Попередня робоча рекомендація 16,5% не стає автоматично прогнозом усіх трьох місяців і ще не є затвердженим командою рішенням. Наступні кроки потрібно переглядати за інфляцією, очікуваннями, валютним ринком і активністю.

Статистичні прогнози є неперервними очікуваннями; вони не означають, що НБУ встановить ставку з точністю до сотих. Грудневе та січневе рішення можуть змінити місячне середнє залежно від дати набрання чинності. Наприклад, зміна з 18 грудня впливає на 14 із 31 дня цього місяця.

![Умовні рішення на три місяці](policy_tradeoffs.png)

Цей останній графік — механічні календарні сценарії, не оцінка впливу на інфляцію або ВВП. Монетарна передача має лаги; з тримісячного прогнозу ставки не випливає гарантоване зниження інфляції до січня. Окремих місячних прогнозів ВВП із квартальної QPM тут немає.

## Відтворення й джерела

`python tools/run_research.py` — поточний тримісячний запуск; `python tools/verify_short_term.py` — незалежна перевірка. Дані й точні ваги: `results/2026-10-22-short-term/`. Вихідні дані НБУ, Eurostat/ЄЦБ та первинні файли описані в [аудиті джерел](../../docs/research/DATA_GAPS_2026_10_22_UA.md). [Офіційний календар НБУ](https://bank.gov.ua/ua/monetary/schedule). Первинна макроісторія завершена вереснем 2026; відомі ставки за 1–22 жовтня використано окремо. Припущення про незмінність майбутньої ставки не видаються за факти.
'''
    (REPORT/'RESEARCH_UA.md').write_text(text)
    def inline(s):
        s=html.escape(s);s=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',s)
        return re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',s)
    body=[];lines=text.splitlines();i=0
    while i<len(lines):
        line=lines[i]
        if line.startswith('| '):
            block=[]
            while i<len(lines) and lines[i].startswith('| '):block.append(lines[i]);i+=1
            rows=[[c.strip() for c in row.strip('|').split('|')] for row in block]
            body.append('<div class="scroll"><table>'+''.join('<tr>'+''.join('<'+('th' if j==0 else 'td')+'>'+inline(c)+'</'+('th' if j==0 else 'td')+'>' for c in row)+'</tr>' for j,row in enumerate(rows) if j!=1)+'</table></div>');continue
        m=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',line)
        if m:body.append(f'<img alt="{html.escape(m[1])}" src="{html.escape(m[2])}">')
        elif line.startswith('## '):body.append('<h2>'+inline(line[3:])+'</h2>')
        elif line.startswith('# '):body.append('<h1>'+inline(line[2:])+'</h1>')
        elif line:body.append('<p>'+inline(line)+'</p>')
        i+=1
    (REPORT/'report.html').write_text('<!doctype html><html lang="uk"><meta charset="utf-8"><title>Прогноз на три місяці</title><style>body{font:17px/1.6 system-ui;max-width:1100px;margin:36px auto;padding:0 20px;color:#263238}img{width:100%;height:auto}h2{margin-top:36px}.scroll{overflow:auto}table{font-size:13px;border-collapse:collapse;width:100%}td,th{padding:8px;text-align:right;border-bottom:1px solid #ddd}th{background:#edf3f2}a{color:#176b65}</style>'+''.join(body)+'</html>')
    write_json(RESULT/'summary.json',dict(as_of='2026-10-22',target_months=MONTHS,
               selected_forecasts=selected[['period','model','prediction','lower80','upper80']].to_dict('records'),
               ensemble_forecasts=forecasts.loc[forecasts.model=='ENSEMBLE',['period','prediction']].to_dict('records'),
               working_policy_recommendation=16.5,policy_status='not team-approved; not a three-month commitment'))
    print('Three-month report:',REPORT/'RESEARCH_UA.md')


if __name__=='__main__':build()
