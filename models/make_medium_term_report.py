"""Build the integrated Ukrainian research note from saved research outputs."""
from pathlib import Path
import csv,html,json,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT,write_json

RESULT=ROOT/'results/2026-10-22-research'
REPORT=ROOT/'reports/2026-10-22-medium-term'
LABELS={'cut_15_5':'Зниження до 15,5%','hold_16':'Збереження 16%',
        'hike_16_5':'Підвищення до 16,5%','endogenous_rule':'Правило QPM'}


def markdown_table(frame):
    # No optional tabulate dependency is required to reproduce this repository.
    rows=[list(frame.columns)]+frame.astype(str).values.tolist()
    return '\n'.join(['| '+' | '.join(rows[0])+' |','| '+' | '.join(['---']*len(rows[0]))+' |']+
                     ['| '+' | '.join(row)+' |' for row in rows[1:]])


def build():
    REPORT.mkdir(parents=True,exist_ok=True)
    monthly=pd.read_csv(RESULT/'monthly_metrics.csv')
    november=pd.read_csv(RESULT/'monthly_november_forecast.csv')
    protocol=json.loads((RESULT/'monthly_protocol.json').read_text())
    calibration=json.loads((RESULT/'qpm_calibration_protocol.json').read_text())
    paths=pd.read_csv(RESULT/'qpm_policy_paths.csv')
    differences=pd.read_csv(RESULT/'qpm_policy_differences.csv')
    loss=pd.read_csv(RESULT/'qpm_policy_losses.csv')
    evidence=json.loads((ROOT/'data/research/2026-10-22/manifest.json').read_text())
    qpm_metrics=pd.read_csv(RESULT/'qpm_calibration_metrics.csv')
    inflation_rmse=float(qpm_metrics.loc[(qpm_metrics.model=='selected')&(qpm_metrics.split=='audit')&(qpm_metrics.metric=='inflation_log_pp'),'RMSE'].iloc[0])
    test=monthly.loc[monthly.split=='test'].copy()
    selected=november.loc[november.selected_by_validation].iloc[0]
    selected_test=test.loc[test.model==selected.model].iloc[0]
    nominal={'NO_CHANGE':'Незмінна ставка','ARIMA':'ARIMA','VAR':'VAR',
             'ONLINE_BLEND_SHRINK':'Онлайнансамбль','QPM_BRIDGE':'QPM + міст'}
    chart=test.set_index('model').loc[list(nominal)]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,metric,title in zip(axes,['MAE_pp','RMSE_pp'],['Середня абсолютна помилка','Корінь середньоквадратичної помилки']):
        ax.barh([nominal[x] for x in chart.index],chart[metric],color=['#176b65']+['#697b8b']*4)
        ax.invert_yaxis();ax.set_xlabel('Помилка середньої ставки, в. п.');ax.set_ylabel('Модель')
        ax.set_title(title);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        for i,v in enumerate(chart[metric]):ax.text(v+.01,i,f'{v:.3f}',va='center',fontsize=9)
        ax.set_xlim(0,float(chart[metric].max())*1.2)
    fig.suptitle('Хронологічний аудит: 24 цільові місяці, жовтень 2024 — вересень 2026')
    fig.text(.5,-.025,'Джерело: власні розрахунки. Вибір моделі — за попереднім validation MAE; це не незайманий зовнішній тест.',ha='center',fontsize=9)
    fig.savefig(REPORT/'monthly_audit.png',dpi=160,bbox_inches='tight');plt.close(fig)

    subset=differences.loc[differences.shock_case=='no_additional_shock']
    periods=sorted(subset.period.unique());x=np.arange(len(periods))
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True,layout='constrained')
    for ax,variable,title,unit in zip(axes,['delta_inflation_yoy_pct','delta_l_gdp_gap','fx_percent_vs_hold'],
              ['Річна інфляція квартального середнього ІСЦ','Розрив випуску','Курс UAH/EUR'],['Різниця, в. п.','Різниця, лог-в. п.','Відхилення від 16%, %']):
        for policy,color in [('hike_16_5','#176b65'),('cut_15_5','#ac6a2a')]:
            g=subset.loc[subset.policy==policy]
            base=g.loc[g.variant=='selected'].set_index('period').loc[periods,variable]
            band=g.groupby('period')[variable].agg(['min','max']).loc[periods]
            ax.plot(x,base,color=color,label=LABELS[policy],linewidth=2)
            ax.fill_between(x,band['min'],band['max'],color=color,alpha=.12)
        ax.axhline(0,color='#888888',linewidth=.8);ax.set_title(title);ax.set_ylabel(unit);ax.grid(alpha=.15)
    axes[0].legend();axes[-1].set_xticks(x,periods,rotation=35);axes[-1].set_xlabel('Квартал')
    fig.suptitle('Вплив альтернативного рішення проти збереження 16%: умовні сценарії QPM')
    fig.text(.5,-.03,'Лінії: основний варіант. Заливка: мінімум/максимум перевірених калібрувань, НЕ довірчий інтервал.\n'
             'Припущення: рішення з 30.10 до 31.12, далі правило QPM; додаткових шоків немає. Джерело: власні розрахунки.',ha='center',fontsize=9)
    fig.savefig(REPORT/'policy_tradeoffs.png',dpi=160,bbox_inches='tight');plt.close(fig)

    base=paths.loc[(paths.variant=='selected')&(paths.shock_case=='no_additional_shock')]
    summary=base.loc[base.period.isin(['2026Q4','2027Q4','2028Q4','2029Q3']),
                     ['period','policy','ir_pol','inflation_yoy_pct','gdp_yoy_pct','l_gdp_gap','fx_uah_per_eur']].copy()
    summary.to_csv(RESULT/'policy_summary.csv',index=False)
    impact=subset.loc[(subset.variant=='selected')&(subset.policy=='hike_16_5')].set_index('period')
    one_year=impact.loc['2027Q3'];peak=float(impact.delta_inflation_yoy_pct.min())
    preferred=loss.loc[(loss.variant=='selected')&(loss.shock_case=='no_additional_shock')&
                       (loss.output_weight==1)&(loss.smoothing_weight==.1)&(loss.inflation_target_pct==5)&(loss.policy!='endogenous_rule')]
    public='https://bank.gov.ua/ua/news/all/'
    text=f'''# Архів розширеного аналізу: поза поточним місячним дослідженням

Цей матеріал збережений для відтворення попереднього підходу. Поточна ціль команди — листопад 2026, грудень 2026 і січень 2027; основний звіт — `reports/2026-10-22-research/RESEARCH_UA.md`.

Інформаційна дата: **22 жовтня 2026 року**. Робоче дослідження для NBU University Challenge 2026; рекомендація потребує погодження команди перед поданням. Не використано рішення 29 жовтня та листопадовий Інфляційний звіт.

## Рішення, яке пропонуємо обговорити

**Робоча рекомендація — помірне підвищення облікової ставки до 16,5% (+50 б. п.). Основна альтернатива — збереження 16%; зниження до 15,5% зараз має слабшу доказову основу.** Аргументи за посилення: прискорення вересневої загальної й базової інфляції, високі очікування, тиск на резерви. Аргументи проти: негативні шоки пропозиції й випуску та ще не завершена передача попередніх підвищень. Величина 50 б. п. — помірний досліджений крок, а не доведений математичний оптимум.

**Прогноз фактичної середньої ставки за листопад — {selected.prediction:.2f}%.** Це окреме питання: статистичний прогноз поведінки НБУ може відрізнятися від нашої рекомендації. Не підганяємо прогноз під бажане рішення.

## 1. Що змінилося в даних

- Вересень: інфляція **10% р/р**, **2,1% м/м**; базова **8,8% р/р**, **2% м/м**. Послуги **13,7% р/р**. НБУ підтверджує суттєве перевищення липневого прогнозу [коментар 12.10]({public}komentar-natsionalnogo-banku-schodo-rivnya-inflyatsiyi-u-veresni-2026-roku). Не екстраполюємо один сезонно не скоригований місяць множенням на 12.
- Очікування через 12 місяців: бізнес **11,5%**, домогосподарства **14,13%**, аналітики **8,26%**. Дати різні: бізнес опитаний 30.07–28.08, домогосподарства — останнє значення серпня, аналітики — вересня. Це не одна синхронна оцінка й не фактична майбутня інфляція.
- ІОДА вересня **48,23** при нейтральній межі 50; квартальний ІДО **100,8** проти 107,0 при межі 100. Сигнали слабшої активності не можна ігнорувати при виборі ставки.
- Резерви на 01.10 **47,1018 млрд USD**, за вересень **−3,2%**; балансові продажі валюти НБУ **5,4499 млрд USD** [реліз 07.10]({public}mijnarodni-rezervi-stanovili-471-mlrd-dol-ssha-za-pidsumkami-veresnya). Стабільний середній курс сам по собі не означає відсутності валютного тиску.

Новий доказовий датасет: **{evidence['rows']} спостереження, {evidence['variables']} показники**, з одиницями, джерелами, локаторами, статусами та датами публікації. Лише {evidence['publication_date_known_rows']} рядків мають підтверджену дату публікації використаного джерела; вона не обов'язково є датою першої публікації конкретного числа. Інші дати явно позначені як невідомі. Це не строгий історичний real-time датасет. Новий вересневий резервний snapshot використано для політичних ризиків, не для оцінювання VAR на одному спостереженні.

## 2. Прогноз: перевірка 12 кандидатів

Порівняні незмінна ставка, два локальні тренди, три ridge-моделі, ARIMA, SARIMA, VAR, QPM + місячний міст та два онлайнансамблі. Validation: 68 цільових місяців, лютий 2019 — вересень 2024; пізніший аудит: 24 місяці, жовтень 2024 — вересень 2026. Правило вибору — найнижчий validation MAE, далі RMSE. Вибрано **{protocol['selected_model']}**; за аудитом модель повторно не переобирали.

Історичні дати вже були видимі в попередньому дослідженні. Це чесно позначений хронологічний аудит, **не новий незайманий зовнішній тест**. Масштабування ridge, навчальні цілі, онлайнваги й інтервали використовують лише завершене минуле. Поточний неповний жовтень не є готовою міткою.

{markdown_table(test[['model','n','MAE_pp','RMSE_pp','coverage80','coverage95']].round(4))}

![Точність місячних прогнозів](monthly_audit.png)

VAR має нижчий audit RMSE, але вищий MAE, ніж незмінна ставка. Парний блоковий bootstrap різниць MAE не дає переконливих доказів переваги ARIMA чи VAR. Не обираємо метрику після того, як побачили зручного переможця.

Листопадовий прогноз **{selected.prediction:.2f}%**; історично калібрований інтервал 80%: **{selected.lower80:.2f}–{selected.upper80:.2f}%**, 95%: **{selected.lower95:.2f}–{selected.upper95:.2f}%**. Це симетричні квантилі минулих помилок, не гарантія покриття війни чи позачергових рішень. На {int(selected_test.n)} audit-місяцях відповідне покриття baseline {100*selected_test.coverage80:.1f}% та {100*selected_test.coverage95:.1f}%; вибірка мала й режим залежний. За [календарем НБУ](https://bank.gov.ua/ua/monetary/schedule) планові рішення — 29 жовтня та 17 грудня. Якщо жовтневе рішення діятиме весь листопад без позачергової зміни, середня листопада дорівнюватиме його рівню.

## 3. Калібрування QPM: що перевірено і чого не доведено

Перевірено **{calibration['candidate_count']} кандидатів**: курсове згладжування, сила реакції на інфляцію, інерція ставки, пам'ять інфляції й масштаб цінового шоку. Структурні рівняння та первинний шаблон збережені. Коефіцієнти перебираються в явних межах; це не повне Bayesian/MLE-оцінювання і не причинна ідентифікація.

Масштаби помилок визначені на development-origin 2015Q4–2019Q4. Налаштування порівнювалися на validation-origin 2020Q1–2022Q4: однаково зважена стандартизована Huber-втрата прогнозів ставки, інфляції, курсу й ВВП на один квартал. Пізніший audit-origin 2023Q1–2026Q2 не використано для відбору. Для останнього GDP2026Q3 факту немає: його не рахують як наявну ціль.

Найкраще налаштування поліпшило validation-критерій на **{100*calibration['validation_improvement_fraction']:.2f}%**. За попередньо заданим правилом потрібні щонайменше 5%, тому основний варіант збережений: **навчальне калібрування**. Найкращий validation-кандидат залишений у перевірці чутливості. Не стверджуємо, що калібрування стало «ідеальним» лише тому, що сітка велика; 12 validation-кварталів замало для впевненого оцінювання багатьох параметрів.

## 4. Три рішення в структурних сценаріях

Порівнюємо **15,5 / 16 / 16,5%**. Експеримент припускає: ставка 16% до 29 жовтня включно; новий рівень з 30 жовтня до 31 грудня; з 2027Q1 — ендогенне правило. Тоді середня Q4: **15,6576 / 16 / 16,3424%**. Це гіпотетичне утримання в Q4, а не прогноз або обіцянка не переглядати рішення на засіданні 17 грудня.

Рівняння розв'язано спільно на 12 кварталів, із заздалегідь заданими сценарними шоками та стабільною термінальною умовою. Немає підміни раціональних очікувань механічним додаванням незалежних імпульсів. Структурні сценарії стартують із відфільтрованого стану 2026Q3; доступні жовтневі курсові спостереження не введені як додаткові вимірювання, а повний жовтневий ІСЦ на 22 жовтня ще невідомий. Квартальна модель не відтворює точний денний час передачі рішення всередині жовтня.

За основного калібрування підвищення до 16,5% проти 16% дає в **2027Q3**:

- річна інфляція квартального середнього нижча на **{-one_year.delta_inflation_yoy_pct:.3f} в. п.**;
- розрив випуску нижчий на **{-one_year.delta_l_gdp_gap:.3f} лог-в. п.**;
- UAH/EUR нижчий на **{-one_year.fx_percent_vs_hold:.3f}%**, тобто модельна гривня трохи сильніша.

Найбільше модельне зниження інфляції становить близько **{-peak:.3f} в. п.**. Для масштабу: одноквартальний audit RMSE інфляції QPM близько **{inflation_rmse:.3f} лог-в. п.** Це різні об'єкти/горизонти, але порівняння показує, чому маленьку різницю сценаріїв не можна представляти як точно виміряний емпіричний ефект.

![Модельні компроміси](policy_tradeoffs.png)

Перевірено відсутність додаткових шоків, умовний ціновий шок +2 річні лог-в. п. в Q4 та шок премії за ризик +2 в. п. Величини шоків — припущення, а не оцінений прогноз чи їх ймовірності. Варіанти курсового згладжування, довгострокової премії та найкращого validation-калібрування змінюють результати. Окремо перевірено точне перетворення цінових цілей: **5 лог-% у шаблоні відповідає 5,1271% звичайного темпу**, а точним 5% відповідає 4,8790 лог-%. Ця різниця порівнянна з малим модельним ефектом рішення; вона не має приховано визначати точність рекомендації. Додано варіант із точними цілями України/єврозони та чутливість втрат до 5% / 5,1271%. Заливка на графіку — діапазон перевірених параметричних варіантів, **не довірчий інтервал**.

Таблиця нижче — **умовний макросценарій основного калібрування без додаткових шоків**. Автоматична траєкторія ще не є повністю узгодженим із новими ризиками фінальним макропрогнозом команди: для цього потрібні погоджені зовнішні, фіскальні й енергетичні припущення. Інфляція — зміна сезонно скоригованого квартального середнього ІСЦ до відповідного кварталу попереднього року, **не грудень до грудня і не місячна інфляція**. GDPyoy — зміна модельного квартального реального ВВП; gap — латентна модельна оцінка. Курс — грн/EUR, не грн/USD. Не трактувати ці значення як усі вже затверджені прогнозні орієнтири команди.

{markdown_table(summary.round(3))}

Різке довгострокове зниження ставки залишається властивістю шаблону: припущення про нейтральну ставку, тренди, відсутність нових шоків і повернення до правила. Воно не доводить, що НБУ фактично знизить ставку за цією траєкторією. Окремої структурної моделі базової інфляції, резервів, бюджету та валютних обмежень тут немає.

## 5. Чому не називаємо результат математичним оптимумом

Діагностична функція втрат — інфляційне відхилення від цілі, розрив випуску та зміни ставки. Перевірено ваги випуску 0 / 0,25 / 1 / 2 і згладжування ставки 0 / 0,1 / 1, а також цілі 5% і 5,1271% для перевірки логарифмічного наближення. Це припущення дослідника, не оцінені суспільні переваги. Нормативна ціль НБУ залишається 5%.

За ваг випуску 1 і згладжування 0,1 в основному варіанті:

{markdown_table(preferred[['policy','discounted_loss']].round(5))}

За інших допустимих ваг і параметрів перемагають також збереження та зниження. Кількість перемог у сітці не є ймовірністю правильності рішення. Вибір 16,5% захищаємо передусім як обережну реакцію на інфляційні й валютні ризики, з чітким визнанням втрат випуску та модельної невизначеності.

## 6. Операційний дизайн і валютна політика

Чинні підтверджені параметри: ДС овернайт **16%**, забезпечені кредити овернайт **20%**; останній 14-денний тендер рефінансування **20%**. З 07.08 тримісячні ДС — **процентні тендери з оголошеним обсягом раз на два тижні**, а не старий фіксований спред [модернізація]({public}nbu-rozpochinaye-modernizatsiyu-operatsiynogo-dizaynu-protsentnoyi-politiki).

Наша пропозиція при 16,5%: зберегти ДС овернайт на рівні облікової; зберегти спред забезпеченого кредиту овернайт +400 б. п., тобто **20,5%**. Це явна пропозиція, не автоматичне чинне правило. Зберегти ринковий формат 3M-ДС і зв'язок із строковими депозитами; точний обсяг наступного тендера без розрахунку ліквідності не призначаємо. Стежити за UONIA, депозитними ставками/обсягами та результатами аукціонів.

Валютна рекомендація: **керована гнучкість, згладжування надмірних коливань, обережна поетапна лібералізація за передумовами**. Не обіцяти фіксований курс чи широку негайну відміну капітальних обмежень за підвищених ризиків резервів. Повний юридичний аудит постанови №18 не виконано; конкретні ліміти переказів/репатріації не пропонуємо змінювати без окремої перевірки та оцінки попиту.

## 7. Умови перегляду й комунікація

**На користь збереження замість підвищення:** стійке уповільнення сезонно скоригованого базового тиску, ознаки глибшого падіння попиту та зайнятості, стабілізація очікувань і валютного попиту. **На користь подальшого посилення:** повторні широкі цінові сюрпризи, погіршення очікувань, стійкий валютний тиск. **Для зниження:** сукупні докази дезінфляції й охолодження попиту за забезпеченого зовнішнього фінансування. Не вигадуємо офіційних числових порогів цих реакцій.

Комунікаційний меседж: «Пропонуємо помірно посилити процентну політику для обмеження стійкого цінового тиску й підтримки гривневих заощаджень. Рішення не усуває воєнні руйнування, тому поєднуємо його з керованою курсовою гнучкістю й обережним підходом до обмежень. Наступні кроки залежать від інфляції, очікувань, валютного ринку та стану економіки».

## 8. Що є й чого ще немає

Є дані, відтворювані моделі, чесний відбір, 109 перевірок калібрування, структурні сценарії, альтернативи, ризики й робоча рекомендація. Немає доказу найкращої моделі серед усіх можливих, причинної оцінки кожного каналу, повного набору real-time вінтажів, фінальної командної згоди та конкурсної презентації. Результати 3M-аукціонів, депозитна трансмісія, виконання бюджету та календар допомоги залишаються наступними цільовими доповненнями. Технічні вимоги з листа НБУ ще не прочитані.

Основні докази: [політична записка](../../docs/research/POLICY_EVIDENCE_2026_10_22_UA.md), [аудит даних](../../docs/research/DATA_GAPS_2026_10_22_UA.md), [місячна методика](../../docs/methodology/MONTHLY_RESEARCH_UA.md), [протокол агентів](../../docs/methodology/AGENT_RESEARCH_PROTOCOL_UA.md). Числа й діагностика: `results/2026-10-22-research/`. Офлайновий запуск: `python tools/run_research.py`.
'''
    (REPORT/'RESEARCH_UA.md').write_text(text)
    # Render this deliberately small Markdown subset to a self-contained report.
    body=[];lines=text.splitlines();i=0
    def inline(value):
        value=html.escape(value)
        value=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',value)
        value=re.sub(r'`(.+?)`',r'<code>\1</code>',value)
        value=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',value)
        return value
    while i<len(lines):
        line=lines[i]
        if line.startswith('| '):
            block=[]
            while i<len(lines) and lines[i].startswith('| '):block.append(lines[i]);i+=1
            rows=[[c.strip() for c in row.strip('|').split('|')] for row in block]
            body.append('<div class="scroll"><table><thead><tr>'+''.join('<th>'+inline(c)+'</th>' for c in rows[0])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+inline(c)+'</td>' for c in row)+'</tr>' for row in rows[2:])+'</tbody></table></div>');continue
        image=re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)',line)
        if image:body.append(f'<img alt="{html.escape(image[1])}" src="{html.escape(image[2])}">')
        elif line.startswith('## '):body.append('<h2>'+inline(line[3:])+'</h2>')
        elif line.startswith('# '):body.append('<h1>'+inline(line[2:])+'</h1>')
        elif line.startswith('- '):body.append('<p class="bullet">• '+inline(line[2:])+'</p>')
        elif line:body.append('<p>'+inline(line)+'</p>')
        i+=1
    doc='<!doctype html><html lang="uk"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Дослідження УКУ — рішення НБУ</title><style>body{font:17px/1.65 system-ui,sans-serif;color:#202e35;max-width:1100px;margin:40px auto;padding:0 24px}h1{font-size:30px;line-height:1.25}h2{margin-top:40px;font-size:23px}a{color:#176b65}img{width:100%;height:auto;margin:22px 0}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:8px;border-bottom:1px solid #d8dfe1;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}thead{background:#edf3f2}.bullet{margin:8px 0}code{font-size:14px;background:#f0f2f2;padding:2px 5px}@media print{body{font-size:11px}h2{break-after:avoid}img{break-inside:avoid}}</style><body>'+''.join(body)+'</body></html>'
    (REPORT/'report.html').write_text(doc)
    write_json(RESULT/'medium_term_summary.json',dict(as_of='2026-10-22',
               forecast_model=protocol['selected_model'],november_average_forecast=float(selected.prediction),
               forecast80=[float(selected.lower80),float(selected.upper80)],forecast95=[float(selected.lower95),float(selected.upper95)],
               working_policy_recommendation=16.5,policy_status='conditional research recommendation, not team-approved final submission',
               primary_alternative=16.,qpm_candidates=calibration['candidate_count'],
               qpm_improvement_fraction=calibration['validation_improvement_fraction'],qpm_selected_settings=calibration['selected_settings'],
               evidence_rows=evidence['rows'],evidence_variables=evidence['variables'],
               hike_vs_hold_2027Q3=one_year.to_dict(),uncertainty='no claim of statistical/welfare optimality'))
    print('Integrated report:',REPORT/'RESEARCH_UA.md')


if __name__=='__main__':build()
