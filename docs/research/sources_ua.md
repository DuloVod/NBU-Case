# Остаточні джерела даних

Інформаційний зріз: основна база 20.10.2026, додаткові джерела 21.10.2026; останній місяць — вересень 2026, якщо вже опублікований.

23 джерельні файли й записи не означають 23 незалежні установи. У більшості українських рядів НБУ є поширювачем статистики Держстату.

Прогнози НБУ та відповіді опитувань не підставляються замість реалізованих макропоказників.

## 1. CPI_m.xlsx

Установа: **НБУ / Держстат**. [Першоджерело](https://bank.gov.ua/files/macro/CPI_m.xlsx).

Файл у репозиторії: [`data/raw/CPI_m.xlsx`](../../data/raw/CPI_m.xlsx).

Вибірка: 2: рядок 3, дати в рядку 2; 1: рядок 3.

Призначення й особливості: Загальна інфляція м/м та опублікована р/р. Основа цінового блоку. Місячні зміни вже у %, 100 не віднімати.

Результат перевірки: Прийнято.

## 2. CCPI_m.xlsx

Установа: **НБУ / Держстат**. [Першоджерело](https://bank.gov.ua/files/macro/CCPI_m.xlsx).

Файл у репозиторії: [`data/raw/CCPI_m.xlsx`](../../data/raw/CCPI_m.xlsx).

Вибірка: 1: рядки 4 та 6, дати в рядку 2.

Призначення й особливості: Базова інфляція м/м та р/р для перевірки стійкості цінового тиску. Не підміняє загальний ІСЦ.

Результат перевірки: Прийнято.

## 3. _dodatok_sezonka_vvp_2010-2026.xlsx

Установа: **Держстат**. [Першоджерело](https://stat.gov.ua/sites/default/files/2026-09/_dodatok_sezonka_vvp_2010-2026.xlsx).

Файл у репозиторії: [`data/raw/_dodatok_sezonka_vvp_2010-2026.xlsx`](../../data/raw/_dodatok_sezonka_vvp_2010-2026.xlsx).

Вибірка: 2010-2026: C8:BP8, дати C4:BP4.

Призначення й особливості: Реальний ВВП, млн грн у цінах 2021 року, сезонно скоригований. Основа розриву випуску, який ще треба оцінити.

Результат перевірки: Прийнято.

## 4. PInterestRates.xls

Установа: **НБУ**. [Першоджерело](https://bank.gov.ua/files/PInterestRates.xls).

Файл у репозиторії: [`data/raw/PInterestRates.xls`](../../data/raw/PInterestRates.xls).

Вибірка: Аркуші 2006–2026: колонка з заголовком «Облікова ставка».

Призначення й особливості: 5187 рядків звірено з архівом рішень, розбіжностей немає. Старі аркуші 2001–2005 не містять потрібного ряду облікової ставки; не використовувати ставку овернайт як заміну.

Результат перевірки: Прийнято з обмеженням історії.

## 5. Exchange_r.xls

Установа: **НБУ**. [Першоджерело](https://bank.gov.ua/files/Exchange_r.xls).

Файл у репозиторії: [`data/raw/Exchange_r.xls`](../../data/raw/Exchange_r.xls).

Вибірка: 1996-2009, 2010-2018, 2019, 2020-2024, 2025-2026: рядки EUR та USD.

Призначення й особливості: Місячні середні офіційні курси. До 2020 року значення за 100 одиниць поділено на 100. Річні середні пропущено. Текстове число USD за 2002-10 очищено від пробілу.

Результат перевірки: Прийнято після перетворення.

## 6. prc_hicp_minr__custom_23410998_spreadsheet.xlsx

Установа: **Eurostat**. [Першоджерело](https://ec.europa.eu/eurostat/databrowser/view/prc_hicp_minr/default/table?lang=en).

Файл у репозиторії: [`data/raw/prc_hicp_minr__custom_23410998_spreadsheet.xlsx`](../../data/raw/prc_hicp_minr__custom_23410998_spreadsheet.xlsx).

Вибірка: Sheet 1: рядок 13, дати рядок 9.

Призначення й особливості: HICP Total, індекс 2025=100, єврозона змінного складу. 321 місяць, 2000-01–2026-09. Не сезонно скоригований. Решту країн відсіяно.

Результат перевірки: Прийнято.

## 7. namq_10_gdp__custom_23412352_spreadsheet.xlsx

Установа: **Eurostat**. [Першоджерело](https://ec.europa.eu/eurostat/databrowser/view/namq_10_gdp/default/table?lang=en).

Файл у репозиторії: [`data/raw/namq_10_gdp__custom_23412352_spreadsheet.xlsx`](../../data/raw/namq_10_gdp__custom_23412352_spreadsheet.xlsx).

Вибірка: Sheet 1: рядок 13, дати рядок 10; Footnotes 1.

Призначення й особливості: ВВП за ринковими цінами, ланцюгові обсяги, 2020=100, SCA. Єврозона змінного складу узгоджена з HICP. 106 кварталів.

Результат перевірки: Прийнято.

## 8. data.csv

Установа: **ECB**. [Першоджерело](https://data-api.ecb.europa.eu/service/data/FM/D.U2.EUR.4F.KR.DFR.LEV?startPeriod=1999-01-01&endPeriod=2026-09-30&format=csvdata).

Файл у репозиторії: [`data/raw/data.csv`](../../data/raw/data.csv).

Вибірка: KEY, TIME_PERIOD, OBS_VALUE, OBS_STATUS.

Призначення й особливості: Єдиний ряд FM.D.U2.EUR.4F.KR.DFR.LEV, депозитна ставка ЄЦБ, % річних. 10135 щоденних значень. Збережено від’ємні ставки.

Результат перевірки: Прийнято.

## 9. Surveys_price.xlsx

Установа: **НБУ**. [Першоджерело](https://bank.gov.ua/files/macro/Surveys_price.xlsx).

Файл у репозиторії: [`data/raw/Surveys_price.xlsx`](../../data/raw/Surveys_price.xlsx).

Вибірка: UKR: колонка A дата; B–E групи респондентів.

Призначення й особливості: Очікувана інфляція на наступні 12 місяців: банки, підприємства, домогосподарства, аналітики. У двох мовних копіях 504 #N/A, що не означає пошкодження. Новіший Q3 для підприємств взято окремо з BOS.

Результат перевірки: Прийнято; пропуски збережено.

## 10. Surveys_rate.xlsx

Установа: **НБУ**. [Першоджерело](https://bank.gov.ua/files/macro/Surveys_rate.xlsx).

Файл у репозиторії: [`data/raw/Surveys_rate.xlsx`](../../data/raw/Surveys_rate.xlsx).

Вибірка: UKR: колонка A дата; B–E групи респондентів.

Призначення й особливості: Курсові очікування на наступні 12 місяців, грн/долар. 504 #N/A у двох мовних копіях. Це очікуваний курс, а не фактичний курс і не темп девальвації.

Результат перевірки: Прийнято; пропуски збережено.

## 11. monthly_long_nf_publ.xlsx

Установа: **НБУ**. [Першоджерело](https://bank.gov.ua/admin_uploads/article/monthly_long_nf_publ.xlsx).

Файл у репозиторії: [`data/raw/monthly_long_nf_publ.xlsx`](../../data/raw/monthly_long_nf_publ.xlsx).

Вибірка: ІОДА_УКР: роки рядок 2, місяці рядок 5; основні індекси рядки 14,18,56,94,124.

Призначення й особливості: ІОДА загалом і за 4 секторами, нейтральна межа 50. Окремо 11588 числових клітинок деталей. Пропуски березня–травня 2022 не інтерпольовано. Позначені * індекси закупівельних цін інвертовані.

Результат перевірки: Прийнято; розрив 2022 збережено.

## 12. BOS_2026-Q3.pdf

Установа: **НБУ**. [Першоджерело](https://bank.gov.ua/admin_uploads/article/BOS_2026-Q3.pdf).

Файл у репозиторії: [`data/raw/BOS_2026-Q3.pdf`](../../data/raw/BOS_2026-Q3.pdf).

Вибірка: 73 сторінки; підсумки с.3, паспорт вибірки с.4.

Призначення й особливості: 7 ключових показників Q3 та їх Q2-порівняння. Опитано 664 підприємства 30.07–28.08.2026. ІДО 100.8 з нейтральною межею 100. Не змішувати з місячним ІОДА.

Результат перевірки: Прийнято.

## 13. Macroeconomic_Forecast_table_2026_07.xlsx

Установа: **НБУ**. [Першоджерело](https://bank.gov.ua/admin_uploads/article/Macroeconomic_Forecast_table_2026_07.xlsx).

Файл у репозиторії: [`data/raw/Macroeconomic_Forecast_table_2026_07.xlsx`](../../data/raw/Macroeconomic_Forecast_table_2026_07.xlsx).

Вибірка: Базовий сценарій; Припущення базового сценарію; Перегляд прогнозу.

Призначення й особливості: Поточний і попередній прогнози, історичні оцінки, зовнішні припущення. Зведено окремо; не використано як факти для оцінки власної моделі.

Результат перевірки: Прийнято як прогнозний орієнтир.

## 14. Tables_Charts_2026-Q3.xlsx

Установа: **НБУ**. [Першоджерело](https://bank.gov.ua/admin_uploads/article/Tables_Charts_2026-Q3.xlsx).

Файл у репозиторії: [`data/raw/Tables_Charts_2026-Q3.xlsx`](../../data/raw/Tables_Charts_2026-Q3.xlsx).

Вибірка: Каталог усіх аркушів; вибірка 2.18,3.18,4.1,4.5,4.8,4.9.

Призначення й особливості: Траєкторія ставки, оцінка розриву ВВП, сировинні припущення та зовнішні показники. 103 #N/A у 1.8 та 3.12. На 4.10 напис «немає дозволу»: числові OIS-дані недоступні, їх не вигадано.

Результат перевірки: Прийнято з обмеженнями окремих графіків.

## 15. IR_2026-Q3.pdf

Установа: **НБУ**. [Першоджерело](https://bank.gov.ua/admin_uploads/article/IR_2026-Q3.pdf).

Файл у репозиторії: [`data/raw/IR_2026-Q3.pdf`](../../data/raw/IR_2026-Q3.pdf).

Вибірка: 56 сторінок; таблиці с.53–55, текст і примітки.

Призначення й особливості: Перевірка визначень, прогнозного статусу, ризиків і логіки сценарію. Підсумкові числа звірено з XLSX. Дані PDF не додано повторно як незалежні спостереження.

Результат перевірки: Прийнято як методичне джерело.

## 16. nbu_policy_archive_rows.json

Установа: **НБУ**. [Першоджерело](https://bank.gov.ua/ua/monetary/archive-rish).

Файл у репозиторії: [`data/raw/nbu_policy_archive_rows.json`](../../data/raw/nbu_policy_archive_rows.json).

Вибірка: Таблиця рішень, дати набрання чинності.

Призначення й особливості: Доповнює ранню історію та дає календарний ряд ставки. Перенесення між рішеннями означає незмінну чинну ставку, не статистичну інтерполяцію.

Результат перевірки: Додатково отримано 20.10.2026.

## 17. WorldBank_commodities.xlsx

Установа: **World Bank**. [Першоджерело](https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/CMO-Historical-Data-Monthly.xlsx).

Файл у репозиторії: [`data/raw/WorldBank_commodities.xlsx`](../../data/raw/WorldBank_commodities.xlsx).

Вибірка: Brent; Europe natural gas USD/MMBtu; US SRW/HRW wheat USD/tonne.

Призначення й особливості: Additional observed series, retrieved 2026-10-21

Результат перевірки: Verified source, native definition retained.

## 18. FAO_food.csv

Установа: **FAO**. [Першоджерело](https://www.fao.org/worldfoodsituation/foodpricesindex/en/).

Файл у репозиторії: [`data/raw/FAO_food.csv`](../../data/raw/FAO_food.csv).

Вибірка: FAO nominal food price indices, base 2014–2016=100.

Призначення й особливості: Additional observed series, retrieved 2026-10-21

Результат перевірки: Verified source, native definition retained.

## 19. BOP_m.xlsx

Установа: **НБУ**. [Першоджерело](https://bank.gov.ua/files/ES/BOP_m.xlsx).

Файл у репозиторії: [`data/raw/BOP_m.xlsx`](../../data/raw/BOP_m.xlsx).

Вибірка: BPM6 monthly flows, million USD; provisional flags preserved.

Призначення й особливості: Additional observed series, retrieved 2026-10-21

Результат перевірки: Verified source, native definition retained.

## 20. Wages_m.xlsx

Установа: **НБУ**. [Першоджерело](https://bank.gov.ua/files/macro/Wages_m.xlsx).

Файл у репозиторії: [`data/raw/Wages_m.xlsx`](../../data/raw/Wages_m.xlsx).

Вибірка: Nominal average salaried-worker wage, UAH, KVED2010.

Призначення й особливості: Additional observed series, retrieved 2026-10-21

Результат перевірки: Verified source, native definition retained.

## 21. Unemp_q.xlsx

Установа: **НБУ**. [Першоджерело](https://bank.gov.ua/files/macro/Unemp_q.xlsx).

Файл у репозиторії: [`data/raw/Unemp_q.xlsx`](../../data/raw/Unemp_q.xlsx).

Вибірка: Sheet3: discrete quarterly ILO age15+; not cumulative YTD.

Призначення й особливості: Additional observed series, retrieved 2026-10-21

Результат перевірки: Verified source, native definition retained.

## 22. Financial_markets.xlsx

Установа: **НБУ**. [Першоджерело](https://bank.gov.ua/files/4-Financial_markets.xlsx).

Файл у репозиторії: [`data/raw/Financial_markets.xlsx`](../../data/raw/Financial_markets.xlsx).

Вибірка: New NFC loans in UAH, with and without overdraft.

Призначення й особливості: Additional observed series, retrieved 2026-10-21

Результат перевірки: Verified source, native definition retained.

## 23. UONIA.xls

Установа: **НБУ**. [Першоджерело](https://bank.gov.ua/files/UONIA.xls).

Файл у репозиторії: [`data/raw/UONIA.xls`](../../data/raw/UONIA.xls).

Вибірка: Published UONIA business days, no pre-UONIA splice.

Призначення й особливості: Additional observed series, retrieved 2026-10-21

Результат перевірки: Verified source, native definition retained.

## Де результати обробки

- `data/model/quarterly_training.csv`: 66 повних кварталів 2010Q1–2026Q2, сім основних спостережень.

- `data/model/quarterly_levels.csv`: рівні, ставки та останній неповний квартал.

- `data/model/auxiliary_monthly.csv`, `auxiliary_quarterly.csv`: сировина, FAO, торгівля, кредитні ставки, UONIA, зарплата та доступне безробіття.

- `data/model/surveys_*.csv`, `business_*.csv`: очікування й ділова активність.

- `data/model/nbu_benchmarks.csv`, `nbu_external_assumptions.csv`: прогнозні орієнтири, окремо від фактів.

- `data/documentation/dictionary.csv`, `coverage.csv`, `auxiliary_provenance.csv`: одиниці, покриття, точні клітинки й статус спостережень.


Info Sapiens як окремий індекс споживчих настроїв до фінального набору не включений. Після 2021 року фактичного безробіття МОП у вибраному файлі немає; зарплата також має розриви.
