"""Reproducible extraction. Originals are read-only; no estimation of QPM parameters."""
from pathlib import Path
import json,csv,re,shutil,hashlib,sys,math
import pandas as pd
import numpy as np
from openpyxl.utils import get_column_letter as col

ROOT=Path(__file__).resolve().parents[1]
PACKAGED=True
AUD=ROOT/'reproduce/extracted'
OUT=ROOT
for p in ['raw','data','documentation','reproduce']: (OUT/p).mkdir(parents=True,exist_ok=True)
D=json.loads((AUD/'cells.json').read_text())
manifest=json.loads((AUD/'manifest.json').read_text())
cutoff=pd.Timestamp('2026-09-30')
checks=[]; obs=[]; desc={}
def check(name,ok,detail):
 checks.append({'check':name,'status':'PASS' if ok else 'REVIEW','detail':str(detail)})
def numeric(v):
 return isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v)
def add(series,period,value,unit,source,locator,status='observed',flag='',note=''):
 if not numeric(value):return
 obs.append(dict(series=series,period=str(period),value=float(value),unit=unit,source=source,
                 locator=locator,status=status,source_flag=flag or '',note=note,vintage_date='2026-10-20'))
def row_series(fn,sheet,header,row,series,unit,startcol=0):
 rows=D[fn][sheet]
 for j,dt in enumerate(rows[header-1]):
  if j<startcol or not isinstance(dt,str) or not re.match(r'^\d{4}-\d{2}-\d{2}',dt):continue
  add(series,pd.Period(dt[:10],freq='M'),rows[row-1][j],unit,fn,f"'{sheet}'!{col(j+1)}{row}")
row_series('CPI_m.xlsx','2',2,3,'CPI_UA_MOM','pct_mom')
row_series('CPI_m.xlsx','1',2,3,'CPI_UA_YOY_PUBLISHED','pct_yoy')
for row,key,unit in [(4,'CORE_CPI_UA_MOM','pct_mom'),(6,'CORE_CPI_UA_YOY_PUBLISHED','pct_yoy')]:
 row_series('CCPI_m.xlsx','1',2,row,key,unit)

# Official monthly average FX; skip annual-average columns and pre-euro ECU.
fxspec=[('1996-2009',1996,2009,100),('2010-2018',2010,2018,100),('2019',2019,2019,100),('2020-2024',2020,2024,1),('2025-2026',2025,2026,1)]
for sheet,first,last,divisor in fxspec:
 rows=D['Exchange_r.xls'][sheet]
 for key,pattern in [('ER_UAH_EUR_AVG',r'^(100|1) [Єє]вро'),('ER_UAH_USD_AVG',r'^(100|1) долар')]:
  matches=[(i,row) for i,row in enumerate(rows) if re.search(pattern,str(row[0]),re.I)]
  assert len(matches)==1,(sheet,key,matches)
  i,row=matches[0]
  for year in range(first,last+1):
   if year<1999:continue
   for month in range(1,13):
    j=1+(year-first)*13+month-1
    if j>=len(row):continue
    per=pd.Period(f'{year}-{month:02}',freq='M')
    if per.end_time.normalize()>cutoff:continue
    raw=row[j]
    textclean=False
    if isinstance(raw,str) and re.fullmatch(r'\s*\d+(?:[.,]\d+)?\s*',raw):
     raw=float(raw.strip().replace(',','.'));textclean=True
    add(key,per,raw/divisor if numeric(raw) else None,'UAH_per_currency_unit','Exchange_r.xls',f"'{sheet}'!{col(j+1)}{i+1}",note=f'Original divisor {divisor}; monthly average'+('; numeric text with nonbreaking space normalized' if textclean else ''))

# Archive is authoritative for historical rate decisions; XLS independently cross-checks.
archive=json.loads((AUD/'nbu_policy_archive_rows.json').read_text())
events=[];year=None
for i,row in enumerate(archive):
 if len(row)==1 and re.search(r'\d{4}\s*рік',row[0]):year=int(re.search(r'\d{4}',row[0])[0])
 elif year and len(row)>1 and re.match(r'з \d{2}\.\d{2}',row[0]):
  day,month=map(int,re.search(r'(\d{2})\.(\d{2})',row[0]).groups())
  events.append([pd.Timestamp(year,month,day),float(row[1].replace(',','.').replace('\xa0','')),i+1])
events=pd.DataFrame(events,columns=['date','rate','archive_row']).sort_values('date')
assert not events.date.duplicated().any()
events.to_csv(OUT/'data/policy_decisions.csv',index=False,date_format='%Y-%m-%d')
daily=events.set_index('date').rate.reindex(pd.date_range(events.date.min(),cutoff,freq='D')).ffill()
daily=daily.loc['1999-01-01':]
daily.rename_axis('date').rename('IR_POLICY').to_csv(OUT/'data/policy_daily.csv',date_format='%Y-%m-%d')
dailygroup=daily.groupby(daily.index.to_period('M'))
for method,values in [('EOM',dailygroup.last()),('CALENDAR_AVG',dailygroup.mean())]:
 for p,v in values.items():
  add('IR_POLICY_'+method,p,v,'pct_per_annum','nbu_policy_archive_rows.json','effective-date decisions',note='Official rate carried through calendar days until next effective decision')
xlscomparison=[]
for sheet,rows in D['PInterestRates.xls'].items():
 if not any('Облікова ставка' in r for r in rows[:12]):continue
 for i,row in enumerate(rows):
  if len(row)>1 and numeric(row[0]) and row[0]>30000 and numeric(row[1]):
   dt=pd.Timestamp('1899-12-30')+pd.Timedelta(days=row[0])
   if dt<=cutoff and dt in daily.index:
    xlscomparison.append([dt.strftime('%Y-%m-%d'),sheet,i+1,row[1],daily.loc[dt],row[1]-daily.loc[dt]])
cross=pd.DataFrame(xlscomparison,columns=['date','sheet','row','xls_rate','archive_rate','difference'])
cross.to_csv(OUT/'documentation/policy_crosscheck.csv',index=False)
bad=cross.loc[abs(cross.difference)>1e-9]
check('Policy XLS versus official decision dates',bad.empty,f'{len(cross)} comparable rows; {len(bad)} differences; see policy_crosscheck.csv')
if not bad.empty: print('POLICY DIFFERENCES',bad.to_string(index=False))

# Eurostat selected dimensions; source flags retained per cell.
for fn,time_row,val_row,key,unit,freq in [
 ('prc_hicp_minr__custom_23410998_spreadsheet.xlsx',9,13,'HICP_EA_NSA','index_2025_100','M'),
 ('namq_10_gdp__custom_23412352_spreadsheet.xlsx',10,13,'GDP_EA_SCA','chain_volume_index_2020_100','Q')]:
 rows=D[fn]['Sheet 1']
 assert str(rows[val_row-1][0]).startswith('Euro area (EA11-1999')
 for j,dt in enumerate(rows[time_row-1]):
  if not isinstance(dt,str) or not re.match(r'^\d{4}-(?:Q[1-4]|\d\d)$',dt):continue
  per=pd.Period(dt.replace('-Q','Q'),freq=freq)
  flag=rows[val_row-1][j+1] if j+1<len(rows[val_row-1]) else ''
  add(key,per,rows[val_row-1][j],unit,fn,f"'Sheet 1'!{col(j+1)}{val_row}",flag=flag,
      note='Euro area changing composition; GDP SCA, HICP NSA')
rows=D['_dodatok_sezonka_vvp_2010-2026.xlsx']['2010-2026']
for j,p in enumerate(rows[3]):
 if isinstance(p,str) and re.match(r'^\d{4}Q[1-4]$',p):
  add('GDP_UA_SA',p,rows[7][j],'million_UAH_2021_prices','_dodatok_sezonka_vvp_2010-2026.xlsx',f"'2010-2026'!{col(j+1)}8")
# ECB daily data: exactly one verified deposit facility series.
ecb=pd.read_csv(OUT/'raw/data.csv')
assert set(ecb.KEY)=={'FM.D.U2.EUR.4F.KR.DFR.LEV'}
ecb['date']=pd.to_datetime(ecb.TIME_PERIOD)
assert not ecb.date.duplicated().any()
assert ecb.date.max()==cutoff
ecbs=ecb.set_index('date').OBS_VALUE.sort_index()
check('ECB daily calendar coverage',len(ecbs)==len(pd.date_range(ecbs.index.min(),ecbs.index.max())),f'{len(ecbs)} daily observations')
ecbs.rename_axis('date').rename('IR_ECB_DFR').to_csv(OUT/'data/ecb_daily.csv',date_format='%Y-%m-%d')
for method,s in [('EOM',ecbs.groupby(ecbs.index.to_period('M')).last()),('CALENDAR_AVG',ecbs.groupby(ecbs.index.to_period('M')).mean())]:
 for p,v in s.items():add('IR_ECB_DFR_'+method,p,v,'pct_per_annum','data.csv','TIME_PERIOD / OBS_VALUE',note='FM.D.U2.EUR.4F.KR.DFR.LEV; '+method)

# Surveys preserve missing values and group definitions; no interpolation.
survey=[]
for fn,prefix,unit in [('Surveys_price.xlsx','EXP_INFL_12M','pct_expected_inflation'),('Surveys_rate.xlsx','EXP_FX_USD_12M','UAH_per_USD_expected')]:
 rows=D[fn]['UKR']
 for i,row in enumerate(rows):
  if not isinstance(row[0],str) or not re.match(r'^\d{4}-\d{2}-\d{2}',row[0]):continue
  per=str(pd.Period(row[0][:10],freq='M'))
  for j,group in enumerate(['banks','business','households','analysts'],1):
   value=row[j] if numeric(row[j]) else None
   survey.append(dict(period=per,series=prefix+'_'+group,value=value,group=group,horizon_months=12,
      unit=unit,source=fn,locator=f"'UKR'!{col(j+1)}{i+1}",
      status='survey' if value is not None else 'missing',reference_frequency='M',
      publication_date=None,note='Source observation month; not forecast realization; missing not interpolated'))
survey=pd.DataFrame(survey)
survey.to_csv(OUT/'data/surveys_monthly.csv',index=False)

# Monthly business activity: principal indices and complete quantitative survey rows.
rows=D['monthly_long_nf_publ.xlsx']['ІОДА_УКР']
months={'січень':1,'лютий':2,'березень':3,'квітень':4,'травень':5,'червень':6,'липень':7,'серпень':8,'вересень':9,'жовтень':10,'листопад':11,'грудень':12}
periodcols={};year=None
for j in range(1,len(rows[4])):
 if rows[1][j] is not None:
  hit=re.search(r'20\d{2}',str(rows[1][j]))
  if hit:year=int(hit[0])
 if rows[4][j] in months and year:periodcols[j]=f'{year}-{months[rows[4][j]]:02}'
business=[]
sector='all'
for i,row in enumerate(rows):
 label=str(row[0] or '').strip()
 if label in ['Промисловість','Будівництво','Торгівля','Послуги']:sector=label
 if i<5 or i>=156 or not label:continue
 for j,p in periodcols.items():
  if numeric(row[j]):
   business.append(dict(period=p,source_row=i+1,sector=sector,label=label,value=row[j],
    source='monthly_long_nf_publ.xlsx',locator=f"'ІОДА_УКР'!{col(j+1)}{i+1}",
    note='Input-price DI marked * is inverted: higher input prices reduce DI; see row 161' if '*' in label else ''))
business=pd.DataFrame(business)
business.to_csv(OUT/'data/business_survey_detail.csv',index=False)
bm=pd.DataFrame(index=pd.period_range('2019-07','2026-09',freq='M'))
for r,name in [(14,'BAEI_ALL'),(18,'BAEI_INDUSTRY'),(56,'BAEI_CONSTRUCTION'),(94,'BAEI_TRADE'),(124,'BAEI_SERVICES')]:
 bm[name]=pd.Series({pd.Period(p,freq='M'):rows[r-1][j] if numeric(rows[r-1][j]) else np.nan for j,p in periodcols.items()})
bm.index.name='period';bm.to_csv(OUT/'data/business_monthly.csv')

# Latest quarterly enterprise survey, visually verified PDF pp.3-4.
bos=[]
for name,unit,v2,v3 in [
 ('EXP_INFL_12M_business','pct_expected_inflation',11.6,11.5),
 ('EXP_FX_USD_12M_business','UAH_per_USD_expected',46.12,46.90),
 ('EXP_FX_EUR_12M_business','UAH_per_EUR_expected',54.42,54.85),
 ('BOI_business','index_neutral_100',107.0,100.8),
 ('PRODUCTION_EXPECT_BALANCE','pct_positive_minus_negative',7.6,-11.9),
 ('BORROWING_NEED_BALANCE','pct_positive_minus_negative',35.3,32.6),
 ('PLAN_BANK_CREDIT_SHARE','pct_respondents',36.0,35.6)]:
 for q,v in [('2026Q2',v2),('2026Q3',v3)]:
  bos.append(dict(period=q,series=name,value=v,unit=unit,status='survey',source='BOS_2026-Q3.pdf',locator='p.3; sample p.4',
      publication_date='2026-10-15',fieldwork_start='2026-07-30' if q=='2026Q3' else None,
      fieldwork_end='2026-08-28' if q=='2026Q3' else None,sample_n=664 if q=='2026Q3' else None,
      note='Q2 retrospective comparison in Q3 publication' if q=='2026Q2' else 'Quarterly survey label, not September fieldwork'))
bos=pd.DataFrame(bos);bos.to_csv(OUT/'data/business_quarterly_latest.csv',index=False)
sq=[]
for (series,q),g in survey.dropna(subset=['value']).assign(quarter=lambda x:pd.PeriodIndex(x.period,freq='M').asfreq('Q').astype(str)).groupby(['series','quarter']):
 sq.append(dict(period=q,series=series,value=g.sort_values('period').iloc[-1].value,source=g.iloc[-1].source,
    n_observations=len(g),method='last observed survey month in quarter',status='survey'))
sq=pd.DataFrame(sq).set_index(['period','series'])
for _,r in bos.iterrows():
 if r.series in survey.series.unique():sq.loc[(r.period,r.series),:]=[r.value,r.source,1,'PDF quarterly survey replaces stale spreadsheet value',r.status]
sq.reset_index().sort_values(['period','series']).to_csv(OUT/'data/surveys_quarterly.csv',index=False)

# Complete supplied baseline table as a distinct vintage of NBU benchmarks.
benchmark=[]
fn='Macroeconomic_Forecast_table_2026_07.xlsx';r=D[fn]['Базовий сценарій']
periodspec={2:('2023','annual','historical_or_estimate'),3:('2024','annual','historical_or_estimate'),4:('2025','annual','historical_or_estimate')}
for year,first,annual,previous in [(2026,5,9,10),(2027,11,15,16),(2028,17,21,22)]:
 for k in range(4):periodspec[first+k]=(f'{year}Q{k+1}','quarterly','mixed_actual_estimate' if year==2026 and k<2 else 'forecast')
 periodspec[annual]=(str(year),'annual','forecast')
 periodspec[previous]=(str(year),'annual','previous_forecast')
for i,row in enumerate(r):
 if i<6 or not row[1] or not any(numeric(v) for v in row[2:]):continue
 for j,(p,freq,st) in periodspec.items():
  if numeric(row[j]):
   benchmark.append(dict(period=p,frequency=freq,indicator=row[1],value=row[j],status=st,
     vintage='2026-04' if st=='previous_forecast' else '2026-07',source=fn,
     locator=f"'Базовий сценарій'!{col(j+1)}{i+1}",note='Units in indicator; default pct_yoy except monetary accounts pct_since_Dec; never merged as official actual data'))
assumptions=[]
r=D[fn]['Припущення базового сценарію'];lastlabel=''
for i,row in enumerate(r):
 if row[1]:lastlabel=str(row[1])
 if i<4 or not row[2]:continue
 for j,year in enumerate(range(2023,2029),3):
  if numeric(row[j]):assumptions.append(dict(period=str(year),indicator=lastlabel,unit=row[2],value=row[j],
   status='historical_or_estimate' if year<2026 else 'NBU_assumption',vintage='2026-07',source=fn,
   locator=f"'Припущення базового сценарію'!{col(j+1)}{i+1}"))
# Selected charts: key policy path, annual GDP gap, EA growth, commodities.
charts=D['Tables_Charts_2026-Q3.xlsx']
for i,row in enumerate(charts['3.18'][3:47],3):
 if not numeric(row[1]):continue
 q=pd.Period('2018Q1',freq='Q')+(i-3)
 if q>pd.Period('2028Q4'):continue
 benchmark.append(dict(period=str(q),frequency='quarterly',indicator='Облікова ставка, середня за квартал, %',value=row[1],
   status='historical_or_estimate' if q<=pd.Period('2026Q2') else 'forecast',vintage='2026-07',source='Tables_Charts_2026-Q3.xlsx',
   locator=f"'3.18'!B{i+1}",note='Quarter average, not quarter-end rate'))
for i,row in enumerate(charts['2.18']):
 if numeric(row[0]) and 20<=row[0]<=28 and numeric(row[1]):
  benchmark.append(dict(period=str(2000+int(row[0])),frequency='annual',indicator='Розрив ВВП, % потенційного ВВП',value=row[1],
   status='NBU_model_estimate' if row[0]<26 else 'forecast',vintage='2026-07',source='Tables_Charts_2026-Q3.xlsx',
   locator=f"'2.18'!B{i+1}",note='Model estimate, not measured GDP and not own QPM calibration target'))
for i,row in enumerate(charts['4.8'][3:8],3):
 assumptions.append(dict(period=str(int(row[3])),indicator='Euro area real GDP growth',unit='pct_yoy_annual',value=row[4],
  status='historical_or_estimate' if row[3]<2026 else 'NBU_assumption',vintage='2026-07',source='Tables_Charts_2026-Q3.xlsx',locator=f"'4.8'!E{i+1}"))
for sheet,columns in [('4.1',[(1,'Brent','USD_per_barrel'),(3,'Dutch_TTF','USD_per_1000m3')]),('4.5',[(1,'Corn','USD_per_tonne'),(3,'Wheat','USD_per_tonne'),(5,'Sunflower_oil','USD_per_tonne')]),('4.9',[(2,'Euro area inflation at quarter end','pct_yoy_eop')])]:
 for i,row in enumerate(charts[sheet]):
  if not isinstance(row[0],str) or not re.match(r'^(I|II|III|IV)\.\d\d$',row[0]):continue
  roman,yr=row[0].split('.');q=f'20{yr}Q'+str(['I','II','III','IV'].index(roman)+1)
  for j,name,unit in columns:
   if numeric(row[j]):assumptions.append(dict(period=q,indicator=name,unit=unit,value=row[j],status='historical_or_estimate' if q<='2026Q2' else 'NBU_assumption',vintage='2026-07',source='Tables_Charts_2026-Q3.xlsx',locator=f"'{sheet}'!{col(j+1)}{i+1}"))
benchmark=pd.DataFrame(benchmark);assumptions=pd.DataFrame(assumptions)
benchmark.to_csv(OUT/'data/nbu_benchmarks.csv',index=False)
assumptions.to_csv(OUT/'data/nbu_external_assumptions.csv',index=False)
catalog=[]
for name,rows in charts.items():
 labels=[str(v) for v in rows[0] if isinstance(v,str) and len(v)>20] if rows else []
 catalog.append(dict(sheet=name,rows=len(rows),title=labels[-1] if labels else '',
  numeric_cells=sum(numeric(v) for row in rows for v in row),
  na_cells=sum(v=='#N/A' for row in rows for v in row),
  restricted=any('немає дозволу' in str(v) for row in rows for v in row)))
pd.DataFrame(catalog).to_csv(OUT/'documentation/report_charts_catalog.csv',index=False)

# Standardized long table is the authoritative source for source-derived observations.
observations=pd.DataFrame(obs).sort_values(['series','period'])
check('Unique series-period keys',not observations.duplicated(['series','period']).any(),len(observations))
observations.to_csv(OUT/'data/observations_long.csv',index=False)
monthlykeys=[x for x in observations.series.unique() if x not in ['GDP_UA_SA','GDP_EA_SCA']]
monthly=observations[observations.series.isin(monthlykeys)].pivot(index='period',columns='series',values='value')
monthly.index=pd.PeriodIndex(monthly.index,freq='M')
monthly=monthly.reindex(pd.period_range('1999-01','2026-09',freq='M'))
monthly.index.name='period'
cpi=monthly.CPI_UA_MOM.dropna()
assert cpi.index.equals(pd.period_range(cpi.index.min(),cpi.index.max(),freq='M'))
level=(1+cpi/100).cumprod()
level=level/level.loc['2021'].mean()*100
monthly['CPI_UA_INDEX_NSA_2021_100']=level
monthly['CPI_UA_YOY_COMPOUNDED']=100*((1+cpi/100).rolling(12).apply(np.prod,raw=True)-1)
diff=(monthly.CPI_UA_YOY_COMPOUNDED-monthly.CPI_UA_YOY_PUBLISHED).dropna()
check('CPI published annual versus chained rounded monthly',abs(diff).max()<0.5,f'Max abs deviation {abs(diff).max():.6f} pp; rounding tolerance 0.5 pp')
monthly.to_csv(OUT/'data/core_monthly.csv')
quarter=pd.DataFrame(index=pd.period_range('2000Q1','2026Q3',freq='Q'))
quarter.index.name='period'
for series in ['GDP_UA_SA','GDP_EA_SCA']:
 t=observations[observations.series==series].set_index('period').value
 t.index=pd.PeriodIndex(t.index,freq='Q')
 quarter[series]=t
for source,target in [('CPI_UA_INDEX_NSA_2021_100','CPI_UA_INDEX_NSA_AVG'),('ER_UAH_EUR_AVG','ER_UAH_EUR_AVG'),('ER_UAH_USD_AVG','ER_UAH_USD_AVG'),('HICP_EA_NSA','HICP_EA_INDEX_NSA_AVG'),('IR_POLICY_EOM','IR_POLICY_MONTHLY_EOM_AVG')]:
 g=monthly[source].groupby(monthly.index.asfreq('Q'))
 quarter[target]=g.mean().where(g.count()==3)
for source,target in [('IR_POLICY_EOM','IR_POLICY_EOQ'),('IR_ECB_DFR_EOM','IR_ECB_DFR_EOQ')]:
 quarter[target]=monthly[source].loc[monthly.index.month.isin([3,6,9,12])].set_axis(monthly.loc[monthly.index.month.isin([3,6,9,12])].index.asfreq('Q'))
quarter['IR_POLICY_CALENDAR_AVG']=daily.groupby(daily.index.to_period('Q')).mean()
quarter['IR_ECB_DFR_CALENDAR_AVG']=ecbs.groupby(ecbs.index.to_period('Q')).mean()
for series in ['GDP_UA_SA','GDP_EA_SCA','CPI_UA_INDEX_NSA_AVG','HICP_EA_INDEX_NSA_AVG']:
 quarter['D4LOG_'+series]=100*np.log(quarter[series]/quarter[series].shift(4))
 # qoq annualised inflation deliberately omitted until seasonal adjustment.
 if series.startswith('GDP'):
  quarter['DLOG4_'+series]=400*np.log(quarter[series]/quarter[series].shift(1))
quarter.to_csv(OUT/'data/core_quarterly.csv')
corecols=['GDP_UA_SA','GDP_EA_SCA','CPI_UA_INDEX_NSA_AVG','ER_UAH_EUR_AVG','HICP_EA_INDEX_NSA_AVG','IR_POLICY_CALENDAR_AVG','IR_ECB_DFR_EOQ']
balanced=quarter.loc['2010Q1':'2026Q2',corecols]
check('Balanced observed QPM core 2010Q1–2026Q2',not balanced.isna().any().any(),f'{len(balanced)} quarters x {len(corecols)} core variables')
check('GDP latest quarter remains unknown',pd.isna(quarter.loc['2026Q3','GDP_UA_SA']) and pd.isna(quarter.loc['2026Q3','GDP_EA_SCA']),'2026Q3 missing in both GDP series; no interpolation')
for name in ['GDP_UA_SA','GDP_EA_SCA','CPI_UA_INDEX_NSA_AVG','ER_UAH_EUR_AVG','HICP_EA_INDEX_NSA_AVG']:
 check('Positive log input '+name,quarter[name].dropna().gt(0).all(),f'{quarter[name].notna().sum()} observations')
check('UA GDP continuity',quarter.GDP_UA_SA.loc['2010Q1':'2026Q2'].notna().all(),'66 uninterrupted quarters')
check('FX unit boundary 2019–2020',abs(monthly.loc['2020-01','ER_UAH_EUR_AVG']/monthly.loc['2019-12','ER_UAH_EUR_AVG']-1)<0.2,
 f"2019-12={monthly.loc['2019-12','ER_UAH_EUR_AVG']}; 2020-01={monthly.loc['2020-01','ER_UAH_EUR_AVG']}")
check('UA CPI average 2021 equals 100',abs(monthly.loc['2021','CPI_UA_INDEX_NSA_2021_100'].mean()-100)<1e-10,'Rebasing changes levels only')
check('ECB negative rates retained',ecbs.min()<0,f'Min {ecbs.min()}%')
check('Surveys missing values retained',survey.value.isna().any(),'No filling of survey gaps')
for f in manifest:check('Original unchanged: '+f['file'],hashlib.sha256((OUT/'raw'/f['file']).read_bytes()).hexdigest()==f['sha256'],f"{f['bytes']} bytes")
pd.DataFrame(checks).to_csv(OUT/'documentation/validation.csv',index=False)
(OUT/'documentation/file_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))

coverage=[]
for series,g in observations.groupby('series'):
 coverage.append(dict(series=series,first=g.period.min(),last=g.period.max(),n=len(g),unit=g.unit.iloc[0],sources='; '.join(g.source.unique())))
pd.DataFrame(coverage).to_csv(OUT/'documentation/coverage.csv',index=False)
bundle={'monthly':monthly.reset_index().assign(period=lambda x:x.period.astype(str)).replace({np.nan:None}).to_dict('records'),
 'quarterly':quarter.reset_index().assign(period=lambda x:x.period.astype(str)).replace({np.nan:None}).to_dict('records'),
 'business':bm.reset_index().assign(period=lambda x:x.period.astype(str)).replace({np.nan:None}).to_dict('records'),
 'surveys':survey.replace({np.nan:None}).to_dict('records'),
 'bos':bos.replace({np.nan:None}).to_dict('records'),
 'benchmarks':benchmark.replace({np.nan:None}).to_dict('records'),
 'assumptions':assumptions.replace({np.nan:None}).to_dict('records'),
 'coverage':coverage,'checks':checks}
(AUD/'bundle.json').write_text(json.dumps(bundle,ensure_ascii=False,allow_nan=False))
if not PACKAGED:
 shutil.copy2(__file__,OUT/'reproduce/prepare_dataset.py')
print('OUTPUT',OUT)
print('MONTHLY',monthly.shape,'QUARTERLY',quarter.shape,'OBS',len(observations),'SURVEYS',len(survey),'BUSINESS_DETAIL',len(business))
print('CHECKS',len(checks),'REVIEW',[c for c in checks if c['status']!='PASS'])
print(quarter.tail(3)[corecols].to_string())
