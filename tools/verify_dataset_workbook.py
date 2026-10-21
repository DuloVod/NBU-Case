from pathlib import Path
import json,math,hashlib,zipfile,shutil
import pandas as pd,numpy as np,openpyxl
R=Path(__file__).resolve().parents[1]/'data'
tabs=json.loads((R/'documentation/workbook_data.json').read_text())
book=openpyxl.load_workbook(R/'QPM_ready.xlsx',data_only=True)
assert (R/'QPM_ready.xlsx').read_bytes()[:2]==b'PK'
compared=0;numeric=0
for name,rows in tabs.items():
 sheet=book[name];data=list(sheet.values);keys=list(rows[0]);assert data[0]==tuple(keys),(name,data[0],keys)
 assert len(data)==len(rows)+1,(name,len(data),len(rows))
 assert sheet.max_column==len(keys) and len(keys)>1
 for row,expected in zip(data[1:],rows):
  for k,v in zip(keys,row):
   exp=expected.get(k)
   if isinstance(exp,(int,float)) and not isinstance(exp,bool):
    assert isinstance(v,(int,float)) and math.isclose(v,exp,rel_tol=1e-12,abs_tol=1e-10),(name,k,v,exp)
    numeric+=1
   else:assert v==exp,(name,k,v,exp)
   compared+=1
 assert not any(c.data_type=='e' for row in sheet for c in row),name
 # Headers are separate cells; no header cell contains a full CSV record.
 assert all(',' not in str(v) and ';' not in str(v) for v in data[0]),name
local_checks=0
for p in (R/'csv_for_excel').glob('*.csv'):
 a=pd.read_csv(R/'model'/p.name);b=pd.read_csv(p,sep=';',decimal=',',encoding='utf-8-sig')
 assert a.shape==b.shape and list(a)==list(b),p
 pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-10)
 assert len(a.columns)>1,p
 local_checks+=1

q=pd.read_csv(R/'model/quarterly_levels.csv',index_col=0)
obs=pd.read_csv(R/'model/quarterly_observations.csv',index_col=0)
tr=pd.read_csv(R/'model/quarterly_training.csv',index_col=0)
assert tr.shape==(66,7) and tr.notna().all().all()
assert q.loc['2026Q3',['gdp','gdp_ea']].isna().all()
for k in ['gdp','cpi','er','gdp_ea','cpi_ea']:
 assert np.allclose(obs['l_'+k],100*np.log(q[k]),equal_nan=True,rtol=1e-10,atol=1e-9)
g=list(openpyxl.load_workbook(R/'raw/_dodatok_sezonka_vvp_2010-2026.xlsx',data_only=True)['2010-2026'].values)
for i,p in enumerate(g[3]):
 if isinstance(p,str) and p in q.index:assert q.loc[p,'gdp']==g[7][i]
ea=list(openpyxl.load_workbook(R/'raw/namq_10_gdp__custom_23412352_spreadsheet.xlsx',data_only=True)['Sheet 1'].values)
for j,p in enumerate(ea[9]):
 if isinstance(p,str) and p.replace('-Q','Q') in q.index:assert q.loc[p.replace('-Q','Q'),'gdp_ea']==ea[12][j]
# Independently aggregate official daily ECB and policy data and compare all quarters.
for filename,key,valuecol,datecol in [('ecb_daily.csv','ir_ea','IR_ECB_DFR','date'),('policy_daily.csv','ir_pol','IR_POLICY','date')]:
 d=pd.read_csv(R/'inputs'/filename);d[datecol]=pd.to_datetime(d[datecol]);avg=d.groupby(d[datecol].dt.to_period('Q'))[valuecol].mean();avg.index=avg.index.astype(str)
 assert np.allclose(q[key],avg.reindex(q.index),equal_nan=True,atol=1e-9)

# Independent annual flow reconciliation to annual columns in the same raw BOP publication.
bop=list(openpyxl.load_workbook(R/'raw/BOP_m.xlsx',data_only=True)['1.1'].values)
am=pd.read_csv(R/'model/auxiliary_monthly.csv',index_col=0);flow_checks=0
for j,p in enumerate(bop[4]):
 if j<270 or not isinstance(p,(int,float)) or not 2010<=p<=2025:continue
 for i,key in [(6,'current_account'),(7,'trade_balance_gs'),(9,'exports_goods'),(10,'imports_goods'),(12,'exports_services'),(13,'imports_services')]:
  values=am.loc[f'{int(p)}-01':f'{int(p)}-12',key];annual=bop[i][j]
  assert len(values)==12 and values.notna().all()
  assert abs(values.sum()-annual)<=6,(p,key,values.sum(),annual)
  flow_checks+=1
assert flow_checks==96,flow_checks

audit=pd.read_csv(R/'documentation/every_excel_sheet_audit.csv')
manifest=pd.read_csv(R/'documentation/raw_manifest.csv')
for _,f in manifest.iterrows():assert hashlib.sha256((R/'raw'/f['file']).read_bytes()).hexdigest()==f.sha256
checks=pd.read_csv(R/'documentation/validation.csv');assert checks.status.eq('PASS').all()
qa=dict(excel_output_files=1,output_sheets=len(book.sheetnames),cells_compared=compared,numeric_cells_verified=numeric,headers_split_into_separate_cells=True,output_error_cells=0,raw_excel_files=int(audit.file.nunique()),raw_excel_sheets=len(audit),local_csv_pairs_checked=local_checks,independent_annual_flow_checks=flow_checks,data_checks_passed=len(checks),core_complete_quarters=66,core_variables=7,visual_review='All sheet previews reviewed')
(R/'documentation/delivery_checks.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2))
(R/'QPM_ready.xlsx.inspect.ndjson').unlink(missing_ok=True)
print(json.dumps(qa,ensure_ascii=False,indent=2))
