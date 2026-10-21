import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const root=path.resolve(import.meta.dirname,'../data');
const tabs=JSON.parse(await fs.readFile(path.join(root,'documentation/workbook_data.json'),'utf8'));
const wb=Workbook.create();
const guide=wb.worksheets.add('Почніть тут');guide.showGridLines=false;guide.tabColor='#17365D';
guide.getRange('B2').values=[['Датасет для власної QPM']];guide.getRange('B2').format.font={name:'Arial',size:16,bold:true,color:'#17365D'};
const rows=[
 ['Версія','21.10.2026. Основні джерела: 20.10; додаткові: 21.10'],
 ['Повна вибірка','66 кварталів: 2010Q1–2026Q2; 7 основних спостережень'],
 ['Початок роботи','Аркуш «Повна вибірка» або model/quarterly_training.csv'],
 ['Комірки Excel','Кожен показник і кожне число записані в окрему комірку'],
 ['Ціни','Дослідницька SA: robust log-STL; оригінали NSA збережено'],
 ['ВВП','Офіційні SA / SCA, без повторного коригування'],
 ['Ставки','НБУ та ЄЦБ: середні за календарні дні, % річних'],
 ['III квартал 2026','Ціни й ставки є; фактичний ВВП ще відсутній'],
 ['Додаткові ряди','UONIA, кредитні ставки, зарплата, торгівля, сировина, FAO'],
 ['Безробіття МОП','Квартальний ряд віку 15+ за 2019–2021; далі факту немає'],
 ['Прогнози НБУ','Окремий матеріал для порівняння; не включений у факти'],
 ['Пропуски','Порожні комірки збережені; нулями не замінено'],
 ['Коефіцієнти моделі','Потенціал ВВП, нейтральні ставки й параметри ще оцінює QPM'],
 ['Чутливість SA','У звіті є порівняння налаштувань; SA не є офіційною'],
 ['CSV для моделей','Кома між полями, десяткова крапка; відкривати імпортом'],
 ['CSV для Excel','Окрема папка: крапка з комою між полями, десяткова кома'],
 ['Для перегляду','Відкривайте цей XLSX — він не залежить від CSV-розділювача'],
 ['Методика та джерела','Докладно у README.md, documentation/Методика.md'],
];
guide.getRangeByIndexes(4,1,rows.length,2).values=rows;
guide.getRange('B5:C22').format.font={name:'Arial',size:11,color:'#202B3C'};
guide.getRange('B5:B22').format.font.bold=true;
guide.getRange('B5:B22').format.columnWidth=30;guide.getRange('C5:C22').format.columnWidth=90;
guide.getRange('B5:C22').format.rowHeight=28;
guide.getRange('B5:C22').format.verticalAlignment='center';
guide.getRange('B9:C9').format.fill='#FFF2CC';guide.getRange('B12:C12').format.fill='#FFF2CC';
const all=['Почніть тут'];
for(const name of ['Повна вибірка',...Object.keys(tabs).filter(x=>x!=='Повна вибірка')]){
 const records=tabs[name];
 if(!records.length)continue;
 const s=wb.worksheets.add(name);all.push(name);s.showGridLines=false;
 const keys=Object.keys(records[0]);const matrix=[keys,...records.map(r=>keys.map(k=>r[k]??null))];
 const range=s.getRangeByIndexes(0,0,matrix.length,keys.length);range.values=matrix;
 range.format.font={name:'Arial',size:10,color:'#202B3C'};range.format.rowHeight=21;range.format.verticalAlignment='center';
 const head=s.getRangeByIndexes(0,0,1,keys.length);head.format.fill='#17365D';head.format.font={name:'Arial',size:10,bold:true,color:'#FFFFFF'};head.format.wrapText=true;head.format.rowHeight=44;
 s.freezePanes.freezeRows(1);
 for(let j=0;j<keys.length;j++){
  const key=keys[j];const rg=s.getRangeByIndexes(0,j,matrix.length,1);
  let width=24;
  if(['period','first','last','n','value'].includes(key))width=16;
  if(['series','variable'].includes(key))width=34;
  if(['source','sources','file','url'].includes(key))width=50;
  if(['meaning','transformation','check','detail','selection','purpose','missing_periods','method'].includes(key))width=58;
  if(['status','role','unit','verdict'].includes(key))width=28;
  rg.format.columnWidth=width;
  if(records.some(r=>typeof r[key]==='number'))s.getRangeByIndexes(1,j,records.length,1).setNumberFormat(['n','sample_n','horizon_months','missing_in_2010_2026Q2','n_observations','sheets','numeric_cells','source_error_cells'].includes(key)?'0':'#,##0.0000');
 }
 if(['Словник','Джерела','Перевірки','Покриття'].includes(name)){
  range.format.wrapText=true;range.format.verticalAlignment='top';range.format.rowHeight=name==='Джерела'?78:name==='Словник'?52:34;head.format.rowHeight=34;
 }
 if(name==='Повна вибірка')s.tabColor='#17365D';
 if(name==='Перевірки')s.getRangeByIndexes(1,1,records.length,1).conditionalFormats.add('containsText',{text:'REVIEW',format:{fill:'#FCE4D6',font:{color:'#9C0006',bold:true}}});
}
wb.recalculate();
console.log((await wb.inspect({kind:'table',range:"'Повна вибірка'!A1:H4",include:'values',tableMaxRows:4,tableMaxCols:8})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},summary:'Final errors'})).ndjson);
const exported=await SpreadsheetFile.exportXlsx(wb);await exported.save(path.join(root,'QPM_ready.xlsx'));
const pv=path.join(import.meta.dirname,'previews');await fs.mkdir(pv,{recursive:true});
for(const name of process.argv.includes('--final-touch')?['Почніть тут','Квартальні очікування','Аудит Excel-джерел']:all){
 const range=name==='Почніть тут'?'B2:C22':name==='Джерела'?'A1:D5':name==='Словник'?'A1:D5':name==='Перевірки'?'A1:C6':name==='Покриття'?'A1:F7':'A1:H8';
 const img=await wb.render({sheetName:name,range,scale:1,format:'png'});await fs.writeFile(path.join(pv,name+'.png'),new Uint8Array(await img.arrayBuffer()));
}
await fs.writeFile(path.join(import.meta.dirname,'rendered.json'),JSON.stringify(all));
console.log('Saved QPM_ready.xlsx, sheets: '+all.length);
