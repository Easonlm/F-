import fs from 'node:fs/promises';
import path from 'node:path';
import { SpreadsheetFile, Workbook } from '@oai/artifact-tool';

const root = path.resolve('problem1');
const input = JSON.parse(await fs.readFile(path.join(root, 'outputs/tables/v2_blind_review_items.json'), 'utf8'));
const wb = Workbook.create();
const info = wb.worksheets.add('说明');
const sheet = wb.worksheets.add('盲评');

info.getRange('A1').values = [['问题一质量代理人工盲评']];
info.getRange('A3:B8').values = [
  ['样本', `${input.length} 条 A1 文本，按领域、Q 分位和冲突状态分层抽取`],
  ['评审方式', '两名评审者各自复制此表，独立评分；评审前不要查看另存的抽样对照表。'],
  ['质量维度', '教育性、可读性、洁净度、结构质量、总体质量均填 1–5；5 为最好。'],
  ['问题标记', '广告、乱码、重复、事实问题填 1=有，0=无；无法判断则留空。'],
  ['长文本', '文本单元格最多保留原文前 4000 字；点击单元格可在编辑栏查看完整单元格内容。'],
  ['评估边界', '评分可检验 Q 与人工判断的一致性，不能单独证明训练 Loss 的因果变化。'],
];
info.getRange('A1:B8').format.font = { name: 'Arial', size: 11, color: '#1F2937' };
info.getRange('A1').format.font = { name: 'Arial', size: 15, bold: true, color: '#1F2937' };
info.getRange('A3:A8').format.font = { name: 'Arial', size: 11, bold: true, color: '#334155' };
info.getRange('A:A').format.columnWidth = 18;
info.getRange('B:B').format.columnWidth = 90;
info.getRange('B3:B8').format.wrapText = true;
info.getRange('A3:B8').format.rowHeight = 40;
info.showGridLines = false;

const headers = ['评审编号','文本','教育性 1–5','可读性 1–5','洁净度 1–5','结构质量 1–5','总体质量 1–5','广告 0/1','乱码 0/1','重复 0/1','事实问题 0/1','备注'];
const rows = [headers, ...input.map(x => [x.review_id, x.text, null, null, null, null, null, null, null, null, null, null])];
sheet.getRange(`A1:L${rows.length}`).values = rows;
sheet.getRange('A1:L1').format = { fill: '#334155', font: { name: 'Arial', size: 10, color: '#FFFFFF', bold: true } };
sheet.getRange(`A2:L${rows.length}`).format.font = { name: 'Arial', size: 10, color: '#1F2937' };
sheet.getRange(`B2:B${rows.length}`).format.wrapText = true;
sheet.getRange(`B2:B${rows.length}`).format.verticalAlignment = 'top';
sheet.getRange('A:A').format.columnWidth = 15;
sheet.getRange('B:B').format.columnWidth = 95;
sheet.getRange('C:G').format.columnWidth = 18;
sheet.getRange('H:K').format.columnWidth = 17;
sheet.getRange('L:L').format.columnWidth = 38;
sheet.getRange(`A2:L${rows.length}`).format.rowHeight = 150;
sheet.getRange('A1:L1').format.rowHeight = 32;
sheet.freezePanes.freezeRows(1);
sheet.showGridLines = false;
sheet.getRange(`C2:G${rows.length}`).dataValidation = { rule: { type: 'list', values: ['1','2','3','4','5'] } };
sheet.getRange(`H2:K${rows.length}`).dataValidation = { rule: { type: 'list', values: ['0','1'] } };
const table = sheet.tables.add(`A1:L${rows.length}`, true, 'BlindReview');
table.style = 'TableStyleMedium2';

const check = await wb.inspect({ kind: 'table', range: '盲评!A1:L3', include: 'values,formulas', tableMaxRows: 3, tableMaxCols: 12, maxChars: 1600 });
console.log(check.ndjson);
const errors = await wb.inspect({ kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!', options: { useRegex: true, maxResults: 20 }, maxChars: 1000 });
console.log(errors.ndjson);
const outputDir = path.join(root, 'outputs/v2_review');
await fs.mkdir(outputDir, { recursive: true });
for (const name of ['说明','盲评']) {
  const preview = await wb.render({ sheetName: name, range: name === '说明' ? 'A1:B8' : 'A1:D3', scale: 1, format: 'png' });
  await fs.writeFile(path.join(outputDir, `${name}_preview.png`), new Uint8Array(await preview.arrayBuffer()));
}
const out = await SpreadsheetFile.exportXlsx(wb);
await out.save(path.join(outputDir, '人工盲评模板.xlsx'));
console.log(`saved ${path.join(outputDir, '人工盲评模板.xlsx')}`);
