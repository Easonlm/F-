import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = path.resolve("problem1");
const items = JSON.parse(await fs.readFile(path.join(root, "outputs/tables/v2_blind_review_items.json"), "utf8"));
if (items.length !== 140 || new Set(items.map(x => x.review_id)).size !== 140) {
  throw new Error("Expected exactly 140 unique blind review items");
}
const wb = Workbook.create();
const sheet = wb.worksheets.add("Ratings");
sheet.showGridLines = false;
sheet.getRange("A1:I141").values = [
  ["sample_id", "text", "rater_id", "overall_quality", "fluency",
   "information_education", "cleanliness", "ad_or_spam", "notes"],
  ...items.map(x => [x.review_id, x.text, "", null, null, null, null, null, ""]),
];
sheet.getRange("A1:I1").format = {
  fill: "#273E5E", font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
};
sheet.getRange("A2:I141").format.font = { name: "Arial", size: 10, color: "#20242A" };
sheet.getRange("A2:I141").format.verticalAlignment = "top";
sheet.getRange("C2:I141").format.fill = "#FFF2D7";
sheet.getRange("A1:I1").format.rowHeight = 42;
sheet.getRange("A1:I1").format.wrapText = true;
sheet.getRange("A2:I141").format.rowHeight = 104;
sheet.getRange("B2:B141").format.wrapText = true;
sheet.getRange("I2:I141").format.wrapText = true;
sheet.getRange("A:A").format.columnWidth = 14;
sheet.getRange("B:B").format.columnWidth = 75;
sheet.getRange("C:C").format.columnWidth = 13;
sheet.getRange("D:G").format.columnWidth = 19;
sheet.getRange("H:H").format.columnWidth = 15;
sheet.getRange("I:I").format.columnWidth = 35;
sheet.freezePanes.freezeRows(1);
sheet.freezePanes.freezeColumns(1);
for (const col of ["D", "E", "F", "G"]) {
  sheet.dataValidations.add({
    range: col + "2:" + col + "141",
    rule: { type: "whole", operator: "between", formula1: 1, formula2: 5 },
  });
}
sheet.getRange("H2:H141").dataValidation = { rule: { type: "list", values: ["0", "1"] } };

const guide = wb.worksheets.add("Instructions");
guide.showGridLines = false;
guide.getRange("A1:B10").values = [
  ["Field", "How to fill"],
  ["sample_id", "Fixed random ID; do not change."],
  ["text", "Read the displayed text. The cell stores up to 4,000 characters."],
  ["rater_id", "Enter one consistent reviewer code on every scored row, such as R1."],
  ["overall_quality", "Integer 1–5: 1 very poor, 3 ordinary, 5 excellent."],
  ["fluency", "Integer 1–5: readability and linguistic coherence."],
  ["information_education", "Integer 1–5: useful, informative or educational content."],
  ["cleanliness", "Integer 1–5: absence of noise, duplication and formatting damage."],
  ["ad_or_spam", "Enter 1 if obvious advertising/spam appears, otherwise 0."],
  ["notes", "Optional explanation; leave blank if none."],
];
guide.getRange("A12:B14").values = [
  ["Procedure", "Each reviewer makes an independent copy, scores without discussing samples, then sends the completed file for analysis."],
  ["Blinding", "The rating sheet intentionally contains no Q, conflict flag, source domain or model prediction."],
  ["Scope", "The 140 texts were selected by Q strata and extra conflict samples; analysis is not a random-sample population estimate."],
];
guide.getRange("A1:B1").format = {
  fill: "#273E5E", font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
};
guide.getRange("A2:B14").format.font = { name: "Arial", size: 10, color: "#20242A" };
guide.getRange("A:A").format.columnWidth = 25;
guide.getRange("B:B").format.columnWidth = 95;
guide.getRange("A1:B14").format.rowHeight = 32;
guide.getRange("B2:B14").format.wrapText = true;

const check = await wb.inspect({kind: "table", sheetId: "Ratings", range: "A1:I3",
  include: "values", tableMaxRows: 3, tableMaxCols: 9, maxChars: 1200});
console.log(check.ndjson);
const errors = await wb.inspect({kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: {useRegex: true, maxResults: 20}, maxChars: 1200});
console.log(errors.ndjson);
const outDir = path.join(root, "outputs/v2_review");
await fs.mkdir(outDir, {recursive: true});
const preview = await wb.render({sheetName: "Ratings", range: "A1:I3", scale: 1.5, format: "png"});
await fs.writeFile(path.join(outDir, "人工质量盲评_140条_preview.png"),
  new Uint8Array(await preview.arrayBuffer()));
const guidePreview = await wb.render({sheetName: "Instructions", range: "A1:B14", scale: 1.3, format: "png"});
await fs.writeFile(path.join(outDir, "人工质量盲评_说明_preview.png"),
  new Uint8Array(await guidePreview.arrayBuffer()));
const xlsx = await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(path.join(outDir, "人工质量盲评_140条.xlsx"));
console.log("Created 140-item blind rating workbook");
