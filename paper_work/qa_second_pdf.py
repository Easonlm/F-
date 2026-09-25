"""Render a contact sheet and representative full pages for visual QA."""

from pathlib import Path
import pypdfium2 as pdfium
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parent
PDF = ROOT / "四问正式论文_第二版.pdf"
OUT = ROOT / "qa_render" / "second_version"
OUT.mkdir(parents=True, exist_ok=True)

pdf = pdfium.PdfDocument(str(PDF))
thumbs = []
for i in range(len(pdf)):
    page = pdf.get_page(i)
    bitmap = page.render(scale=0.6)
    im = bitmap.to_pil().convert("RGB")
    im.thumbnail((220, 310))
    cell = Image.new("RGB", (240, 340), "white")
    cell.paste(im, ((240 - im.width) // 2, 10))
    ImageDraw.Draw(cell).text((110, 318), str(i + 1), fill="black")
    thumbs.append(cell)
    if i + 1 in {1, 2, 7, 15, 23, 29, 30, 31, 39, 49, 50}:
        page.render(scale=1.6).to_pil().convert("RGB").save(OUT / f"page_{i+1:02d}.png")

sheet = Image.new("RGB", (240 * 5, 340 * 10), "#e4e8ec")
for i, im in enumerate(thumbs):
    sheet.paste(im, ((i % 5) * 240, (i // 5) * 340))
sheet.save(OUT / "contact_sheet.png")
print(f"Rendered {len(pdf)} pages to {OUT}")
