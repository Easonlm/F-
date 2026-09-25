"""Render the long Chinese manuscript as a paginated A4 review PDF."""

from pathlib import Path
from io import BytesIO
import re
import subprocess

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "paper_work"
SOURCE = WORK / "四问正式论文_第二版.md"
HTML = WORK / "四问正式论文_第二版.html"
RAW_PDF = WORK / "四问正式论文_第二版_未编号.pdf"
FINAL_PDF = WORK / "四问正式论文_第二版.pdf"
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

css = r"""
@page { size: A4; margin: 24mm 24mm 23mm 24mm; }
html { font-size: 12pt; }
body { color: #000; font-family: SimSun, 'Songti SC', serif; font-size: 12pt;
       line-height: 1.15; text-align: justify; margin: 0; max-width: none;
       padding: 0; width: auto; }
h1 { font-family: SimHei, sans-serif; font-size: 16pt; font-weight: bold;
     text-align: center; line-height: 1.35; margin: 0 0 12pt; }
h2 { font-family: SimHei, sans-serif; font-size: 14pt; font-weight: bold;
     text-align: center; line-height: 1.25; margin: 18pt 0 10pt; break-after: avoid; }
h2[id^="附录-a"] { break-before: page; }
h3 { font-family: SimHei, sans-serif; font-size: 12pt; font-weight: bold;
     text-align: left; margin: 13pt 0 6pt; break-after: avoid; }
h4 { font-family: SimHei, sans-serif; font-size: 12pt; font-weight: bold;
     margin: 10pt 0 5pt; break-after: avoid; }
p { margin: 0 0 7pt; text-indent: 2em; orphans: 2; widows: 2; }
ul,ol { margin-top: 4pt; margin-bottom: 8pt; padding-left: 2.2em; }
li { margin-bottom: 3pt; }
li p { text-indent: 0; }
table { border-collapse: collapse; width: 100%; margin: 9pt auto 11pt;
        font-size: 9.5pt; line-height: 1.15; table-layout: auto; }
th,td { border: 0.5pt solid #555; padding: 3pt 4pt; vertical-align: top;
        overflow-wrap: anywhere; }
th { font-family: SimHei, sans-serif; font-weight: bold; background: #f5f5f5; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
figure { text-align: center; margin: 10pt auto 12pt; break-inside: avoid; }
figure img, p > img { display: block; margin: 0 auto; max-width: 90%; max-height: 145mm; object-fit: contain; }
figcaption { text-align: center; font-size: 10pt; margin-top: 4pt; }
img { max-width: 100%; }
blockquote { margin: 0 0 7pt; padding: 0; border: 0; color: #000; }
blockquote p { font-size: 10pt; text-indent: 0; }
code { font-family: Consolas, monospace; font-size: 9pt; overflow-wrap: anywhere; }
pre { font-family: Consolas, monospace; font-size: 8.5pt; white-space: pre-wrap;
      overflow-wrap: anywhere; break-inside: avoid; }
math { font-family: 'Times New Roman', SimSun, serif; font-size: 11pt; }
.display.math { display: block; overflow: hidden; text-align: center; margin: 8pt 0 11pt;
                break-inside: avoid; }
.cover { height: 240mm; display: flex; flex-direction: column; align-items: center;
         justify-content: space-between; text-align: center; break-after: page; }
.cover-logos { width: 100%; display: grid; grid-template-columns: repeat(4, 1fr);
               align-items: center; gap: 7mm; margin-top: 6mm; }
.cover-logos img { max-width: 100%; max-height: 25mm; object-fit: contain; }
.cover-top { margin-top: 21mm; font-family: SimHei, sans-serif; font-size: 16pt; }
.cover-title { font-family: SimHei, sans-serif; font-size: 23pt; font-weight: bold;
               line-height: 1.55; margin: 25mm 10mm; }
.cover-meta { font-size: 14pt; line-height: 2.1; text-align: left; }
.cover-bottom { font-size: 11pt; margin-bottom: 8mm; }
.body-start { break-before: page; }
"""

subprocess.run(
    ["pandoc", str(SOURCE), "--from=markdown+tex_math_dollars+tex_math_single_backslash+pipe_tables",
     "--to=html5", "--standalone", "--mathml", "--metadata=title:算力约束下大语言模型训练资源的分层配置与能力前沿分析", "--output", str(HTML)],
    check=True, cwd=WORK,
)
html = HTML.read_text(encoding="utf-8")
html = html.replace("<title>四问正式论文_长稿</title>",
                    "<title>算力约束下大语言模型训练资源的分层配置与能力前沿分析</title>")
html = html.replace("</head>", f"<style>{css}</style></head>")
html = re.sub(r'<h1 class="title">.*?</h1>\s*', '', html, count=1, flags=re.S)
cover = """<section class="cover">
<div class="cover-logos"><img src="template_media/jpg_1.jpg" alt="华为标识">
<img src="template_media/png_1.png" alt="中国研究生创新实践系列大赛标识">
<img src="template_media/png_2.png" alt="中国研究生数学建模竞赛标识">
<img src="template_media/png_3.png" alt="西安交通大学标识"></div>
<div class="cover-top">2026 年中国研究生数学建模竞赛<br>F 题论文</div>
<div class="cover-title">算力约束下大语言模型训练资源的<br>分层配置与能力前沿分析</div>
<div class="cover-meta">参赛队编号：〔待填写〕<br>参赛学校：〔待填写〕<br>参赛队员：〔待填写〕</div>
<div class="cover-bottom">封面信息待参赛队按官方模板核对后定稿</div>
</section>"""
html = html.replace("<body>", "<body>" + cover, 1)
html = re.sub(r'(<h2[^>]*>1 问题重述</h2>)', r'<div class="body-start"></div>\1', html, count=1)
HTML.write_text(html, encoding="utf-8")

subprocess.run(
    [str(CHROME), "--headless", "--disable-gpu", "--no-sandbox", "--disable-extensions",
     "--allow-file-access-from-files", "--no-pdf-header-footer", "--virtual-time-budget=30000",
     f"--print-to-pdf={RAW_PDF}", HTML.as_uri()],
    check=True, cwd=WORK, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)

reader = PdfReader(str(RAW_PDF))
writer = PdfWriter()
for i, page in enumerate(reader.pages):
    if i:
        overlay = BytesIO()
        c = canvas.Canvas(overlay, pagesize=(float(page.mediabox.width), float(page.mediabox.height)))
        c.setFont("Helvetica", 9)
        c.drawCentredString(float(page.mediabox.width) / 2, 21, str(i))
        c.save()
        overlay.seek(0)
        page.merge_page(PdfReader(overlay).pages[0])
    writer.add_page(page)
with FINAL_PDF.open("wb") as f:
    writer.write(f)
print(f"{FINAL_PDF}: {len(reader.pages)} pages")
