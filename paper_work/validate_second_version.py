"""Read-only structural checks for the second-version manuscript and PDF."""

from pathlib import Path
import re

from pypdf import PdfReader

work = Path(__file__).resolve().parent
manuscript = work / "四问正式论文_第二版.md"
pdf = work / "四问正式论文_第二版.pdf"
source = manuscript.read_text(encoding="utf-8")
reader = PdfReader(pdf)

images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", source)
missing = [item for item in images if not (work / item).resolve().is_file()]
assert not missing, f"Missing figures: {missing}"
assert 48 <= len(reader.pages) <= 52, len(reader.pages)
assert len(re.findall(r"(?m)^## 摘要$", source)) == 1
assert "## 目录" not in source
for heading in [
    "## 1 问题重述", "## 2 问题分析", "## 3 模型假设",
    "## 4 符号说明", "## 5 数据预处理与基础分析",
    "## 6 问题一", "## 7 问题二", "## 8 问题三",
    "## 9 问题四", "## 10 模型综合检验", "## 11 模型评价与推广",
    "## 参考文献", "## 附录 A", "## 附录 B", "## 附录 C",
    "## 附录 D", "## 附录 E", "## 附录 F",
]:
    assert heading in source, heading

page_text = [page.extract_text() or "" for page in reader.pages]
assert all(len(text) > 30 for text in page_text)
for value in ["0.2543", "0.2702", "0.08144", "0.05235", "11.864", "22.35"]:
    assert value in source, value

print(f"PASS: {len(reader.pages)} pages, {len(images)} figure references resolved, sections and key values present")
