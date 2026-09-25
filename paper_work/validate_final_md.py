"""Check that the reviewer-facing Markdown remains self-consistent."""

from pathlib import Path
import re

work = Path(__file__).resolve().parent
manuscript = work / "四问正式论文_第二版.md"
source = manuscript.read_text(encoding="utf-8")

assert not re.search(r"\bV[12]\b", source, re.I)
assert "\t" not in source and "\b" not in source

images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", source)
missing = [item for item in images if not (work / item).resolve().is_file()]
assert not missing, missing

non_image = "\n".join(row for row in source.splitlines() if not row.lstrip().startswith("!["))
code_tokens = re.findall(r"`([^`\n]+)`", non_image)
local_tokens = [token for token in code_tokens if re.search(r"\.(?:csv|json|py|joblib|md)$|(?:outputs|rebuild)/", token)]
assert not local_tokens, local_tokens
assert not re.search(r"(?<!!)\[[^]]+\]\((?!https?://)[^)]+\)", non_image)

for heading in [
    "## 摘要", "## 1 问题重述", "## 6 问题一", "## 7 问题二",
    "## 8 问题三", "## 9 问题四", "## 10 模型综合检验",
    "## 11 模型评价与推广", "## 参考文献", "## 附录 A",
    "## 附录 B", "## 附录 C", "## 附录 D", "## 附录 E", "## 附录 F",
]:
    assert heading in source, heading

for value in ["0.254346", "0.052346", "11.864", "35.897475", "1.9794"]:
    assert value in source, value

assert "| **Q×N 条件项** | **0.051804** | **0.052346** |" in source
print(f"PASS: {len(source.splitlines())} lines, {len(images)} figures resolved, no version labels or local file citations in prose")
