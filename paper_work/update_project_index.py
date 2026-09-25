"""Refresh the manuscript figure map and validate repository-relative index links."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_work" / "四问正式论文_第二版.md"
INDEX = ROOT / "论文结果与项目文件索引.md"
START = "<!-- FIGURE_INDEX_START -->"
END = "<!-- FIGURE_INDEX_END -->"
IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def repo_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def main() -> None:
    paper = PAPER.read_text(encoding="utf-8-sig")
    index = INDEX.read_text(encoding="utf-8")
    if index.count(START) != 1 or index.count(END) != 1:
        raise ValueError("索引缺少唯一的图片区标记")

    section = ""
    rows = ["| 序号 | 论文位置 | 图题 | 仓库相对路径 |", "|---:|---|---|---|"]
    figure_files: set[str] = set()
    for line_number, line in enumerate(paper.splitlines(), start=1):
        if re.match(r"^#{2,4} ", line):
            section = line.lstrip("# ")
        for match in IMAGE.finditer(line):
            source = (PAPER.parent / unquote(match.group(2))).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"论文第 {line_number} 行图片不存在：{source}")
            rel = repo_path(source)
            figure_files.add(rel)
            caption = match.group(1).replace("|", "\\|")
            location = section.replace("|", "\\|")
            rows.append(
                f"| {len(rows) - 1} | {location}（第 {line_number} 行） | {caption} | [{rel}]({rel}) |"
            )

    before, remainder = index.split(START, 1)
    _, after = remainder.split(END, 1)
    refreshed = before + START + "\n" + "\n".join(rows) + "\n" + END + after

    # Validate every link from the repository root; only remote URLs are exempt.
    links = [unquote(raw.split("#", 1)[0]) for raw in LINK.findall(refreshed)]
    local_links = [path for path in links if path and not re.match(r"^[a-zA-Z]+://", path)]
    missing = [path for path in local_links if not (ROOT / path).exists()]
    if missing:
        raise FileNotFoundError("索引中存在失效链接：\n" + "\n".join(sorted(set(missing))))

    tracked = {
        entry.decode("utf-8").replace("\\", "/")
        for entry in subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).split(b"\0")
        if entry
    }
    untracked = [path for path in local_links if not (ROOT / path).is_dir() and path not in tracked]
    if untracked:
        raise ValueError("链接目标未纳入 Git 跟踪：\n" + "\n".join(sorted(set(untracked))))

    INDEX.write_text(refreshed, encoding="utf-8")
    print(f"图片引用 {len(rows) - 2} 处，不同图片 {len(figure_files)} 个；相对链接 {len(local_links)} 个均存在且文件已纳入 Git 跟踪。")
    print(f"索引：{INDEX}")


if __name__ == "__main__":
    main()
