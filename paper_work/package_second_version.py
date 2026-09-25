"""Package the current second-version manuscript and its figures for sharing."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import zipfile
from pathlib import Path


BASE = Path(__file__).resolve().parent
SOURCE = BASE / "四问正式论文_第二版.md"
PACKAGE = BASE / "四问正式论文_第二版_图文分享包_20260926"
ARCHIVE = PACKAGE.with_suffix(".zip")
PANDOC = Path(r"C:\T_download\APP_\Anaconda\Anaconda3\Scripts\pandoc.exe")
IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def main() -> None:
    if PACKAGE.exists() or ARCHIVE.exists():
        raise FileExistsError("分享包已存在，避免覆盖：" + str(PACKAGE))

    original = SOURCE.read_bytes()
    manuscript = original.decode("utf-8-sig")
    matches = list(IMAGE_PATTERN.finditer(manuscript))
    if not matches:
        raise ValueError("未找到图片引用")

    resolved = []
    for match in matches:
        image_path = (SOURCE.parent / match.group(2)).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(str(image_path))
        resolved.append(image_path)

    PACKAGE.mkdir()
    figures_dir = PACKAGE / "图片"
    figures_dir.mkdir()
    copied: dict[Path, str] = {}
    figure_list = []
    match_index = 0

    def rewrite_image(match: re.Match[str]) -> str:
        nonlocal match_index
        image_path = resolved[match_index]
        match_index += 1
        if image_path not in copied:
            name = f"{len(copied) + 1:02d}_{image_path.name}"
            shutil.copy2(image_path, figures_dir / name)
            copied[image_path] = name
        name = copied[image_path]
        figure_list.append((match.group(1), name))
        return f"![{match.group(1)}](图片/{name})"

    revised = IMAGE_PATTERN.sub(rewrite_image, manuscript)
    packaged_md = PACKAGE / SOURCE.name
    packaged_md.write_text(revised, encoding="utf-8")

    index_lines = ["# 图片索引", "", "图片均在 `图片` 文件夹，可单独复制。", ""]
    for ordinal, (caption, name) in enumerate(figure_list, start=1):
        index_lines.append(f"{ordinal}. {caption}：`图片/{name}`")
    (PACKAGE / "图片索引.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    readme = """# 分享说明

请先解压整个 ZIP，再打开其中的 `四问正式论文_第二版.md`。Markdown 与 `图片` 文件夹须保持在同一级目录，图片才会正常显示。

也可以直接用浏览器打开 `四问正式论文_第二版_单文件.html`；它已嵌入所有图片，不依赖图片文件夹，适合只发送一个文件。浏览器内可右键复制或保存图片。

全部图片原文件都放在 `图片` 文件夹，可直接复制、另存或拖入其他文档。`图片索引.md` 列出正文图片与文件名的对应关系。
"""
    (PACKAGE / "分享说明.md").write_text(readme, encoding="utf-8")

    html = PACKAGE / "四问正式论文_第二版_单文件.html"
    subprocess.run(
        [
            str(PANDOC),
            str(packaged_md.name),
            "--from=markdown+tex_math_dollars+tex_math_single_backslash+pipe_tables",
            "--to=html5",
            "--standalone",
            "--self-contained",
            "--mathml",
            "--metadata=title:四问正式论文（第二版）",
            f"--output={html.name}",
        ],
        cwd=PACKAGE,
        check=True,
    )

    packaged_matches = IMAGE_PATTERN.findall(packaged_md.read_text(encoding="utf-8"))
    assert len(packaged_matches) == len(matches)
    assert all((PACKAGE / path).is_file() for _, path in packaged_matches)
    assert SOURCE.read_bytes() == original
    html_text = html.read_text(encoding="utf-8")
    assert "data:image/" in html_text
    assert not re.search(r'<img[^>]+src="图片/', html_text)

    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for path in sorted(PACKAGE.rglob("*")):
            if path.is_file():
                output.write(path, arcname=path.relative_to(PACKAGE.parent))
    with zipfile.ZipFile(ARCHIVE) as output:
        assert output.testzip() is None
        assert len([n for n in output.namelist() if "/图片/" in n]) == len(copied)

    print(f"原稿 SHA-256: {hashlib.sha256(original).hexdigest()}")
    print(f"正文图片引用: {len(matches)}；独立图片文件: {len(copied)}")
    print(f"单文件 HTML: {html.stat().st_size:,} 字节")
    print(f"ZIP: {ARCHIVE} ({ARCHIVE.stat().st_size:,} 字节)")


if __name__ == "__main__":
    main()
