"""MkDocs hooks: aggregate article count and character count for the homepage."""

from __future__ import annotations

import re
from pathlib import Path

from mkdocs.config.defaults import MkDocsConfig

_FRONTMATTER = re.compile(r"^---\s*\r?\n.*?\r?\n---\s*\r?\n", re.DOTALL)
_FENCED = re.compile(r"```[\s\S]*?```", re.MULTILINE)


def _strip_frontmatter(text: str) -> str:
    m = _FRONTMATTER.match(text)
    return text[m.end() :] if m else text


def _strip_fenced_code(text: str) -> str:
    return _FENCED.sub("", text)


def _char_count_for_markdown(text: str) -> int:
    """Non-whitespace characters in body text (excludes front matter and fenced code)."""
    body = _strip_frontmatter(text)
    body = _strip_fenced_code(body)
    return len(re.sub(r"\s+", "", body))


def _compute_stats(docs_dir: Path) -> dict[str, int]:
    md_paths = sorted(docs_dir.rglob("*.md"))
    total_chars = 0
    for path in md_paths:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            continue
        total_chars += _char_count_for_markdown(raw)
    return {"articles": len(md_paths), "total_chars": total_chars}


def on_pre_build(config: MkDocsConfig) -> None:
    stats = _compute_stats(Path(config.docs_dir))
    config.extra["site_stats"] = stats


def on_page_markdown(markdown: str, *, page, config: MkDocsConfig, files) -> str:
    if page.file.src_path.replace("\\", "/") != "index.md":
        return markdown
    stats = config.extra.get("site_stats") or {}
    articles = int(stats.get("articles", 0))
    total_chars = int(stats.get("total_chars", 0))
    return (
        markdown.replace("__SITE_ARTICLES__", str(articles)).replace(
            "__SITE_TOTAL_CHARS__", f"{total_chars:,}"
        )
    )
