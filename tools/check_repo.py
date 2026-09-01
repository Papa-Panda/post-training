#!/usr/bin/env python3
"""Dependency-free structural quality gate for the whole repository.

Checks documentation contracts that are easy to regress in review:
- GitHub display math stays on one line and avoids unsupported forms;
- local Markdown links resolve;
- Markdown contains no control characters;
- Python sources parse without importing optional ML dependencies.

This is a structural gate, not a substitute for each topic's semantic tests.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache"}
LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def files_with_suffix(suffix: str) -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob(f"*{suffix}")
        if not IGNORED_PARTS.intersection(path.relative_to(ROOT).parts)
    )


def check_markdown(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    relative = path.relative_to(ROOT)

    if CONTROL_RE.search(text):
        errors.append(f"{relative}: contains a control character")
    if "\\operatorname" in text:
        errors.append(f"{relative}: use \\mathrm instead of \\operatorname")
    if "\\[" in text or "\\]" in text:
        errors.append(f"{relative}: use one-line $$...$$ display math")

    for line_no, line in enumerate(text.splitlines(), 1):
        delimiters = line.count("$$")
        if delimiters % 2:
            errors.append(
                f"{relative}:{line_no}: display math must open and close on one line"
            )
    if text.count("$$") % 2:
        errors.append(f"{relative}: unpaired $$ delimiter")

    for match in LINK_RE.finditer(text):
        target = match.group(1).strip().split(maxsplit=1)[0].strip("<>")
        if not target or target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target = unquote(target.split("#", 1)[0])
        if target and not (path.parent / target).resolve().exists():
            errors.append(f"{relative}: broken local link: {target}")
    return errors


def check_python(path: Path) -> list[str]:
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as exc:
        return [f"{path.relative_to(ROOT)}: {exc}"]
    return []


def main() -> int:
    errors: list[str] = []
    markdown_files = files_with_suffix(".md")
    python_files = files_with_suffix(".py")
    for path in markdown_files:
        errors.extend(check_markdown(path))
    for path in python_files:
        errors.extend(check_python(path))

    if errors:
        print("quality gate failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(
        f"quality gate passed: {len(markdown_files)} Markdown files, "
        f"{len(python_files)} Python files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
