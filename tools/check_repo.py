#!/usr/bin/env python3
"""Dependency-free structural quality gate for the whole repository.

Checks documentation contracts that are easy to regress in review:
- GitHub display math stays on one line and avoids unsupported forms;
- GitHub inline math renders: no doubled backslashes in prose, every
  prose `$` is either valid `$...$` math (whitespace-separated from
  adjacent text so GitHub parses it as a math node, especially next to
  CJK) or an escaped `\$` literal (currency must never be a bare `$`);
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

# --- inline-math rendering rules (GitHub) ---------------------------------
FENCE_RE = re.compile(r"(```.*?```)", re.S)
CODE_SPAN_RE = re.compile(r"(`+)(.+?)\1")
DOLLAR_RE = re.compile(r"(?<!\\)(?<!\$)\$(?!\$)")
BS_RE = re.compile(r"\\\\([a-zA-Z])")
MATH_CHARS_RE = re.compile(r"[=<>^_{}]")
CJK_RE = re.compile(r"[一-鿿　-〿＀-￯]")


def _looks_math(inner: str) -> bool:
    s = inner.strip()
    if not s:
        return False
    s = s.replace("\\$", "")  # escaped \$ is literal text, not a TeX command
    if "\\" in s:  # TeX command: \in \times \mu \%
        return True
    if MATH_CHARS_RE.search(s):  # = < > ^ _ { }
        return True
    if re.fullmatch(r"[A-Za-z](_[A-Za-z0-9]+)?", s):  # I, x, Q_i
        return True
    if re.fullmatch(r"[\d.,]+\s*[A-Za-z]+", s):  # 4N, 2MNK
        return True
    if re.fullmatch(r"[A-Za-z\\]+[\(（].*[\)）]", s):  # TracIn(...), f(...)
        return True
    if CJK_RE.search(s):
        return False
    if " " in s or "\t" in s:
        return False  # spaced $...$ with only +*/|- inside is a currency mispair
    if re.search(r"[+*/|]", s):  # K/V, n/p, (G-1)/G
        return True
    if re.search(r"[A-Za-z]", s) and "-" in s:  # p-1, -g (but not 5-10)
        return True
    return False


def _fix_prose_line(line: str) -> str:
    line = BS_RE.sub(r"\\\1", line)
    out: list[str] = []
    pos = 0
    while True:
        m1 = DOLLAR_RE.search(line, pos)
        if not m1:
            out.append(line[pos:])
            break
        p = m1.start()
        m2 = DOLLAR_RE.search(line, m1.end())
        out.append(line[pos:p])
        if not m2:
            out.append("\\$")
            pos = m1.end()
            continue
        q = m2.start()
        if _looks_math(line[m1.end() : q]):
            if out and out[-1] and out[-1][-1] not in " \t":
                out.append(" ")
            out.append(line[p : m2.end()])
            if m2.end() < len(line) and line[m2.end()] not in " \t\n":
                out.append(" ")
            pos = m2.end()
        else:
            out.append("\\$")
            pos = m1.end()
    return "".join(out)


def _check_inline_math(relative: Path, text: str) -> list[str]:
    """Every prose `$` must be valid math or an escaped `\\$` literal."""
    errors: list[str] = []
    in_fence = False
    for line_no, line in enumerate(text.split("\n"), 1):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        # check prose segments between inline code spans
        segments: list[str] = []
        last = 0
        for m in CODE_SPAN_RE.finditer(line):
            segments.append(line[last : m.start()])
            last = m.end()
        segments.append(line[last:])
        for seg in segments:
            if _fix_prose_line(seg) != seg:
                errors.append(
                    f"{relative}:{line_no}: inline math must be `$...$` "
                    f"separated from text, or escape currency as `\\$`: "
                    f"{line.strip()[:110]}"
                )
                break
    return errors


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

    errors.extend(_check_inline_math(relative, text))

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
