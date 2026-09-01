import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOPIC = ROOT / "gpu-architecture"


class DocumentationContractTest(unittest.TestCase):
    def test_github_display_math_is_single_line(self):
        for path in TOPIC.glob("*.md"):
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                self.assertEqual(
                    line.count("$$") % 2,
                    0,
                    f"{path.name}:{line_no}: display math must open and close on one line",
                )
                self.assertNotIn("\\[", line, f"{path.name}:{line_no}")
                self.assertNotIn("\\]", line, f"{path.name}:{line_no}")
                self.assertNotIn("\\operatorname", line, f"{path.name}:{line_no}")

    def test_local_markdown_links_resolve(self):
        files = [ROOT / "README.md", *TOPIC.glob("*.md")]
        pattern = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
        for path in files:
            for target in pattern.findall(path.read_text(encoding="utf-8")):
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                relative = target.split("#", 1)[0]
                if relative:
                    self.assertTrue((path.parent / relative).exists(), f"{path}: broken link {target}")

    def test_navigation_covers_every_chapter(self):
        readme = (TOPIC / "README.md").read_text(encoding="utf-8")
        for chapter in sorted(TOPIC.glob("[0-9][0-9]_*.md")):
            self.assertIn(f"({chapter.name})", readme)
            text = chapter.read_text(encoding="utf-8")
            self.assertIn("## 导航", text)

    def test_public_topic_has_no_employer_identifiers(self):
        terms = ("Me" + "ta", "A" + "AI")
        forbidden = re.compile(
            r"(?<![A-Za-z0-9_])(" + "|".join(terms) + r")(?![A-Za-z0-9_])",
            re.IGNORECASE,
        )
        for path in TOPIC.rglob("*"):
            if path.is_file() and path.suffix in {".md", ".py"}:
                self.assertIsNone(forbidden.search(path.read_text(encoding="utf-8")), str(path))


if __name__ == "__main__":
    unittest.main()
