import re
import unittest
from pathlib import Path


TOPIC = Path(__file__).resolve().parents[1]


class TestDocumentation(unittest.TestCase):
    def test_github_math_is_single_line_and_uses_allowed_commands(self):
        for path in TOPIC.rglob("*.md"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(r"\operatorname", text, path)
            self.assertNotIn(r"\[", text, path)
            self.assertNotIn(r"\]", text, path)
            for line_no, line in enumerate(text.splitlines(), start=1):
                if "$$" in line:
                    self.assertEqual(line.count("$$"), 2, f"{path}:{line_no}")

    def test_no_internal_citation_markers(self):
        for path in TOPIC.rglob("*.md"):
            self.assertNotIn("【", path.read_text(encoding="utf-8"), path)

    def test_relative_markdown_links_resolve(self):
        link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
        for path in TOPIC.rglob("*.md"):
            text = path.read_text(encoding="utf-8")
            for target in link_pattern.findall(text):
                if "://" in target or target.startswith("#"):
                    continue
                target_path = target.split("#", 1)[0]
                self.assertTrue((path.parent / target_path).exists(), f"broken link {path}: {target}")


if __name__ == "__main__":
    unittest.main()
