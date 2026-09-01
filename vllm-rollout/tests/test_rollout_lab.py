import importlib.util
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("rollout_lab", ROOT / "code" / "rollout_lab.py")
lab = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = lab
SPEC.loader.exec_module(lab)


class CapacityTests(unittest.TestCase):
    def test_kv_bytes_mha_and_gqa(self):
        mha = lab.kv_bytes_per_token(lab.ModelShape(32, 32, 128, 2))
        gqa = lab.kv_bytes_per_token(lab.ModelShape(32, 8, 128, 2))
        self.assertEqual(mha, 524288)
        self.assertEqual(gqa, 131072)
        self.assertEqual(mha / gqa, 4)

    def test_capacity_subtracts_non_kv_memory(self):
        shape = lab.ModelShape(1, 1, 1, 1)
        result = lab.estimate_kv_capacity_tokens(10, 0.8, 3, 1, shape)
        self.assertEqual(result["kv_budget_gib"], 4)
        self.assertEqual(result["estimated_kv_tokens"], 2 * 1024**3)


class SimulationTests(unittest.TestCase):
    def load(self, name):
        return lab.SimConfig(**json.loads((ROOT / "configs" / name).read_text()))

    def test_accounting_invariants(self):
        result = lab.simulate(self.load("stable.json"))
        counts, tokens = result["counts"], result["tokens"]
        self.assertEqual(
            counts["arrived"],
            counts["completed"] + counts["timed_out"] + counts["capacity_failed"] + counts["unfinished"],
        )
        self.assertLessEqual(tokens["generated_accepted"], tokens["generated_completed"])
        self.assertLessEqual(tokens["generated_completed"], tokens["generated_attempted"])
        self.assertEqual(tokens["wasted_generated"], tokens["generated_attempted"] - tokens["generated_accepted"])

    def test_overload_degrades_queue_and_completion(self):
        stable = lab.simulate(self.load("stable.json"))
        overloaded = lab.simulate(self.load("overload.json"))
        self.assertGreater(overloaded["peaks"]["queue_depth"], stable["peaks"]["queue_depth"])
        self.assertGreater(
            overloaded["counts"]["timed_out"] + overloaded["counts"]["capacity_failed"],
            stable["counts"]["timed_out"] + stable["counts"]["capacity_failed"],
        )
        self.assertGreaterEqual(overloaded["policy_lag_versions"]["p95"], stable["policy_lag_versions"]["p95"])

    def test_impossible_prompt_is_capacity_failure(self):
        c = lab.SimConfig(duration_s=2, drain_s=1, arrival_rate_rps=3, prompt_tokens_mean=100, output_tokens_mean=4, length_cv=0, kv_capacity_tokens=64)
        result = lab.simulate(c)
        self.assertEqual(result["counts"]["capacity_failed"], result["counts"]["arrived"])
        self.assertEqual(result["counts"]["completed"], 0)


class DocumentationContractTests(unittest.TestCase):
    def test_math_and_relative_links(self):
        for path in ROOT.glob("*.md"):
            text = path.read_text()
            self.assertNotIn("\\operatorname", text, path)
            self.assertNotIn("\\[", text, path)
            self.assertNotIn("\\]", text, path)
            self.assertEqual(text.count("$$") % 2, 0, path)
            for line in text.splitlines():
                if "$$" in line:
                    self.assertEqual(line.count("$$"), 2, f"display math must be one line: {path}: {line}")
        for path in ROOT.rglob("*.md"):
            text = path.read_text()
            for target in __import__("re").findall(r"\[[^]]+\]\((?!https?://|#)([^)#]+)(?:#[^)]+)?\)", text):
                self.assertTrue((path.parent / target).resolve().exists(), f"broken link {path}: {target}")

    def test_no_employer_identifier(self):
        banned = __import__("re").compile(r"\b(?:meta|facebook)\b", __import__("re").IGNORECASE)
        for path in ROOT.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path != pathlib.Path(__file__):
                self.assertIsNone(banned.search(path.read_text(errors="ignore")), path)


if __name__ == "__main__":
    unittest.main()
