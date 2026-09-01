import importlib.util
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("metrics_profile", ROOT / "code" / "metrics_profile.py")
metrics = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = metrics
SPEC.loader.exec_module(metrics)


class MetricsParserTests(unittest.TestCase):
    def test_labels_and_histogram_rows_are_preserved(self):
        text = '''# HELP ignored x
vllm:num_requests_running{model_name="m",engine="0"} 3
vllm:time_to_first_token_seconds_bucket{le="0.1"} 7
process_start_time_seconds 1.23e+09
bad line
'''
        rows = metrics.parse_prometheus(text)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["metric"], "vllm:num_requests_running")
        self.assertIn('model_name="m"', rows[0]["labels"])
        self.assertEqual(rows[1]["value"], 7.0)
        self.assertEqual(len(metrics.selected(rows, False)), 2)
        self.assertEqual(len(metrics.selected(rows, True)), 3)


if __name__ == "__main__":
    unittest.main()
