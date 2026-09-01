#!/usr/bin/env python3
"""Capture selected vLLM Prometheus samples in long-form CSV.

No values are fabricated when the endpoint is unavailable. Metric names are
version-sensitive; unknown names remain visible when --all is used.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List
from urllib.request import urlopen

SAMPLE_RE = re.compile(r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>.*)\})?\s+(?P<value>[-+0-9.eE]+)$')
DEFAULT_PREFIXES = (
    "vllm:num_requests_",
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
    "vllm:num_preemptions",
    "vllm:prompt_tokens",
    "vllm:generation_tokens",
    "vllm:request_success",
    "vllm:request_queue_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_prompt_tokens",
    "vllm:request_generation_tokens",
    "vllm:request_time_per_output_token_seconds",
    "vllm:time_to_first_token_seconds",
    "vllm:inter_token_latency_seconds",
    "vllm:time_per_output_token_seconds",
    "vllm:e2e_request_latency_seconds",
)


def parse_prometheus(text: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = SAMPLE_RE.match(line)
        if not match:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        rows.append({"metric": match.group("name"), "labels": match.group("labels") or "", "value": value})
    return rows


def scrape(endpoint: str, timeout_s: float) -> List[Dict[str, object]]:
    with urlopen(endpoint, timeout=timeout_s) as response:  # nosec B310: explicit operator endpoint
        return parse_prometheus(response.read().decode("utf-8"))


def selected(rows: Iterable[Dict[str, object]], include_all: bool) -> List[Dict[str, object]]:
    if include_all:
        return list(rows)
    return [r for r in rows if str(r["metric"]).startswith(DEFAULT_PREFIXES)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/metrics")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=0.0, help="0 captures once")
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--out", default="vllm_metrics.csv")
    parser.add_argument("--all", action="store_true", help="retain every metric")
    args = parser.parse_args()
    if args.interval <= 0 or args.duration < 0 or args.timeout <= 0:
        parser.error("interval and timeout must be positive; duration must be non-negative")

    out = Path(args.out)
    deadline = time.monotonic() + args.duration
    captures = 0
    try:
        with out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["captured_unix_s", "metric", "labels", "value"])
            writer.writeheader()
            while True:
                now = time.time()
                rows = selected(scrape(args.endpoint, args.timeout), args.all)
                for row in rows:
                    writer.writerow({"captured_unix_s": now, **row})
                handle.flush()
                captures += 1
                if args.duration == 0 or time.monotonic() >= deadline:
                    break
                time.sleep(args.interval)
    except Exception as exc:
        print(f"metrics scrape failed: {exc}", file=sys.stderr)
        try:
            out.unlink()
        except FileNotFoundError:
            pass
        raise SystemExit(2)
    print(f"captured {captures} scrape(s) to {out}")


if __name__ == "__main__":
    main()
