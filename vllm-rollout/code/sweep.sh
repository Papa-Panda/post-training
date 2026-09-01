#!/usr/bin/env bash
# CPU-only arrival-rate sweep. Run from vllm-rollout/ or its parent.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$HERE/../configs/stable.json}"
OUT="${2:-/tmp/vllm-rollout-sweep.jsonl}"
: > "$OUT"
for rate in 1 4 8 16 24 32 40 48; do
  python3 "$HERE/rollout_lab.py" simulate --config "$CONFIG" --arrival-rate-rps "$rate" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps({"arrival_rate_rps":d["config"]["arrival_rate_rps"],"completed":d["counts"]["completed"],"timed_out":d["counts"]["timed_out"],"queue_p95":d["latency_s"]["queue_p95"],"ttft_p95":d["latency_s"]["ttft_p95"],"useful_goodput_tps":d["rates"]["useful_goodput_tps"],"preemptions":d["counts"]["preemptions"]}))' \
    >> "$OUT"
done
cat "$OUT"
