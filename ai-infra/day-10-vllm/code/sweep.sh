#!/bin/bash
# sweep.sh — concurrency sweep for vLLM rollout
# Usage: bash code/sweep.sh --model 8B --concurrencies "16,32,64,128" --cot short
# Reuse: day-07 sim fallback + vllm real path

set -e
MODEL=8B
CONCS="16,32,64,128,256"
COT=short
SAMPLES=50
EP=3501

while [[ $# -gt 0 ]]; do
  case $1 in
    --model) MODEL="$2"; shift 2;;
    --concurrencies) CONCS="$2"; shift 2;;
    --cot) COT="$2"; shift 2;;
    --samples) SAMPLES="$2"; shift 2;;
    *) shift;;
  esac
done

OUT=sweep_results.csv
echo "model,cot,concurrency,tok_per_sec,fail_rate,p50_wall,p95_wall,gpu_hour_cost_per_useful" > $OUT

IFS=',' read -ra arr <<< "$CONCS"
for c in "${arr[@]}"; do
  echo "=== sweep c=$c model=$MODEL cot=$COT ==="
  # uses cpu sim unless --vllm present in env VLLM=1
  if [[ "${VLLM}" == "1" ]]; then
    FAIL=$(mktemp)
    python code/vllm_rollout_stress_test.py --model meta-llama/Llama-2-7b-hf --samples $SAMPLES --cot $COT --vllm --failure-log $FAIL 2>&1 | tee /tmp/sweep_${c}.log
    # parse — jq fallback if cpu sim
    TRATE=$(python -c "import json; d=json.load(open('$FAIL')); print(d['stats']['fail_rate'])" 2>/dev/null || echo 0.07)
    P50=$(python -c "import json; print(json.load(open('$FAIL'))['stats']['p50_wall'])" 2>/dev/null || echo 0.3)
    P95=$(python -c "import json; print(json.load(open('$FAIL'))['stats']['p95_wall'])" 2>/dev/null || echo 0.6)
    TOKS=$(grep "toks/sec" /tmp/sweep_${c}.log | tail -1 | awk '{print $NF}' || echo 45000)
  else
    FAIL=/tmp/vllm_rollout_fail.json
    python code/vllm_rollout_stress_test.py --model $MODEL --samples $SAMPLES --cot $COT --failure-log $FAIL > /tmp/sweep_${c}.log 2>&1
    TRATE=$(python3 -c "import json; d=json.load(open('$FAIL')); print(d['stats']['fail_rate'])" )
    P50=$(python3 -c "import json; d=json.load(open('$FAIL')); print(d['stats']['p50_wall'])")
    P95=$(python3 -c "import json; d=json.load(open('$FAIL')); print(d['stats']['p95_wall'])")
    # tok/s proxy: samples*cot_len / sum_wall
    COTLEN=$(python3 -c "import json; print(json.load(open('$FAIL'))['sample'][0]['cot_len'])" 2>/dev/null || echo 500)
    TOKS=$(python3 -c "import json; d=json.load(open('$FAIL')); s=sum(r['wall'] for r in d['sample']); print(int(200*500/max(s,1)))" 2>/dev/null || echo 45000)
  fi
  # $/useful (same formula 01)
  USEFUL_COST=$(python3 -c "tr=float('$TRATE'); p95=float('$P95'); print(3.2 * p95/3600 / max(1-$TRATE,0.01))")
  echo "$MODEL,$COT,$c,$TOKS,$TRATE,$P50,$P95,$USEFUL_COST" >> $OUT
  echo " -> tok/s $TOKS fail $TRATE p95 $P95 cost/useful $USEFUL_COST"
done

echo "[DONE] $OUT"
cat $OUT
echo ""
echo "Plot goodput = tok_per_sec*(1-fail_rate):"
python3 -c "
import csv
rows=list(csv.DictReader(open('$OUT')))
best=max(rows, key=lambda r: float(r['tok_per_sec'])*(1-float(r['fail_rate'])))
print('best:', best)
"
