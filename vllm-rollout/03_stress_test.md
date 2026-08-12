# 03 — Stress Test Procedure

Based on `day-07-h100-beyond-7b` vllm_rollout_stress_test.py + FSDP profiler.

## Steps

### 1. Single model sanity (real vLLM)
```bash
pip install vllm
python code/vllm_rollout_stress_test.py --model meta-llama/Llama-3-8B --samples 50 --cot short --vllm
python code/vllm_rollout_stress_test.py --model meta-llama/Llama-3-8B --samples 50 --cot long  --vllm
```
Expected:
- short 500 tok: 40-60k tok/s, fail 5-8%, P50 wall ~0.25s, P95 ~0.6s
- long 5000 tok: 8-15k tok/s, fail 12-18%, P50 ~2.5s, P95 ~6s, queue doubling

### 2. Concurrency sweep (Little's law)
```bash
bash code/sweep.sh --model 8B --concurrencies "16,32,64,128,256" --cot short
bash code/sweep.sh --model 13B --concurrencies "16,32,64" --cot long
```
Metrics per concurrency:
- tokens/sec vs `max_num_seqs`
- P50/P95 TTFT, TPOT, queue depth (from metrics_profile.py)
- OOM count (engine returns `EngineDeadError` or empty outputs)
- Find knee: tokens/sec stops growing when GPU util >85% and TPOT rises.

Formula to pick optimal:
```
optimal_L = argmax_L ( goodput = tokens/sec * (1-fail_rate) )
goodput peaks before OOM knee. Usually L=64-128 for 7B H100.
```

### 3. Prompt length mix (prod realistic)
Mix = 20% 128 tok, 50% 1024 tok, 20% 4096 tok, 10% 16384 tok.
- Short prompts TTFT 10-20ms, long 800ms-1.5s (chunked prefill halves).
- Detect contiguity failure: vLLM allocator `allocate_slots` fails even though free blocks exist (fragmentation). Log `num_free_blocks` vs `num_freed`.

### 4. 7B/13B/30B expectations

| Model | TP | H100 Count | Short tok/s | Long tok/s | Fail short | Fail long | TTFT 4k |
|-------|----|------------|-------------|------------|------------|-----------|---------|
| 7B    | 1  | 1          | 50k         | 12k        | 6%         | 14%       | 35ms    |
| 13B   | 2  | 2          | 30k         | 7k         | 7%         | 16%       | 55ms    |
| 30B   | 4  | 4          | 18k         | 4k         | 8%         | 20%       | 90ms    |
| 70B   | 8  | 8          | 9k          | 1.8k       | 9%         | 25%       | 180ms   |

> Numbers are expected ranges before first H100 run, mark as `EST -> H100` in CSV. Don't claim as measured.

### 5. OOM / contiguity detection

- **True OOM**: `torch.cuda.OutOfMemoryError`, `KVCacheMemoryError` in logs.
- **Contiguity fake-OOM**: free blocks exist but longest contiguous run < needed (long prompt). Detect via:
```python
# vLLM engine internals (code/metrics_profile.py)
metrics["free_blocks"] = engine.cache_engine.num_free_blocks
metrics["frag"] = 1 - max_contiguous / free_blocks
if frag > 0.4 and OOM: -> "contiguity", not capacity.
```
- Stress test action: halve `max_num_batched_tokens` or enable chunked prefill.

### 6. Artifacts
- `/tmp/vllm_rollout_fail.json` per run — same schema as day-07
- `sweep_results.csv`: `model,cot,concurrency,toks/s,fail_rate,p50,p95,frag,gpu_util`

Run `python code/metrics_profile.py --endpoint http://localhost:8000/metrics` while serving to scrape live GPU util.
