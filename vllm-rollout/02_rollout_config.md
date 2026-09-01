# 02 — Continuous batching and KV-cache capacity

## 1. Prefill and decode are different resources

Prefill processes many prompt tokens in parallel and is commonly compute-intensive. Autoregressive decode advances each active sequence by one or a small number of tokens per scheduler iteration and repeatedly reads weights plus growing KV state. Mixing them creates a latency/throughput trade-off.

vLLM V1 enables chunked prefill by default whenever possible. Its documented scheduler prioritizes pending decode work, then uses the remaining `max_num_batched_tokens` budget for prefills, chunking a prefill that does not fit. The official tuning guide therefore treats this budget as a knob: smaller values can favor ITL, while larger values can improve TTFT and throughput for some workloads. Those are directions, not universal settings; benchmark the installed release.

Continuous batching means the batch membership can change between engine steps. A completed sequence leaves and a waiting sequence can enter without waiting for the entire original batch. It does **not** remove queueing, KV limits, or head-of-line effects.

## 2. Logical KV-cache memory

For a decoder with $L$ layers, $H_{kv}$ key/value heads, head dimension $d_h$, and $b$ bytes per stored scalar, the logical KV bytes per sequence token are:

$$m_{KV/token}=2LH_{kv}d_hb$$

The factor 2 is for key and value. For full multi-head attention, $H_{kv}$ equals the number of query heads. Grouped-query or multi-query attention lowers $H_{kv}$ and therefore KV bytes per token.

For sequence length $S$ and $B$ simultaneously resident sequences, a first-order logical estimate is:

$$M_{KV}=BSm_{KV/token}$$

This omits block rounding, cache sharing, padding, temporary workspaces, graph capture, allocator overhead, multimodal state, and backend-specific layout. Tensor or pipeline parallelism can change per-rank storage. Treat the formula as a dimensional check, then compare with engine-reported cache blocks and observed memory.

The calculator makes the residual-memory assumption explicit:

```bash
python3 code/rollout_lab.py kv \
  --layers 32 --kv-heads 8 --head-dim 128 --dtype-bytes 2 \
  --total-memory-gib 80 --gpu-memory-utilization 0.9 \
  --weights-gib 16 --non-kv-gib 8
```

Its budget is:

$$M_{KV,budget}=uM_{device}-M_{weights}-M_{nonKV}$$

Do not mistake the result for vLLM's actual profiled cache capacity. Startup logs and metrics from the pinned engine are authoritative for the run.

## 3. What PagedAttention changes

The PagedAttention paper stores a sequence's KV state in fixed-size, non-contiguous physical blocks addressed through a block table. This avoids requiring one contiguous region sized for the request's maximum possible length and allows sharing in supported decoding patterns.

Consequences:

- External holes are not diagnosed with a made-up “largest contiguous KV span” metric.
- The partially filled final block creates bounded internal waste per sequence; block metadata and implementation details still cost memory.
- “Free blocks exist but allocation failed” does not by itself prove fragmentation. Inspect engine logs, block demand, and version-specific metrics.
- KV pressure can trigger request preemption. Current V1 documentation says recomputation is the default preemption mode; preemption count and latency are therefore paired signals.

## 4. Prefix caching

Automatic prefix caching reuses KV blocks for identical reusable prompt prefixes. Official design documentation describes block hashes, full-block caching, reference counts, and eviction of unreferenced blocks under pressure.

What it can improve:

- repeated system or few-shot prefixes;
- repeated long document prefixes;
- repeated multi-turn history when tokenization and cache identity match.

What it does not improve:

- generation of new output tokens;
- prompts with little token-identical shared prefix;
- policy freshness after weights change unless cache lifecycle and model identity are handled correctly.

Measure hit **tokens / queried tokens**, not merely hit requests. Record cache salt, adapter identity, model revision, and engine version when relevant.

## 5. Config knobs as hypotheses

| Knob | Direct constraint | Likely trade-off to test |
|---|---|---|
| `max_num_seqs` | concurrent sequences in a scheduler batch | more decode concurrency versus more KV pressure and graph sizes |
| `max_num_batched_tokens` | scheduler token budget per iteration | prefill progress/throughput versus decode ITL |
| `gpu_memory_utilization` | fraction used when sizing engine memory | more KV room versus less headroom for other allocations |
| `max_model_len` | maximum supported sequence length | feasibility guard; can affect cache sizing and startup validation |
| tensor/pipeline/data parallel sizes | model placement and replicas | per-rank memory, communication, latency, and aggregate capacity |
| KV dtype | bytes and numerical representation of cache | capacity versus accuracy/calibration requirements |
| prefix caching | reuse of compatible full prefix blocks | lower repeated prefill work versus cache management/security policy |
| eager/compilation controls | graph capture and compiled execution | runtime speed versus startup time, graph memory, and compatibility |

Never encode a parameter-count-to-GPU-count table as fact. Measure weight memory and non-KV reserve, calculate a safe KV envelope, then run the stress matrix.
