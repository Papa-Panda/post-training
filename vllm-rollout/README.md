# vLLM rollout infrastructure

A measurement-first guide to the inference side of reinforcement-learning post-training. It connects a rollout request's lifecycle to queue stability, token economics, KV-cache capacity, policy staleness, failure handling, and observability. It deliberately contains **no hardware throughput promises**: values produced by the CPU lab are synthetic inputs, while real capacity must come from a version-pinned benchmark on the target model, engine, and hardware.

## Learning path

1. [Queueing, latency, and token accounting](01_metrics.md)
2. [Continuous batching and KV-cache capacity](02_rollout_config.md)
3. [Stress-test design and configuration](03_stress_test.md)
4. [Failure taxonomy and response](04_failure_taxonomy.md)
5. [Policy staleness: systems choices change the objective](05_policy_staleness.md)
6. [Observability and evidence](06_observability.md)
7. [Primary and official sources](SOURCES.md)

Adjacent material: [PPO and GRPO objectives](../grpo-vs-ppo/README.md) and [distributed-training fundamentals](../ai-infra/README.md).

## The end-to-end control loop

```text
prompt source -> admission queue -> prefill -> iterative decode -> tool/environment
      ^                                                        |
      |                                                        v
policy snapshot <- learner <- scoring/filtering <- rollout record + behavior logprobs
```

The key separation is between **work offered**, **work executed**, and **work useful to learning**:

- Offered load: requests and requested tokens arriving per second.
- Executed work: prompt tokens processed, recomputed prompt tokens, and all generated tokens.
- Delivered work: completed trajectories.
- Useful work: trajectories accepted by syntax, environment, verifier, freshness, and data-policy gates.

Optimizing only generated tokens/s can increase cost if it also increases retries, stale-policy drops, or unusable trajectories.

## CPU quick start

No third-party package is required.

```bash
cd vllm-rollout
python3 code/rollout_lab.py kv \
  --layers 32 --kv-heads 8 --head-dim 128 --dtype-bytes 2 \
  --total-memory-gib 80 --gpu-memory-utilization 0.9 \
  --weights-gib 16 --non-kv-gib 8

python3 code/rollout_lab.py simulate --config configs/stable.json
python3 code/rollout_lab.py simulate --config configs/overload.json

# Sweep offered load; writes JSON Lines.
bash code/sweep.sh configs/stable.json /tmp/rollout-sweep.jsonl

# Tests
python3 -m unittest discover -s tests -v
```

`rollout_lab.py` is intentionally a queue/scheduler abstraction. It does not predict a GPU's tokens/s. Calibrate `prefill_tokens_per_s`, `decode_tokens_per_s`, scheduler budget, and KV capacity from a real benchmark, then use it for sensitivity analysis and invariants.

## Real-system experiment loop

1. Pin the vLLM release or commit, model revision, tokenizer revision, dtype, quantization, kernels, driver, and hardware topology.
2. Record a workload manifest: arrival process, prompt/output distributions, sampling multiplicity, stop conditions, and shared-prefix fraction.
3. Establish a low-load latency baseline.
4. Sweep one control at a time: offered load, `max_num_seqs`, `max_num_batched_tokens`, memory allocation, and parallelism.
5. Save server config, client results, Prometheus scrape, engine logs, and policy versions in the same run directory.
6. Select a configuration by useful-goodput subject to latency, failure, memory, and policy-lag constraints—not by peak throughput alone.

See [03_stress_test.md](03_stress_test.md) for the run matrix and [06_observability.md](06_observability.md) for the minimum evidence bundle.

## What this topic does not claim

- A fixed model-to-GPU mapping. Weight fit, KV capacity, kernels, parallelism, and runtime reserves all matter.
- Universal values for `max_num_seqs`, `max_num_batched_tokens`, or `gpu_memory_utilization`.
- That an application timeout, malformed tool call, verifier rejection, and engine failure are the same event.
- That PPO clipping automatically makes arbitrarily stale rollouts safe.
- That prefix-cache hits accelerate decode; they avoid repeated prompt computation for reusable full blocks.
