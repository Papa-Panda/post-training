# Deprecated estimate note

This file is retained only so old links do not break.

The former contents combined synthetic evaluation changes, placeholder prices, projected throughput/failure rates, and incomplete memory accounting. None of those values were hardware measurements, and several were not reproducibly derived. They were removed during the repository audit.

Use [`README.md`](README.md) for the current model-state capacity method and [`fsdp_h100_profiler_beyond7b.py`](fsdp_h100_profiler_beyond7b.py) for explicit byte accounting. Use [`../day-10-vllm/`](../day-10-vllm/) or [`../../vllm-rollout/`](../../vllm-rollout/) for rollout work.
