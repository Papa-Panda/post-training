# NOTES — r2-Day03 Topology and Collectives

## Corrected mental model

- Separate logical payload, per-rank algorithm traffic, link-rate units, and measured effective bandwidth.
- For ring all-reduce, per-rank sent bytes are $2(p-1)S/p$ and the ring has $2(p-1)$ modeled steps.
- Reduce-scatter and all-gather each contribute half of that volume for the same logical payload.
- `400 Gb/s = 50 GB/s` is only a decimal line-rate conversion; protocol and system overhead reduce payload throughput.
- A published total/bidirectional NVLink number is not automatically the one-way effective bandwidth used in the ring formula.
- `nvidia-smi topo -m` is a local-system matrix. `NODE` means a PCIe path across host bridges within one NUMA node, not a remote machine.

## Example with explicit assumptions

For $p=8$, $S=1$ decimal GB, $B_{effective}=450$ GB/s one-way, and $\alpha=2\ \mu s$ per step:

$$V=2\frac{7}{8}(1\ \mathrm{GB})=1.75\ \mathrm{GB}.$$

$$T\approx14(2\ \mu s)+\frac{1.75\ \mathrm{GB}}{450\ \mathrm{GB/s}}=3.917\ \mathrm{ms}.$$

The 450 GB/s value is an illustrative assumption chosen to avoid treating a 900 GB/s bidirectional headline as one-way payload bandwidth. It is not a benchmark result.

## Validation record

```bash
python3 ai-infra/r2-day-03-topo-nccl/topo_demo.py
python3 -m unittest discover -s ai-infra/r2-day-03-topo-nccl -p 'test_*.py' -v
```

CPU tests cover formulas and semantics only. Hardware follow-up must record:

- GPU/NIC topology and affinity;
- NCCL/CUDA/driver versions;
- message-size sweep, warm-up, iterations, and synchronization;
- algorithm bandwidth versus bus-bandwidth reporting;
- overlap measured inside the actual training step.

No H100, NCCL, NVLink, PCIe, or InfiniBand performance is claimed by this lesson.
