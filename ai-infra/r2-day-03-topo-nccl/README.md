# r2-Day03 — Topology and Ring Collectives

## Connection to Prev

r2-Day02 established a single-process training loop. Scaling it requires two separate models:

1. the **logical collective**: which tensor is reduced or gathered and how many bytes each rank sends;
2. the **physical path**: NVLink/NVSwitch, PCIe, or a NIC/fabric, including topology and contention.

Conflating a product's aggregate link headline with measured collective bandwidth is the main trap in back-of-the-envelope estimates.

## 1. Collective semantics

For $p$ ranks and a logical payload of $S$ bytes per rank, a ring partitions the payload into $p$ chunks.

- Reduce-scatter uses $p-1$ steps and sends $(p-1)S/p$ bytes per rank.
- All-gather uses $p-1$ steps and sends $(p-1)S/p$ bytes per rank.
- Ring all-reduce composes the two, using $2(p-1)$ steps and sending $2(p-1)S/p$ bytes per rank.

$$V_{\mathrm{AR}}=2\frac{p-1}{p}S.$$

This is **per-rank bytes sent by the algorithm**. It is not the sum over all ranks, and it is not application-level tensor bytes alone.

A minimal latency-bandwidth model is:

$$T\approx n_{\mathrm{steps}}\alpha+\frac{V}{B_{\mathrm{effective}}}.$$

- $\alpha$ is startup latency per modeled step;
- $B_{\mathrm{effective}}$ is effective one-way payload bandwidth under a stated assumption or measurement;
- protocol overhead, channels, chunking, contention, routing, and overlap are omitted.

For 8 ranks and a 1 GB payload, ring all-reduce sends 1.75 GB per rank in 14 steps. If one explicitly assumes 450 GB/s effective one-way bandwidth and $2\ \mu s$ startup per step, the model gives $3.917$ ms. That is an estimate, not an H100 or NCCL measurement.

## 2. Units before numbers

- `400 Gb/s` is a bit rate: the decimal line-rate conversion is `50 GB/s` before encoding and protocol overhead.
- `GB/s` and `GiB/s` differ by a factor of $10^9/2^{30}$.
- A vendor's “total” or bidirectional NVLink number must not be inserted as one-way effective ring bandwidth without explaining the conversion.
- NCCL's algorithm bandwidth and bus bandwidth are reporting conventions; compare like with like.

## 3. Reading `nvidia-smi topo -m`

The matrix describes paths among GPUs and NICs **inside one system**.

| Label | Meaning |
|---|---|
| `X` | same device |
| `PIX` | at most one PCIe switch |
| `PXB` | multiple PCIe switches, no PCIe host bridge |
| `PHB` | through a PCIe host bridge |
| `NODE` | through PCIe host bridges within one NUMA node |
| `SYS` | through PCIe and the interconnect between NUMA nodes |
| `NV#` | bonded set of `#` NVLinks |

`NODE` does **not** mean “another machine” and does not identify InfiniBand or RoCE. Inter-host analysis needs the NIC placement, fabric, routing, and benchmark output in addition to the local matrix.

## 4. Why tensor parallelism is topology-sensitive

Tensor parallelism commonly places collectives inside each transformer layer, so startup and transfer costs repeatedly enter the forward/backward critical path. Pipeline parallelism communicates activations at stage boundaries; data parallelism reduces gradient buckets during backward. Those differences explain why tensor-parallel groups are often kept inside a fast local domain, but “TP can never cross a host” is too absolute: feasibility depends on message size, compute/communication overlap, topology, model shape, and the performance target.

The correct workflow is:

1. derive payload and collective frequency from the actual partitioning;
2. inspect GPU–GPU and GPU–NIC locality;
3. benchmark the relevant message-size range;
4. profile overlap and exposed communication on the real training step.

## 5. Run the analytical model

```bash
python3 ai-infra/r2-day-03-topo-nccl/topo_demo.py
python3 -m unittest discover -s ai-infra/r2-day-03-topo-nccl -p 'test_*.py' -v
```

The tests check unit conversion, ring volume/step counts, reduce-scatter + all-gather composition, latency accounting, and the important `NODE` semantic.

For hardware validation, record the machine and software configuration, then run the local topology command and an NCCL benchmark over representative sizes. Do not copy a peak product number into the results table.

## Status

- Verified here: formulas, units, topology-label semantics, and seven CPU tests.
- Not verified here: NCCL algorithm choice, effective bandwidth, latency, overlap, or any H100 timing.

## Primary sources

- NCCL collective semantics: https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2183/user-guide/docs/usage/collectives.html
- `nvidia-smi topo` command and legend: https://man.archlinux.org/man/nvidia-utils/nvidia-smi.1.en
- PyTorch DDP design note (bucketed reductions and overlap): https://docs.pytorch.org/docs/main/notes/ddp
- Ring all-reduce paper: https://doi.org/10.1016/j.jpdc.2009.05.002
