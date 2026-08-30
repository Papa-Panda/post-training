# Papers and evidence ledger

> 原则：已有 `ai-data` 笔记只链接；新论文列出精确标题、年份/状态和本专题使用的 claim。数字若无法从原文核实，不进入 README headline。

## A. Foundations reused from `ai-data`

| Year | Paper | Role here | Existing note / source |
|---|---|---|---|
| 2017 | **Understanding Black-box Predictions via Influence Functions** | $H^{-1}$-corrected attribution | [note](../ai-data/day-02-2017-influence-functions/NOTES.md) · [arXiv](https://arxiv.org/abs/1703.04730) |
| 2020 | **Estimating Training Data Influence by Tracing Gradient Descent** (TracIn) | checkpoint gradient-dot-product attribution | [note](../ai-data/day-03-2020-tracin/NOTES.md) · [arXiv](https://arxiv.org/abs/2002.08484) |
| 2024 | **LESS: Selecting Influential Data for Targeted Instruction Tuning** | reusable low-rank, optimizer-aware gradient store; target selection | [note](../ai-data/day-04-2024-less/NOTES.md) · [arXiv](https://arxiv.org/abs/2402.04333) |
| 2024 | **DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models** | LoRA/Fisher-efficient influence | [note](../ai-data/day-05-2024-datainf/NOTES.md) · [arXiv](https://arxiv.org/abs/2310.00902) |
| 2023 | **The Vendi Score: A Diversity Evaluation Metric for Machine Learning** | spectral effective diversity | [note](../ai-data/day-19-2023-vendi-score/NOTES.md) · [arXiv](https://arxiv.org/abs/2210.02410) |
| 2023 | **SemDeDup: Data-efficient learning at web-scale through semantic deduplication** | embedding-level redundancy baseline | [combined note](../ai-data/day-24-2023-semdedup-d4/NOTES.md) · [arXiv](https://arxiv.org/abs/2303.09540) |
| 2023 | **D4: Improving LLM Pretraining via Document De-Duplication and Diversification** | embedding coverage baseline | [combined note](../ai-data/day-24-2023-semdedup-d4/NOTES.md) · [arXiv](https://arxiv.org/abs/2308.12284) |

Notes on dates: Vendi was first posted in 2022 and published in 2023; DataInf was first posted in 2023 and published at ICLR 2024. The table uses the commonly cited publication year when appropriate.

## B. New anchors for this专题

### TRAK (2023, ICML)

**TRAK: Attributing Model Behavior at Scale**
[arXiv:2303.14186](https://arxiv.org/abs/2303.14186) · [official code](https://github.com/MadryLab/trak)

Verified from the paper abstract:

- effective attribution methods in the comparison may require training thousands of models;
- TRAK uses **only a handful of trained models** to match them;
- demonstrated on ImageNet classifiers, CLIP, BERT and mT5.

Boundary: “handful” is the authors' wording; the abstract does not define one universal count, so this repo does not replace it with a guessed number.

### SPICE (2026, ICLR)

**SPICE: Submodular Penalized Information-Conflict Selection for Efficient Large Language Model Training**
[arXiv v2](https://arxiv.org/pdf/2601.23155v2) · [OpenReview](https://openreview.net/attachment?id=9rCRy58TPF&name=pdf) · [official code](https://github.com/Chang-pw/SPICE)

Verified from the ICLR 2026 paper and official repository:

- selects SFT subsets by combining Fisher/log-det marginal information gain with a negative-cosine penalty against the selected-set mean gradient;
- pure Fisher/log-det is monotone submodular and supports the classical greedy approximation guarantee;
- the paper's interaction bound depends on squared gradient inner products, while the implemented conflict penalty is sign-sensitive;
- supports proxy gradients and SPICE+ early stopping; the paper uses default conflict weight $\lambda=0.1$ and stopping ratio $\omega=0.5$;
- training pool is about 97.5K examples from math, code, ShareGPT and Alpaca; fixed-budget experiments select 10%;
- across 8 benchmarks, Qwen2-7B averages 58.0 versus 56.4 for full-data, while LLaMA2-7B averages 31.1 versus 30.8 for full-data;
- same-family proxy transfer is substantially more reliable than the reported LLaMA-proxy to Qwen2-7B transfer.

Boundary: SPICE measures gradient-space information coverage and selected-set optimization coherence. Its selected-set conflict is not a protected-set retention objective. The submodular/curvature guarantee for pure Fisher/log-det greedy does not fully establish the actual sign-sensitive penalized score. Table 2 labels the LLaMA2 average improvement as `+1.8`, but the displayed averages differ by $31.1-30.8=0.3$; `1.8` is the sum of the eight per-benchmark differences, not their average.

### Prismatic Synthesis / G-Vendi (2025, NeurIPS)

**Prismatic Synthesis: Gradient-based Data Diversification Boosts Generalization in LLM Reasoning**
[arXiv:2505.20161](https://arxiv.org/abs/2505.20161) · [project](https://nvlabs.github.io/prismatic-synthesis/) · [code](https://github.com/omeraj/prismatic-synthesis)

Verified from paper v2:

- empirical analysis spans **more than 300 training runs/models**, with scale and quality controlled;
- synthetic pool exceeds **3 million samples**;
- G-Vendi reaches **Spearman $\rho\approx0.9$** with OOD performance on both NLI and math reasoning;
- gradient sketch uses Rademacher random projection with $d=1024$ in experiments;
- proxy is not the final student nor the 32B/72B generator; main setup uses **Qwen2.5-0.5B-Instruct** with off-the-shelf weights and no extra warm-up/fine-tuning; per-sample full-parameter normalized NLL gradient is projected to 1024 dims for G-Vendi and clustering;
- ablated proxy correlation with OOD: Llama-3.2-1B-Instruct $\rho=0.909$, Qwen2.5-0.5B-Instruct $\rho=0.898$, Qwen2.5-0.5B base $\rho=0.772$, indicating instruction-tuned proxy is substantially better than base while family effect is modest;
- process: gradient-space clustering → few-shot generation → keep sparse-cluster samples;
- final datasets: **1.0M** Nemotron-PrismMath pairs and **515K** PrismNLI pairs;
- PrismMath-7B result is better than R1-Distill-Qwen-7B on **6 of 7** listed benchmarks; PrismNLI improves average OOD accuracy by **8 percentage points** over the best prior mixture in the paper.

Boundary: $\rho\approx0.9$ is a controlled rank-correlation result, not a universal law or causal guarantee.

### GrADS (2025 preprint)

**Learn More, Forget Less: A Gradient-Aware Data Selection Approach for LLM**
[arXiv:2511.08620](https://arxiv.org/abs/2511.08620)

- preliminary one-epoch SFT extracts embedding-layer and LM-head gradients;
- self-adaptive criteria use gradient magnitude/statistical distribution;
- abstract reports 5% selected data surpassing full-data fine-tuning in its tested settings while mitigating forgetting; the detailed robustness section more cautiously says 2.5%–5% is comparable to full-data performance in most cases.

Boundary: preprint claim; the repository treats 5% as an experimental result, not a default budget. The paper does not fully specify the scalar/tensor reduction before combining embedding and LM-head gradients, so this专题 does not invent that missing implementation detail.

### OGS (2026 preprint)

**Training Data Selection with Gradient Orthogonality for Efficient Domain Adaptation**
[arXiv:2602.06359](https://arxiv.org/abs/2602.06359)

- uses a general-knowledge gradient anchor and selects candidates with safer geometry;
- Navigator–Target architecture estimates geometry on a smaller proxy;
- moves gradient-surgery intuition to data selection so target training uses a standard optimizer.

Boundary: recent preprint; proxy-to-target geometry transfer needs independent validation.

### GradAlign (2026 preprint)

**GradAlign: Gradient-Aligned Data Selection for LLM Reinforcement Learning**
[arXiv:2602.21492](https://arxiv.org/abs/2602.21492v2) · [official code](https://github.com/StigLidu/GradAlign)

- scores candidate RL problems by cosine alignment between candidate policy gradients and an aggregated trusted-validation gradient;
- periodically rescoring under the current policy creates an adaptive curriculum;
- evaluated under unreliable reward, distribution imbalance and low-utility corpus regimes.

Boundary: recent preprint aimed at non-stationary LLM RL; do not conflate with the unrelated 2022 large-batch/federated method also named GradAlign.

## C. Claim status

| Claim | Status |
|---|---|
| TRAK reduces “thousands of models” to “a handful” | verified, exact original wording; no fixed count asserted |
| Prismatic uses 300+ training runs/models | verified |
| G-Vendi–OOD Spearman $\rho\approx0.9$ on NLI and math | verified |
| Gradient proxy is Qwen2.5-0.5B-Instruct (no warm-up) with full-param NLL gradient → 1024-dim Rademacher projection; proxy ablation $\rho=0.909$ / $0.898$ / $0.772$ | verified |
| Prismatic sparse-gradient-cluster loop | verified |
| SPICE uses 10% of its roughly 97.5K pool in fixed-budget experiments | verified |
| Pure Fisher/log-det is sign-blind, while SPICE's practical conflict penalty is sign-sensitive | verified from equations |
| Pure Fisher greedy's approximation guarantee automatically applies to the penalized SPICE score | **not claimed** |
| SPICE selected-set conflict guarantees protected-capability retention | **not claimed** |
| G-Vendi guarantees OOD improvement on arbitrary tasks | **not claimed** |
| GrADS universal best subset size is 5% | **not claimed** |
| OGS/GradAlign are mature production standards | **not claimed** |

<!-- NAVIGATION -->
## 导航

- 上一篇：[09 SPICE 协调性](09_spice_information_conflict.md)
- 下一篇：[目录](README.md)
- 回到：[目录 README](README.md) | [论文证据](papers.md) | [路线图](README.md#路线图)

> 串联：01 统一框架 → 02 归因/目标化 → 03 覆盖 → 04 生成 → 05 安全 → 06 系统 → 07 Coding 落地 → 08 边界 → 09 SPICE → 论文证据

