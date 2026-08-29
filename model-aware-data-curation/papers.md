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

### Prismatic Synthesis / G-Vendi (2025, NeurIPS)

**Prismatic Synthesis: Gradient-based Data Diversification Boosts Generalization in LLM Reasoning**
[arXiv:2505.20161](https://arxiv.org/abs/2505.20161) · [project](https://nvlabs.github.io/prismatic-synthesis/) · [code](https://github.com/omeraj/prismatic-synthesis)

Verified from paper v2:

- empirical analysis spans **more than 300 training runs/models**, with scale and quality controlled;
- synthetic pool exceeds **3 million samples**;
- G-Vendi reaches **Spearman $\rho\approx0.9$** with OOD performance on both NLI and math reasoning;
- gradient sketch uses Rademacher random projection with $d=1024$ in experiments;
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
| Prismatic sparse-gradient-cluster loop | verified |
| G-Vendi guarantees OOD improvement on arbitrary tasks | **not claimed** |
| GrADS universal best subset size is 5% | **not claimed** |
| OGS/GradAlign are mature production standards | **not claimed** |
