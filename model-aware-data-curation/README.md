# Model-aware Data Curation

> **基于梯度的数据价值、覆盖与生成**：不是再做一套逐篇 paper notes，而是把归因、目标化选择、多样性覆盖、主动生成与持续学习接成一个 model-in-the-loop 控制闭环。

## 一句话定位

`ai-data/` 回答 **“原料是什么、怎么洗、怎么构造”**；本专题回答 **“当前模型缺什么、哪条数据最能推动目标、下一批应该生成什么”**。

```text
candidate pool / generator
        │
        ▼
gradient probe ──► value / target alignment / coverage / conflict
        │                           │
        ▼                           ▼
select or generate ──► train ──► eval ──► refresh gradient map
        ▲                                      │
        └──────────────────────────────────────┘
```

## 七条线，一套统一几何

设候选样本为 $z=(x,y)$，当前代理模型参数为 $\theta$，目标验证集为 $V$：

$$g_z=-\nabla_\theta\log p_\theta(y\mid x),\qquad \bar g_V=\frac1{|V|}\sum_{v\in V}g_v.$$

| 线 | 核心问题 | 几何对象 | 代表方法 |
|---|---|---|---|
| 归因基础 | 谁导致了这个行为？ | $g_z^\top H^{-1}g_v$ 或轨迹/投影近似 | Influence Functions, TracIn, TRAK |
| 目标化选择 | 哪些样本推动目标能力？ | $\cos(\tilde g_z,\tilde g_V)$ | LESS, DataInf, GradAlign |
| 多样性覆盖 | 当前梯度空间还缺哪些方向？ | 梯度核谱熵或 Fisher/log-det | Vendi, D4/SemDeDup, G-Vendi, FisherSFT |
| 集合协调 | 候选会不会抵消已选集合的平均更新？ | candidate 与 selected-set mean 的负 cosine | SPICE |
| 主动生成 | 如何补稀疏区域而非继续堆重复样本？ | 稀疏梯度簇 + rejection sampling | Prismatic Synthesis |
| 安全/持续学习 | 新能力如何不破坏明确要保护的能力？ | candidate 与 protected gradient 的冲突 | GrADS, OGS |
| 可学性门 | 孤立方向是稀缺能力，还是当前模型学不进去？ | 跨 checkpoint 相似度、reward trend 与训练响应 | RLVR unlearnability analysis |

## 阅读路线：数学 → 系统 → 可运行代码

1. [`01_problem_formulation.md`](01_problem_formulation.md) — 把“价值/覆盖/冲突”写成统一多目标优化。
2. [`02_attribution_to_targeting.md`](02_attribution_to_targeting.md) — 从 IF/TracIn/TRAK 到 LESS/DataInf/GradAlign；已在 `ai-data` 的论文只链接、不复述。
3. [`03_gradient_coverage.md`](03_gradient_coverage.md) — 从 embedding diversity 到 G-Vendi 的模型感知覆盖。
4. [`04_prismatic_synthesis.md`](04_prismatic_synthesis.md) — Prismatic 三步循环与已核实实验数字。
5. [`05_safety_continual_learning.md`](05_safety_continual_learning.md) — GrADS/OGS 与遗忘、冲突、动态课程。
6. [`06_system_architecture.md`](06_system_architecture.md) — gradient datastore、近似计算、调度与可观测性。
7. [`07_coding_data_flywheel.md`](07_coding_data_flywheel.md) — coding-data 闭环：失败簇 → 生成 → 验证 → 训练 → 回归。
8. [`08_ai_data_boundary.md`](08_ai_data_boundary.md) — 与 `ai-data/` 的边界、复用规则和去重清单。
9. [`09_spice_information_conflict.md`](09_spice_information_conflict.md) — SPICE 的 Fisher/log-det、conflict-aware greedy、理论边界，以及 diversity / conflict / retention / unlearnability 四分法。
10. [`papers.md`](papers.md) — 论文索引、年份、状态与 claim 证据。
11. [`code/`](code/) + [`tests/`](tests/) — NumPy 最小 demo：target alignment、G-Vendi、Fisher/log-det、SPICE、稀疏簇选择与安全约束。

```bash
python3 model-aware-data-curation/code/demo.py
python3 -m unittest discover -s model-aware-data-curation/tests -v
```

## 路线图

> 目标：把“选数据”从静态过滤，升级为 **model-in-the-loop 的控制系统**。按下面三阶段走，每阶段都有可验证产出。

### Phase 1 — 统一框架（Day 1–2）

- 读 `01` 建立 Value × Coverage × Safety 的统一优化式，理解 **选有用的、补缺的、不添乱的**；
- 读 `02` 掌握归因到目标化选择的演进：Influence → TracIn/TRAK → LESS/DataInf → GradAlign；
- 产出：能说清为什么不能把三个目标压成一个 cosine score。

### Phase 2 — 覆盖与生成（Day 3–5）

- 读 `03`：embedding diversity → G-Vendi，proxy 选择（Qwen2.5-0.5B-Instruct 不做 warm-up，1024维投影，$\rho=0.909$/$0.898$/$0.772$）；
- 读 `04`：Prismatic 三步循环，稀疏簇生成与质量门；
- 读 `09`：SPICE 的 Fisher/log-det coverage + selected-set conflict，以及四种梯度几何的边界；
- 动手：跑 `code/demo.py` 看 G-Vendi / SPICE / 隔离度计算。

### Phase 3 — 系统与落地（Day 6–10）

- 读 `06`：proxy 选择五原则（白盒可微、loss 匹配、LoRA/head、家族一致性、ranking 相关性验证）与 refresh 策略；
- 读 `05` + `07`：安全/持续学习约束，coding-data flywheel 的失败簇 → 生成 → 执行验证 → 训练 → 回归；
- 读 `08`：与 `ai-data/` 的边界，避免重复造笔记；
- 产出：一个带质量门、conflict gate、G-Vendi 监控和 shadow mode 的最小 curation job。

```text
01 统一框架 ──► 02 归因与目标化 ──► 03 覆盖 (G-Vendi/Fisher)
      │                   │                      │
      ▼                   ▼                      ▼
06 系统架构 ◄── 09 协调与冲突 ◄── 04 主动生成 (Prismatic)
      │                   │
      ▼                   ▼
05 安全/持续学习 ──► 07 coding flywheel
```

完成后，你会得到：**一套可复用的 gradient datastore + 三维 dashboard（correctness × target alignment × coverage）+ 可回放的版本化数据版本**。

## 核心判断

Prismatic Synthesis 把原本分开的三段串成闭环：

$$\underbrace{\text{LESS-like target signal}}_{\text{想学什么}} +\underbrace{\text{G-Vendi coverage}}_{\text{还缺什么}} +\underbrace{\text{synthetic generation}}_{\text{主动补什么}}.$$

但它仍是前沿而非成熟范式：

- G-Vendi 测的是**给定代理模型下**的梯度方向覆盖，不自动保证 correctness、safety 或无污染；
- Prismatic 的 $\rho\approx0.9$ 来自论文控制实验中的 Spearman 相关，不应外推成跨任务定律；
- SPICE 的 Fisher/log-det 衡量 coverage，selected-set negative cosine 衡量 optimization coherence；两者不是同一个几何量；
- diversity、selected-set conflict、protected-set retention 与 unlearnability 分别回答“缺不缺”“抵不抵消”“伤不伤旧能力”“学不学得进去”，不能混成一个 cosine score；
- GradAlign、GrADS、OGS 的适用场景不同，不能把“高价值”“高覆盖”“低冲突”压成一个无条件总分；
- 生产系统应保留质量门、执行验证、去污染和周期性全量评估。

## Verified headline numbers

- **TRAK (ICML 2023)**：原论文措辞是用 *a handful of trained models* 匹配需要 *thousands of models* 的 attribution 方法；没有把 “handful” 固定成一个通用数字。
- **Prismatic Synthesis (NeurIPS 2025)**：分析覆盖 **over 300 training runs**；G-Vendi 与 OOD 表现的 **Spearman $\rho\approx0.9$** 同时报告于 NLI 和数学推理；论文还构造了超过 300 万条样本的合成池。
- **Prismatic 生成闭环**：梯度空间聚类 → few-shot 生成 → 只接收稀疏簇样本；论文实例使用最稀疏的簇（示例为 top 20%，具体实现为保留最小的 $k/2$ 个簇）。
- **SPICE (ICLR 2026)**：从约 97.5K 条训练池选择 10%；Qwen2-7B 平均 58.0、full-data 56.4，LLaMA2-7B 平均 31.1、full-data 30.8。经典 submodular/curvature guarantee 直接对应纯 Fisher/log-det greedy，不完整覆盖加入 sign-sensitive conflict penalty 后的实际 score。

所有数字与出处见 [`papers.md`](papers.md)。
