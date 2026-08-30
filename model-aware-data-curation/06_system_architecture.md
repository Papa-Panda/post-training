# 06 — 系统架构：Gradient Datastore + Curation Control Plane

## 元信息
- 内容类型：跨论文系统设计，不对应单篇论文
- 方法来源：[LESS](https://arxiv.org/abs/2402.04333) · [TRAK](https://arxiv.org/abs/2303.14186) · [Prismatic Synthesis](https://arxiv.org/abs/2505.20161) · [GradAlign](https://arxiv.org/abs/2602.21492v2)
- 本章定位：将论文中的 gradient sketch、attribution、coverage 和动态重算拼成可部署 control plane。


## 1. 数据面与控制面分离

```text
                        ┌────────────────────────────┐
raw / synthetic pool ──► quality + execution gates │
                        └──────────────┬─────────────┘
                                       ▼
┌──────────────┐   per-sample grads   ┌───────────────────┐
│ proxy models │─────────────────────►│ gradient datastore│
└──────────────┘                      └─────────┬─────────┘
                                                │
         ┌──────────────────────────────────────┼──────────────────────┐
         ▼                                      ▼                      ▼
 target alignment                         coverage/clusters       conflict/retention
         └──────────────────────────────────────┬──────────────────────┘
                                                ▼
                                      selector / generator policy
                                                ▼
                                      train -> eval -> refresh
```

- **数据面**：样本、provenance、执行结果、梯度向量、簇标签、版本；
- **控制面**：目标权重、预算、阈值、refresh 条件、生成策略；
- **训练面**：SFT/RL job 与 checkpoint；
- **评估面**：ID/OOD、retention、污染、成本与 rank stability。

## 2. Gradient record schema

```json
{
  "sample_id": "stable-content-hash",
  "dataset_version": "v17",
  "proxy_id": "proxy-family-size-checkpoint",
  "objective": "sft_nll|policy_gradient",
  "layer_scope": "lora|lm_head|selected_blocks",
  "projection_seed": 17,
  "projection_dim": 1024,
  "gradient_uri": "shard://...",
  "norm": 3.14,
  "quality_pass": true,
  "execution_pass": true,
  "contamination_pass": true,
  "created_at": "..."
}
```

必须把 `proxy_id/objective/layer_scope/projection_seed` 放进 key；不同协议的向量不可静默混用。

## 3. 计算路径与 Proxy 选择

核心不是模型必须多大，而是两点：**梯度拿得到**，而且**梯度几何能代表目标模型**。

1. 必须是白盒、可微模型，能计算每条样本的 loss/log-prob 对参数的梯度；只有闭源 API 输出通常不够。
2. loss 要与目标匹配：SFT 用 answer NLL；RL 用当前 policy 的 log-prob × advantage；保护能力用 retention set loss。
3. 不必计算全参数梯度，通常只取 LoRA、LM head 或若干层，再随机投影和归一化。
4. proxy 不必和最终模型一样大，但最好同 tokenizer、同模型家族、同训练目标，而且至少已经具备基本任务能力；太弱的模型只会产生“我什么都不会”的噪声梯度。
5. 必须抽样验证 proxy 与目标模型的数据排名相关性，否则小模型选出的数据未必对大模型有效。

方法之间要求也不同：**TRAK/TracIn** 若要解释某个具体模型，最好直接用该模型或它的训练 checkpoints；**LESS/Prismatic/G-Vendi** 更适合用小型 instruction-tuned proxy；**GradAlign** 最严格，最好使用当前或接近当前的 policy，并随着 RL 训练周期性刷新。实践上可以从 **0.5B–7B proxy + LoRA/LM-head 梯度 + 256/1024维投影**开始，但这只是工程起点，不是理论保证。

### Per-sample gradients

LLM 全参数逐样本梯度不可直接落盘。由便宜到贵：

1. LM-head / embedding gradients；
2. LoRA adapter gradients；
3. 选定 blocks；
4. full gradient sketch。

用 microbatch + vectorized per-sample gradient，立即随机投影并丢弃原始高维向量：

$$\tilde g_i=\frac{1}{\sqrt d}R^\top g_i, \qquad R_{jk}\in\{-1,+1\}.$$

### Storage

$n$ 条、$d=1024$、FP16 的向量约占 $2nd$ bytes：100 万条约 2.05 GB（不含 metadata/index），可按 task/date/proxy 分 shard。在线 ANN 检索可使用 cosine index；谱统计使用 $d\times d$ covariance，避免构造 $n\times n$ kernel。

### Refresh

定义 proxy drift：

$$\Delta_t=1-\mathrm{Spearman} \big(s_{\theta_{t-1}}(Q),s_{\theta_t}(Q)\big)$$

其中 $Q$ 是固定 probe subset。超过阈值才全量刷新；否则只补新数据，降低梯度计算成本。

## 4. 调度策略

| 阶段 | 目标 | 推荐动作 |
|---|---|---|
| bootstrap | 建初始地图 | 小 proxy + 256/1024D sketch + coarse clusters |
| targeted SFT | 快速修能力 | target alignment shortlist，再做 coverage |
| online RL | 非平稳 curriculum | 每 N 轮刷新 validation gradient 与候选排名 |
| synthetic expansion | 补缺口 | 对稀疏且目标相关簇生成与 rejection sampling |
| continual update | 防遗忘 | conflict gate + replay + retention regression |

## 5. 可观测性与防 silent failure

每个 selection job 输出：

- pool → quality pass → gradient pass → selected 的瀑布计数；
- target alignment 分布与 top examples；
- G-Vendi、effective rank、簇 occupancy / entropy；
- proxy–target ranking correlation；
- conflict rate 与保护集 delta；
- duplicate / contamination / execution failure rates；
- GPU-hours、tokens selected、training tokens saved。

失败时禁止自动回退到“全收”：应 fail closed 或显式标记 fallback。

## 6. 最小生产护栏

1. selector 的 validation set 与最终 test set 隔离；
2. generator 不看到 benchmark answers；
3. stable sample ID 防止同义重复跨轮回流；
4. 每轮保留 random control arm，避免只看 selector 自证；
5. 每周/每里程碑跑全量 eval，日常可跑压缩 eval；
6. 新 selector shadow mode 先记录、不影响训练；
7. 数据版本、模型版本和 projection seed 可完全回放。

> 相关：proxy 五原则与方法适配见本文 §3，coverage 原理见 [03](03_gradient_coverage.md)，SPICE 协调性见 [09](09_spice_information_conflict.md)。

<!-- NAVIGATION -->
## 导航

- 上一篇：[05 安全与持续学习](05_safety_continual_learning.md)
- 下一篇：[07 Coding Flywheel](07_coding_data_flywheel.md)
- 回到：[目录 README](README.md) | [论文证据](papers.md) | [路线图](README.md#路线图)

> 串联：01 统一框架 → 02 归因/目标化 → 03 覆盖 → 04 生成 → 05 安全 → 06 系统 → 07 Coding 落地 → 08 边界 → 09 SPICE → 论文证据

