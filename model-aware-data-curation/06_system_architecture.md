# 06 — 系统架构：Gradient Datastore + Curation Control Plane

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

## 3. 计算路径

### Per-sample gradients

LLM 全参数逐样本梯度不可直接落盘。由便宜到贵：

1. LM-head / embedding gradients；
2. LoRA adapter gradients；
3. 选定 blocks；
4. full gradient sketch。

用 microbatch + vectorized per-sample gradient，立即随机投影并丢弃原始高维向量：

\[
\tilde g_i=\frac{1}{\sqrt d}R^\top g_i,
\qquad R_{jk}\in\{-1,+1\}.
\]

### Storage

$n$ 条、$d=1024$、FP16 的向量约占 $2nd$ bytes：100 万条约 2.05 GB（不含 metadata/index），可按 task/date/proxy 分 shard。在线 ANN 检索可使用 cosine index；谱统计使用 $d\times d$ covariance，避免构造 $n\times n$ kernel。

### Refresh

定义 proxy drift：

\[
\Delta_t=1-\mathrm{Spearman}
\big(s_{\theta_{t-1}}(Q),s_{\theta_t}(Q)\big)
\]

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
