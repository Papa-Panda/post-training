# 04 — Systems：memory、communication、throughput、sample efficiency

算法名不会直接决定集群成本。先画数据流，再做逐项 accounting。

## 1. 两条训练数据流

```text
critic PPO:
  prompts -> behavior rollout -> reward/env -> critic values -> GAE
          -> actor forward/backward + critic forward/backward -> refresh snapshot

outcome GRPO:
  prompts -> G grouped behavior rollouts -> reward/env -> group statistics
          -> actor forward/backward -> refresh snapshot
```

两者都可能需要 reference log-probabilities 做 KL；model-based reward 还会增加 reward-model inference。rule/verifier reward 则没有 reward-model parameter memory，但可能有 sandbox、compiler 或 tool-service 成本。

## 2. Model-state memory accounting

设模型有 $P$ 个参数，不要用“几个模型”直接推显存。按状态拆：

| 状态 | actor | critic | reference | behavior |
|---|---:|---:|---:|---:|
| weights | trainable | PPO 中 trainable | frozen | 可由 actor snapshot 或仅存 old log-probs |
| gradients | yes | PPO 中 yes | no | no |
| optimizer states | yes | PPO 中 yes | no | no |
| activations | backward 所需 | PPO backward 所需 | 通常 inference only | rollout engine KV cache |

若 bf16 weights 每参数约 2 bytes，仅 weight 一项是 $2P$ bytes；Adam moments、master weights、gradients 的具体 bytes 取决于 optimizer precision、sharding、offload 与实现。于是删除“GRPO 固定省 30–50%”或“70B 固定省多少 GB”这类脱离配置的数字。

GRPO 确定省掉的是 **critic 的 weights/gradients/optimizer/activations/checkpoint/collectives path**。但总 peak 是否明显下降取决于谁主导：

- rollout KV cache 可能主导长上下文 generation；
- actor optimizer state 可能主导训练；
- reference 可 sharding/offload 或预计算 log-probs；
- colocated rollout 与 trainer 的 peak 不同于 disaggregated deployment；
- critic 可以参数共享、较小模型或独立同规模模型，不能默认必为 “1x actor”。

建议记录实测 breakdown：`actor_state`, `critic_state`, `ref_state`, `activations`, `kv_cache`, `allocator_reserved`，而不是只报 GPU utilization。

## 3. Communication accounting

### Critic PPO 多出来的通信

- critic data-parallel gradient reduce-scatter / all-reduce；
- sharded critic parameter all-gather；
- critic optimizer/checkpoint I/O；
- rollout token/hidden-state 到 critic trainer 的传输（依部署而定）。

### GRPO 多出来或放大的通信

- 同 prompt 的 $G$ 个 rewards 必须在 group 边界聚合；若 group 跨 ranks，要 all-gather/reduce group statistics；
- rollout 数增大后，tokens、masks、old log-probs、reference log-probs 的搬运量一起增长；
- dynamic sampling 需要 scheduler 不断补齐 non-degenerate groups，容易产生 stragglers；
- variable-length responses 造成 padding 或 load imbalance。

因此“无 critic = 通信更少”通常方向正确，但不是无条件：当 rollout-to-trainer payload 或 group synchronization 主导时，节省可能被抵消。

## 4. Compute 与吞吐模型

用一阶 wall-time 分解：

$$T_{step}=T_{rollout}+T_{reward}+T_{logp/ref}+T_{actor\ train}+\mathbf 1_{PPO}(T_{critic\ fwd}+T_{critic\ train})+T_{sync}+T_{idle}$$

固定 rollout token budget 时，GRPO 的 $G$ 不创造免费样本：

$$N_{completion}=N_qG,\qquad B_{tok}\approx\sum_{q=1}^{N_q}\sum_{i=1}^{G}T_{q,i}$$

增加 $G$ 的收益是更稳定的 within-prompt comparison，成本是 prompt coverage 降低、tail latency 增大、KV cache 增长。应该 sweep $G$，画 **reward/eval gain vs generated tokens**，而不是沿用无来源的固定 group size。

Prefix/KV reuse 可以降低同 prompt group generation 的 prefill 成本，但 decode 仍近似随总 generated tokens 增长。不同 engine 的 continuous batching、prefix caching 与 memory layout 会改变收益，必须 benchmark 当前版本。

## 5. Synchronization 与 staleness

同步 loop 最简单：rollout 全完成后再训练。问题是 longest response / slowest tool 决定 barrier。

异步 loop 提高设备利用率，但引入 behavior lag。每条 sample 至少携带：

```text
policy_version, prompt_id, response_id, token_ids, action_mask,
old_logprobs, rewards, terminated, truncated
```

训练侧监控：

- policy-version lag；
- token log-ratio quantiles 与 clip fraction；
- sample age；
- dropped-token fraction；
- group completeness（GRPO）；
- environment failure vs policy failure。

PPO clipping 和 GRPO clipping 都不是无限 staleness 的许可证。若 sample 太旧，应基于已验证的 acceptance rule drop/downweight，而不是伪装成 on-policy。

## 6. Multi-step / tool-use masks

Trajectory 中常混有 assistant actions、tool outputs、system text 和 compaction summaries。policy loss 只能落在定义为 action 的 token 上；reward/critic state 可读取 observation，但不要对环境生成的 token 求 policy gradient。

- **terminated**：任务自然结束，terminal bootstrap 通常为 0。
- **truncated**：超时、context limit、基础设施中断；不一定代表环境终止，PPO 可能需要 value bootstrap。
- **invalid**：sandbox crash、reward service error；应与真正的 task failure 分开统计。

Outcome GRPO 可给整条 assistant action mask 广播一个 reward，但长轨迹 credit 很粗。critic PPO 可估计每个 history 的 value，却不会凭空修复错误/可 hack 的 reward。

## 7. Sample-efficiency dashboard

至少同时报四种 denominator：

1. per prompt；
2. per completion；
3. per generated token；
4. per accelerator-hour。

GRPO 还要报：`non_degenerate_group_rate`, `all_equal_rate`, reward unique count, group std quantiles, effective prompts after filtering。PPO 还要报：value explained variance、value error by horizon、bootstrap/truncation rate。两者都报 exact-match/pass reward 与独立 held-out eval，防止只优化训练 verifier。

## 8. Deployment decision checklist

- critic 是否能通过小规模 fit/eval 得到正 explained variance？
- verifier 是否可靠，environment failure 是否已单独标记？
- 固定 token budget 下，增大 $G$ 是否真的提升 held-out metric？
- memory peak 的主项究竟是 critic、actor optimizer、activation 还是 KV cache？
- groups 是否跨 rank；group statistics 的 collective 是否可见？
- rollout/train 是否 colocated；权重同步成本是多少？
- response/token aggregation 的业务目标是什么？

没有这些测量时，不能从“PPO”或“GRPO”标签推导固定 GPU 数、吞吐率或最佳超参数。
