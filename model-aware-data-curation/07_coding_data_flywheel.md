# 07 — Coding-data Flywheel 落地

## 元信息
- 内容类型：应用设计，不对应单篇论文
- 理论锚点：[LESS](https://arxiv.org/abs/2402.04333)（目标价值）· [Prismatic Synthesis](https://arxiv.org/abs/2505.20161)（覆盖与主动生成）· [OGS](https://arxiv.org/abs/2602.06359)（保护能力）
- 本章定位：把上述信号落到可执行、可验证的 coding-data flywheel。


## 1. 闭环目标

不是生成更多题，而是让每轮数据解决当前模型尚未覆盖的可验证能力：

```text
production/eval failures
  -> error taxonomy + trusted target set
  -> candidate pool + synthetic proposals
  -> parse/compile/test/decontam gates
  -> gradient target + coverage + conflict selection
  -> SFT/RL
  -> ID/OOD + retention regression
  -> next failure map
```

## 2. 样本单位

对 coding 数据，样本不应只有 `(prompt, answer)`，而应保存：

```json
{
  "prompt": "...",
  "solution": "...",
  "language": "python",
  "tests": ["..."],
  "execution": {"pass": true, "runtime_ms": 18},
  "trace": {"tool_calls": [], "errors": []},
  "skill_tags": ["graph", "invariant"],
  "source": "synthetic|human|failure-replay",
  "parent_ids": ["..."],
  "generator": "...",
  "data_version": "..."
}
```

provenance 和 parent IDs 用于发现生成循环中的近重复与污染。

## 3. 四级 gate

1. **Syntax**：parse、依赖白名单、格式；
2. **Execution**：sandbox、unit tests、timeout/memory；
3. **Contamination**：exact hash、10-gram、AST/embedding near-match、benchmark隔离；
4. **Model-aware**：target alignment、coverage gain、conflict。

前三级是硬门，第四级才做预算排序。梯度大或稀疏不能救回错误程序。

## 4. 从失败簇构造 target set

将失败按可操作机制而非 benchmark 名称聚类：

- spec misunderstanding；
- state / invariant tracking；
- algorithm selection；
- long-horizon decomposition；
- tool/API misuse；
- test generation / edge cases；
- efficiency / timeout；
- repair after feedback。

每簇抽取少量去污染、人工/执行可信的 $V_k$，计算 $\bar g_{V_k}$。候选样本可同时拥有多个 alignment：

$$a_{ik}=\cos(\tilde g_i,\bar g_{V_k}).$$

这样能按能力缺口分配预算，而不是把所有 coding data 压成单个总分。

## 5. Prismatic-style generation

对“目标相关但 occupancy 低”的簇生成：

1. 从该簇和相邻簇抽 few-shot；
2. 约束 generator 改变算法结构/约束组合，而非只换故事皮肤；
3. 生成 problem、reference solution、tests；
4. 多解/多语言编译执行验证；
5. 计算候选梯度，拒绝回到密集簇的样本；
6. 去污染后进入 train buffer。

一个批次的 acceptance rate 本身是诊断：若很低，generator prompt 或 seed examples 只会复读已有模式。

## 6. SFT 与 RL 的不同信号

| 阶段 | 梯度 | 候选单位 | 刷新频率 |
|---|---|---|---|
| SFT | token NLL gradient | `(prompt, verified solution)` | checkpoint/数据版本级 |
| RL | policy gradient / advantage-weighted log-prob gradient | problem + rollout group | policy round 级 |

RL 中同一道题随 policy 变化可能从“有学习信号”变成“全对/全错”；因此要把成功率、reward variance、trajectory length 与 GradAlign 一起记录。

## 7. 实验矩阵

固定同一 base model、token budget 与执行 gate，比较：

| Arm | 选择策略 |
|---|---|
| A | random verified |
| B | embedding diversity |
| C | target alignment only |
| D | G-Vendi coverage only |
| E | target + G-Vendi |
| F | target + G-Vendi + conflict gate |
| G | F + Prismatic-style generation |

报告：

- ID/OOD pass@1 与 pass@k；
- 每个失败簇的 recall / accuracy；
- general retention delta；
- G-Vendi 与 embedding Vendi；
- execution/contamination/duplicate rate；
- tokens、GPU-hours、wall time；
- selector 在 proxy 和 target model 上的 rank correlation。

## 8. Go / no-go

只有同时满足以下条件才扩大闭环：

- F/G 相比 random 在 OOD 和目标失败簇有稳定增益；
- retention 不超预算；
- proxy-target rank correlation 可接受；
- 生成新增覆盖而非新增噪声；
- 端到端节省的训练预算大于 gradient/verification 开销。

<!-- NAVIGATION -->
## 导航

- 上一篇：[06 系统架构](06_system_architecture.md)
- 下一篇：[08 与 ai-data 边界](08_ai_data_boundary.md)
- 回到：[目录 README](README.md) | [论文证据](papers.md) | [路线图](README.md#路线图)

> 串联：01 统一框架 → 02 归因/目标化 → 03 覆盖 → 04 生成 → 05 安全 → 06 系统 → 07 Coding 落地 → 08 边界 → 09 SPICE → 论文证据

