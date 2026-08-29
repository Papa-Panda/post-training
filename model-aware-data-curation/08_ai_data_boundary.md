# 08 — 与 `ai-data/` 的边界：不重复，做上层闭环

## 元信息
- 内容类型：仓库边界与去重规则，不对应单篇论文
- 论文总索引：[papers.md](papers.md)
- 本章定位：规定 `ai-data/` 保存逐篇论文笔记，本专题只保存跨论文方法、系统闭环与新增锚点。


## 1. 边界原则

| 维度 | `ai-data/` | `model-aware-data-curation/` |
|---|---|---|
| 内容单位 | 一篇论文一份 NOTES | 跨论文方法论与系统闭环 |
| 主问题 | 数据如何产生、清洗、筛选、去重、验证 | 当前模型缺什么，下一批数据选/造什么 |
| 时间尺度 | dataset recipe / release 级 | checkpoint / policy round 级 |
| 信号 | quality、规则、来源、embedding、执行 | gradient value、target alignment、coverage、conflict |
| 输出 | 可复现数据配方与论文路线图 | selector/generator control plane 与在线 flywheel |
| 代码 | 单论文 demo | 统一 gradient geometry primitives |

## 2. 已覆盖论文：这里只引用

以下内容在 `ai-data/` 已有逐篇笔记，本专题不再写摘要/结果复盘：

- [Influence Functions — Day 02](../ai-data/day-02-2017-influence-functions/NOTES.md)
- [TracIn — Day 03](../ai-data/day-03-2020-tracin/NOTES.md)
- [LESS — Day 04](../ai-data/day-04-2024-less/NOTES.md)
- [DataInf — Day 05](../ai-data/day-05-2024-datainf/NOTES.md)
- [Vendi Score — Day 19](../ai-data/day-19-2023-vendi-score/NOTES.md)
- [D4 / SemDeDup — Day 24](../ai-data/day-24-2023-semdedup-d4/NOTES.md)
- synthetic-data 背景可继续从 [Phi-1](../ai-data/day-06-2023-phi-1/NOTES.md)、[Self-Instruct](../ai-data/day-21-2022-self-instruct/NOTES.md)、[Evol-Instruct](../ai-data/day-22-2023-evol-instruct/NOTES.md)、[OSS-Instruct](../ai-data/day-27-2023-oss-instruct/NOTES.md) 阅读。

本专题只抽取这些论文在统一闭环中的“接口”：

```text
IF/TracIn -> attribution primitive
LESS/DataInf -> target-value primitive
Vendi/D4/SemDeDup -> coverage baseline
synthetic data papers -> generator/input pipeline
```

## 3. 本专题新增的部分

1. **TRAK**：把 attribution 推向可扩展随机投影/ensemble；
2. **GradAlign**：RL 非平稳场景的在线 target-aligned curriculum；
3. **G-Vendi**：把 Vendi kernel 换成模型诱导的梯度方向；
4. **Prismatic Synthesis**：从“挑已有数据”升级到“针对稀疏方向主动生成”；
5. **GrADS / OGS**：把遗忘与梯度冲突纳入选择约束；
6. **统一系统**：gradient datastore、refresh、proxy-target 校准、control plane；
7. **coding-data flywheel**：执行验证、失败簇、生成、选择、训练、回归。

## 4. 未来内容放哪里

判断规则：

- 新论文主要贡献是 corpus 配方、过滤器、数据集、去重或 synthetic recipe → `ai-data/day-xx-*`；
- 新论文主要贡献是根据模型梯度动态决定数据价值/覆盖/生成/保护 → 本专题；
- 同时横跨两边 → `ai-data` 保留逐篇 NOTES，本专题只补系统接口与交叉比较，并用相对链接引用；
- 算法主要改变 PPO/GRPO objective 或 rollout engine → 分别去 `grpo-vs-ppo/` 或 `vllm-rollout/`，不因“使用数据”而塞进这里。

## 5. 防重复检查表

新增文档前检查：

- [ ] 是否已经有 `ai-data/day-*` NOTES？有则只引用。
- [ ] 新内容是否至少连接两类方法或一个系统闭环？
- [ ] 是否解释 model state / gradient 如何改变 curation decision？
- [ ] 是否包含可运行 primitive、架构或实验设计，而非第二份论文摘要？
- [ ] 是否明确 claim 的论文版本、实验范围与外推边界？
