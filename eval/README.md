# eval / 评估轨道索引

本目录是评估相关 flat tracks 的索引，**实际内容在顶层平铺**，不在 `eval/` 下嵌套。

Currently flat at repo root:

- `eval-context-compression/` — 评估压缩（context compression）：Factory 方法 + Hermes probe harness + 三类压缩对比 (prompt / context / KV)，原来 `eval-compression/` 已重命名于此
- `eval-bench-efficiency/` — 评测集高效化（bench efficiency）：Metabench IRT 蒸馏 28k→858 (<3%) + mRMR 特征选择 + DIoR x100 节省，代表作 metabench / BenchBench

设计原则：
- 不做又一个大表格 scoreboard，聚焦 **如何在真实 fixture 上探测失真** 或 **如何在 1% 题里保 99% 排序**
- 保持离线可复现：fixture + compress() + probe + judge / mRMR subset 即可一轮评
- 与 `ICL/`, `grpo-vs-ppo/`, `vllm-rollout/` 同风格 bilingual concise

后续可加：
- `eval-reward-hacking/` — coding agent reward hacking 检测
- `eval-long-horizon/` — SWE-Marathon / FrontierSWE 长程指标
