# eval / 评估轨道

本目录是评估相关专题的索引。

Currently:

- `eval-compression/` — **评估压缩**：Factory 方法 + Hermes probe harness + 三类压缩对比 (prompt / context / KV)

设计原则：
- 不做又一个大表格 scoreboard，聚焦 **如何在真实 fixture 上探测压缩失真**
- 保持离线可复现：fixture + compress() + probe + judge 即可完成一轮评

后续可加：
- `eval-reward-hacking/` — coding agent reward hacking 检测
- `eval-long-horizon/` — SWE-Marathon / FrontierSWE 长程指标

> 所有子 track 保持 bilingual concise，与 `ICL/`, `grpo-vs-ppo/`, `vllm-rollout/` 同风格。
