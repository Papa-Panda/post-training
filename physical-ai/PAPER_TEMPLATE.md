# Paper / System 模板 — physical-ai

> 复制这个模板到 `day-{NN}-{slug}/NOTES.md`

<!--
可视化标注（viz directives）——完成稿硬性要求
------------------------------------------------
标题行 `# ...` 下方放 2-3 个 viz 指令（HTML 注释），阅读版生成器
`~/workspace/tools/md2notes_html.py` 会渲染成可视化卡片。
生成器也会自动提取一部分（元信息→表格、一句话总结里的规模数字→stat 卡、
干净的 A → B → C 链→流程图、带 vs/对比 的百分比→条形图），但显式指令优先、
质量更高，必须手写。

格式（写成 HTML 注释，照抄结构、按论文重写内容）：
  viz:stats: 800K step-level标签 | 2.6× 数据效率
  viz:flow: 数据来源 → 清洗/过滤 → 训练 → 评测
  viz:bars: 方法A 78.2% | 方法B 69.6%
  viz:vs: Process监督 | 逐步骤反馈; 信用分配容易 || Outcome监督 | 只看最终答案; 有误判噪声
即：把上面四行分别包成 HTML 注释（行首加左尖括号+感叹号+两短横，行尾加两短横+右尖括号）。

硬规则：
- 每个数字、每个标签必须出自论文/NOTES 原文，严禁编造、估算或"看起来合理"的补数；
  没有可靠数字时只写 flow 或 vs，不硬画 bars。
- flow 每节点至多 12 字，不许是问句、"待补充"、"怎么做/为什么"类占位语。
- bars 只用于原文明确对比过的数字（vs / 对比 / 从 A 到 B）。
- [待读]骨架不写任何指令。
-->

## 元信息
- Title:
- Authors / Org:
- Link / arXiv / Blog:
- Date read: YYYY-MM-DD
- Tags: [physical-ai, humanoid, world-model, sim2real, isaac-lab, habitat, control, vla, rl-robotics]
- Thread: physical-ai
- Folder: day-NN-xxx
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-NN-xxx

## 一句话总结
这篇 paper / 系统解决了什么 Physical AI 问题，用了什么关键方法，效果怎样。

## 和之前工作的关系

- 接了哪条线：
- 补了哪个短板：
- 替代 / 分叉 / 改进：
- 对之前 Day X 的直接对比：

## 为什么今天读它

## 今天的 3 问
1. 
2. 
3. 

## 核心
1.  **Motivation**: 为什么要做？baseline 为什么不行？跟 Physical AGI 的关系？
2.  **System / Method**: 系统架构 → 感知/控制/学习如何对齐 → 关键 operator / world model / policy 机制
3.  **Training / Data Details**: Sim 数据怎么来？Real 数据怎么来？Sim2Real 怎么做的？Reward / Verifiable signal 是什么？
4.  **Key Tricks**: 3个最值得抄的细节（控制、数据、sim、reward 设计）
5.  **Results**: 在什么 bench / real robot 上，用什么 budget，相对 base 提升多少？

## 可迁移 / Transfer

- 方法在 held-out 上是否 transfer？模型 vs 框架 哪个贡献更大？
- 对你 Infra → Post-training → Physical AI 迁移的 1-2 个直接启发：
- Infra 视角：可扩展性 / 成本 / 评测自动化 / 可复现性：

## 疑问 / 下一步

- 没看懂的 / 想深挖的 1 个问题
- 如果要复现 / 小规模试，第一个实验做什么？

## 原文金句 (1-2句)
> [阅读后补原文，勿凭记忆引用]

## 今晚产出
- 按模板补齐 System / Training / Key Tricks / Results / 可迁移
- 保留并完善「和之前工作的关系」小节

## 连接
- 上一篇: 
- 下一篇预告:
