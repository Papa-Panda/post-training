# 论文笔记模板

复制这个模板写每篇的笔记。要求：每个数字必须出自论文原文，
没有可靠数字时不硬写；公式按 GitHub 渲染规则写
（行内公式的 `$` 外侧留空格、内侧贴紧；独立公式块上下各空一行）。

## 元信息

- Title:
- Authors:
- Link / arXiv:
- Date of publication: YYYY-MM-DD（填论文在 arXiv 上的 published 日期，查 arXiv API，不要填阅读日期）
- Tags: [eval, synthetic-data, verification, decontamination, difficulty, process-supervision, outcome-supervision, rlvr, curation]
- 与前篇的关系：

## 一句话总结

这篇 paper 解决了什么数据问题，用了什么关键方法，效果怎样。

## 3 个问题

偏向实验设计（参看 ai-data repo 的思考题标准）：每道题都要能动手验证，不能只复述论文结论。好问题的样子：给出对照臂/消融臂（A/B/C），指明评什么指标，先写判据（若出现结果 X → 结论 Y；若出现结果 Z → 结论 W），再指出这个实验设计本身的漏洞；或构造一个具体反例、预测一个 2×2 格子的排序并写理由。至少一道题要综合本系列已读的另一篇。答案不在原文里。

1.
2.
3.

## 核心机制

1. **Motivation**：为什么要做这个数据工作？基线痛点是什么？
2. **方法**：数据从哪里来、怎么生成、怎么过滤、怎么验证？
3. **关键数字**：规模、准确率变化、消融结论（注明出自论文哪部分）。

## 边界

- 这篇方法在什么条件下失效？
- 它对哪些题型、难度、语言不适用？
- 它的评测有没有可质疑的地方（污染、过拟合、指标选择）？

## 迁移

- 这个数据做法能搬到别的领域吗？搬过去要改什么？
- 它和本系列已读的哪篇构成对照（比如过程监督 vs 结果监督、规模 vs 质量）？
- 如果让你复现，第一步先做什么？

## 讨论题

- 这篇的结论里，哪一条最可能被后来的工作推翻？
- 数据 pipeline 中哪个环节最容易藏 bug？
- 换成今天的模型规模，这个方法还成立吗？

## 阅读版

- 每篇同时写 `index.html`（与 NOTES.md 同目录），按 ai-data Day35 风格：hero + 目录 + 元信息 + 一句话总结 + 3 个问题 + 核心机制 + 边界 + 迁移 + 讨论题。
- 可视化尽量多：stat strip（关键数字卡片）、flow（方法流程）、bars（关键数字条形对比）、vs（方法对照）、mermaid 流程图、KaTeX 公式；每个数字必须出自论文原文、与 NOTES.md 一致。
- html 版是本体，放入 GitHub notes：`index.html` 与 NOTES.md 配对在同一目录、一起入库。NOTES.md 标题后加一行 `> 📖 阅读版：https://papa-panda.github.io/post-training/synthetic-data/<folder>/`（本仓库公开、Pages 可用，格式与 ai-data 一致）。
