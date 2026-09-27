# Paper 模板

> 复制这个模板到 `{name}/NOTES.md`（现在直接在 ai-data 下平铺，不再有 papers/ 中间层）

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
- Link / arXiv:
- Date read: YYYY-MM-DD
- Tags: [coding-data, sft, rl-data, curation, quality, eval, flywheel]

## 一句话总结
这篇 paper 解决了什么 data 问题，用了什么关键方法，效果怎样。

## 核心
1.  **Motivation**: 为什么要做这个 data 工作？baseline 痛点？
2.  **Data Pipeline**: 数据从哪来 → 怎么洗/合成/过滤 → 怎么评 → 怎么进训练
3.  **Key Tricks**: 3个最值得抄的细节（阈值、模型、规则、去重、合成 prompt）
4.  **Results**: 对 downstream 有多大提升？用什么评的？

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：
- Infra 视角：可扩展性 / 成本 / 评测自动化的启发：

## 疑问 / 下一步
- 没看懂的 / 想深挖的 1 个问题

## 原文金句 (1-2句)
> 
