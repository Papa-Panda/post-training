# ai-data — Data-centric AI Papers 专门读 data 相关的 AI paper 的沉淀区。服务于从 ML for Infra → Post-training / Agentic RL Infra 转型，重点是 **coding data / SFT / RL data / data curation / quality / flywheel**。 ## 结构
```
ai-data/
├── README.md # 这个说明
├── PAPER_TEMPLATE.md # 读一篇 paper 的固定模板
├── reading-log.csv # 快速索引所有读过的 paper
└── papers/ # 每篇 paper 一个文件夹 └── {year}_{short-name}/ ├── NOTES.md # 按模板填 └── assets/ # 图、截图
``` ## 怎么用
1. 在 `ai data` thread 里丢 paper link 或 PDF
2. 我按 `PAPER_TEMPLATE.md` 提炼：问题 / data pipeline / 关键 trick / 可迁移到你现在 coding data 工作的点
3. 落地到 `papers/{xxx}/NOTES.md`，并更新 `reading-log.csv`
4. 每日同步到你主 repo 的 commit 历史，作为 45-day plan 的 Papers track 证据 关联：
- 主计划：`ai_daily.csv` Reasoning Data track
- 日常讨论：在 Hatch 的 `ai data` thread 里进行，不污染 `ai` 主 thread
