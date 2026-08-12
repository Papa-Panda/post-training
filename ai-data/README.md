# ai-data — Data-centric AI Papers 专门读 data 相关的 AI paper 的沉淀区。服务于从 ML for Infra → Post-training / Agentic RL Infra 转型，重点是 **coding data / SFT / RL data / data curation / quality / flywheel**。

## 结构
```
ai-data/
├── README.md
├── PAPER_TEMPLATE.md
├── reading-log.csv   # 快速索引
└── {year}_{short-name}/  # 每篇 paper 一个文件夹
    ├── NOTES.md
    └── assets/
```

> 之前有层多余的 `papers/` 已去掉，现在直接平铺，路径更短。

## 怎么用
1. 在 `ai data` thread 里丢 paper link 或 PDF
2. 我按 `PAPER_TEMPLATE.md` 提炼：问题 / data pipeline / 关键 trick / 可迁移点
3. 落地到 `{xxx}/NOTES.md`，并更新 `reading-log.csv`
4. 每日同步到主 repo 的 commit，作为 45-day plan 的 Papers track

关联：
- 主计划：`ai_daily.csv` Reasoning Data track
- 讨论：在 Hatch 的 `ai data` thread 里，不污染主 thread
