# ai-data — Data-centric AI Papers 专门读 data 相关的 AI paper 的沉淀区。服务于从 ML for Infra → Post-training / Agentic RL Infra 转型，重点是 **coding data / SFT / RL data / data curation / quality / flywheel**。

## 结构
```
ai-data/
├── README.md
├── PAPER_TEMPLATE.md
├── reading-log.csv   # 快速索引
└── day-01-xxx/  # 每篇 paper 一个文件夹，prefix 对齐 rl-infra/day-0x-
    ├── NOTES.md
    └── assets/
```

> 之前用 `{year}_{short-name}/` 已全量重命名为 `day-XX-{year}-{short-name}/`，与 `rl-infra/day-01-xxx` 保持一致，便于 `ai_daily.csv` / `ai data` thread 中 Day N 直连。

命名：
- `day-01` ~ `day-16` 已用，对应 2026-08-04 ~ 2026-08-16 经典 + post-train 主线
- 新 paper 从 `day-17` 递增，格式 `day-{NN}-{year}-{slug}`，如 `day-17-2024-phicoder`，保持两位数零填充
- `NOTES.md` 模板同 `PAPER_TEMPLATE.md`，必须含「和之前工作的关系」小节

关联：
- 主计划：`ai_daily.csv` Reasoning Data track
- infra 轨道：`rl-infra/day-01-ddp-basics/` ~ `day-12-reward-model`，同前缀可直接混排查找
- 讨论：在 Hatch 的 `ai data` thread 里，不污染主 thread
- GitHub 树：https://github.com/Papa-Panda/post-training/tree/master/ai-data
