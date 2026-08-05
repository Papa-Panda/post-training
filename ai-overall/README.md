# ai-overall — General AI Paper Reading

更广义的 AI paper 沉淀区，不限于 data。不像 `ai-data` 专门做 coding data / SFT / RL data，这里放：

- Agentic RL / Post-training Infra / RL Scaling
- AI4AI / Recursive Self-Improvement / Self-Evolution
- LLM Systems / Eval / Reasoning / Code Agents
- 经典 / 热门 但不直接归到 data track 的 paper

## 结构

```
ai-overall/
├── README.md                    # 这个说明
├── PAPER_TEMPLATE.md            # 读一篇 paper 的固定模板（overall 通用版）
├── reading-log.csv              # 快速索引所有读过的 paper
└── papers/
    └── {year}_{short-name}/     # 每篇一个文件夹
        ├── NOTES.md            # 按模板填
        └── assets/             # 图、截图、可选
```

## 怎么用

1. 在 `ai overall` thread（当前 thread）里丢 paper link 或标题
2. 我按 `PAPER_TEMPLATE.md` 提炼：Motivation / Method / System / Results / 可迁移
3. 落地到 `papers/{xxx}/NOTES.md`，并更新 `reading-log.csv`
4. 需要时同步 commit 到主 repo，作为 45-day plan 的 Papers track 证据

关联：

- 主计划：`ai_daily.csv` Phase & Papers track
- 子专项：`ai-data/` 专门做 data track，二者互补不重复
- 日常讨论：在 Hatch 的 `ai overall` thread 里进行，不污染 `ai` / `ai data`

> 规则：公共 repo 不出现雇主标识（已清理 2026-08），保持诚实简洁。
