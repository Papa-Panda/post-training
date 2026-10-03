# CUA-MobileWorld — 面向真实场景的移动端智能体评测基准

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/CUA-mobileworld.html

## 元信息

- 专题：ICLR 2026 CUA Workshop 分享（青稞合集 sid=7971490）
- 标题：MobileWorld：面向真实场景的自主移动端智能体评测基准
- BV：BV1R6oUBBE2j
- 时长：约 19:17
- 提炼日期：2026-10-02
- 讲者：孔雀鱼（字幕自述，字幕作「孔雀玉」；通义实验室）
- B站链接：https://www.bilibili.com/video/BV1R6oUBBE2j/
- 字幕原文存档：本地 `transcripts/CUA-MOBILEWORLD.txt`（412 条，带时间戳）
- 相关工作：MobileWorld（ACL 2026 接收，2025 年底开源）；同合集前一场为 MAI-UI（见 CUA-mai-ui），两者同属通义实验室 GUI agent 工作
- 说明：基于 B站 AI 字幕整理；应用与模型名以可辨者为准（如 MATTERMOST 即 Mattermost、MASTERDA 即 Mastodon、MOBIWORD 即 MobileWorld），个别音译未逐一核对官方拼写。

## 一句话总结

AndroidWorld 分数已经刷到区分度不足，MobileWorld 用「自部署开源 app 拿回后端校验权」这一招同时解决真实场景与确定性校验的两难，再叠加用户交互与 MCP 两个新任务维度，做出更难（平均步数约 2 倍、跨应用任务占比 62%）、更贴近真实的移动端评测基准。

## 核心

1. **背景/问题**：作为 2024 年以来移动 GUI agent 的 go-to benchmark，AndroidWorld 头部分数持续走高、区分度明显不足。更深一层是评判的两难：涉及 Gmail、国产 app 这类闭源后端时拿不到后端权限，只能用 LLM-as-judge，丢了确定性校验；而做确定性校验的 benchmark 又只能退到系统 app 和离线 app，任务场景分布被迫变窄。此外旧基准默认用户指令完全明确，也没有把 MCP 工具调用纳入评测。
2. **方法/设计**：
   - **基准构成**：约 20 个应用、200 多个精心设计任务。任务平均步数接近 AndroidWorld 的 2 倍，跨应用任务占比约 62%（AndroidWorld 只有约 10%），切换过来后主流 SOTA 模型分数明显回落，重新拉开差距。
   - **自部署环境换校验权**：用开源替代 app 自建后端——Mattermost（Slack 替代）、Mastodon（X/Twitter 替代），再加仿淘宝、仿 Gmail 的 mock app；后端以 docker-in-docker 集成进评测环境，配合 emulator 快照与数据库持久化，保证每次评测初始状态一致。校验走四路：文本校验（正则/外部 API，面向信息查询类）、后端 SQL 查询（有数据库权限时用 Python helper 直接查状态变更）、ADB 查 app 内部 SQLite、以及在自研 app 里植入 hook 把关键操作状态写成本地 JSON 再回调校验。
   - **两个新任务维度**：其一是 agent-user 交互——参考 τ-bench 的思路，用一个模型扮演 user agent，把关键信息（如收件人邮箱）从指令里藏掉，agent 探索后发现信息缺失应主动触发 ask-user，回答再拼进上下文继续执行；其二是 MCP 任务——特意构造必须交织调用 MCP 工具与 GUI 操作才能完成的任务，工具侧用手机上没装的东西（高德、GitHub、arXiv 一类），检验两种执行方式交织时的成功率。
   - **统一工具链与真机评测**：正在把 AndroidWorld、AndroidLab 等环境统一集成进 MobileWorld 的评测工具链（AndroidWorld 支持已在 branch 上）；近两个月支持了经 ADB 的真机评测，并持续修任务错误与登录失效一类环境问题、加批量评测服务管理。
3. **实验/实战**：博客更新里他们把前沿闭源模型直接部署到真机做端到端评测（Seed 2.0 Pro、Gemini 3 Pro、Claude 4.5、Kimi K2.5、Qwen 3.5），统一用一套通用端到端 prompt，再按各自的坐标尺度与 grounding 能力做适配；发现给 Seed 2.0 Pro 沿用 OSWorld 的专用模板与通用 prompt 效果差异很大。结论是端到端模型已在 GUI-only 与用户交互任务上达到 SOTA 水平；但闭源模型单轮评测费用要 50 到 100 多美元，开源模型（MAI-UI、Kimi 等）提供了接近 SOTA 的更高性价比选项。Gemini 团队看到评测结果后还主动联系交流。常见失败模式也总结得很具体：该澄清时不澄清导致幻觉、长程任务里算错、MCP 工具返回过长撑爆上下文丢目标、不知道今天几号这类时间感知缺失。
4. **结论/观点**：讲者的立场是评测基准的价值在于「确定性校验 × 真实场景分布」两者都要，不能为了可校验就把任务退化到玩具 app；做法上宁愿自建可控后端，也不接受 LLM-as-judge 的不可复现。同时他把成本摆上台面：闭源模型一轮真机评测 50–100+ 美元，这个量级本身就在改变社区做评测的方式。

## 关键数字

| 指标 | 基线 | 结果 |
|---|---|---|
| 应用 / 任务规模 | — | 约 20 个应用，200+ 任务 |
| 平均任务步数 | AndroidWorld | 约 2 倍 |
| 跨应用任务占比 | AndroidWorld 约 10% | 约 62% |
| 校验方式 | LLM-as-judge（闭源后端无奈之选） | 文本/正则、后端 SQL、ADB SQL、app 内 hook 四路确定性校验 |
| 闭源模型单轮真机评测费用 | — | 约 \$50 至 \$100 以上 |
| 接收情况 | — | ACL 2026 接收，2025 年底开源 |

## 可迁移

- 对 coding data / RL infra 工作的直接可试点：「自部署开源替代 app 拿回后端权限」是评测环境设计的一招通法——任何需要确定性校验的 agent 评测（代码执行、工具调用、内部系统操作），与其在真实闭源系统上用 LLM judge，不如自建带状态后端的等价环境，校验直接查状态变更，顺带白拿可复现的初始状态快照。ask-user 任务维度也值得搬：把关键信息藏进 user agent、要求 agent 主动澄清，是低成本的指令鲁棒性探针。
- Infra 视角：评测成本已经是一等约束——闭源模型单轮真机评测 50–100+ 美元，评测 infra 需要把环境快照、批量调度、跨 benchmark 统一工具链当基础设施来做；MobileWorld 把多个 benchmark 环境统一进一套工具链的做法，对内部评测平台同样成立。

## 疑问 / 下一步

- 自建 mock app（仿淘宝、仿 Gmail）与真实 app 的行为差距会不会系统性低估难度，字幕没有给出与真实 app 的对照实验；另外 ask-user 里 user agent 的回答质量与真实用户的分布差异，可能让交互维度分数偏乐观，想看 paper 里这部分的设计与校验细节。

## 原文金句

> 「我们通过采用自部署开源 app 环境的方式去规避这个问题……确保每次评测任务的初始状态是一致的。」

> 「基本上单轮评测下来都要花上 50 刀，有些甚至要 100 刀多的一个评测费用。」
