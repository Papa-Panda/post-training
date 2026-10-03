# CUA-MAI-UI — 通向可落地的 GUI Agent：MAI-UI 与 UI-Ins

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/CUA-mai-ui.html

## 元信息

- 专题：ICLR 2026 CUA Workshop 分享（青稞合集 sid=7971490）
- 标题：通向可落地的 GUI Agent：通义 GUI 基座模型 MAI-UI、UI-Ins (ICLR 2026) 技术报告
- BV：BV1dqoUB6ECy
- 时长：约 38:46
- 提炼日期：2026-10-02
- 讲者：韩章（字幕无本人自述姓名；据紧随其后的 MobileWorld 讲者孔雀鱼在字幕中以「韩章」称呼前一位讲者，且本场内容与该称呼所指的 grounding 博客一致）
- 机构：通义实验室（据同场 MobileWorld 讲者自述与内容互证）
- B站链接：https://www.bilibili.com/video/BV1dqoUB6ECy/
- 字幕原文存档：本地 `transcripts/CUA-MAI-UI.txt`（978 条，带时间戳）
- 相关工作：MAI-UI 技术报告、UI-Ins（ICLR 2026）、团队 grounding 复现博客、MobileWorld（见同合集 CUA-mobileworld）
- 说明：基于 B站 AI 字幕整理；专名以可辨者为准（如「web bean lab」应为 WebinLab、「卖UI」即 MAI-UI、「ZIN」即 zoom-in），个别音译未逐一核对官方拼写。

## 一句话总结

MAI-UI 是一套覆盖 2B 到 235B 的 GUI 基座模型，靠四件事冲到可落地：把 grounding 的思考方式换成「指令即推理」（UI-Ins）、用 action 级细粒度 judge 把错误轨迹也榨干、纯端到端 online RL（GRPO + 500+ 并行环境）练长程任务、再用端云协同与 MCP/用户交互补齐部署侧的短板。

## 核心

1. **背景/问题**：GUI agent 离真实部署还差几层：grounding 点不准（闭源模型报告的分数按官方 recipe 很难复现）、长程轨迹数据利用率低、纯端侧小模型能力不够而纯云端有隐私与时延问题、用户指令天然带歧义。这场分享按「grounding 怎么做思考、数据怎么榨干、RL 怎么端到端练、部署怎么端云协同」逐层拆。
2. **方法/设计**：
   - **UI-Ins（grounding 的思考方式）**：人描述一个目标会从外观、功能、位置、目的等不同角度出发，UI-Ins 把这些「指令角度」本身作为模型的 reasoning 方式——先给原数据打上多角度指令标签，SFT 阶段训 4~5 种角度，RL 阶段让模型按 reward 学会选最合适的角度。普通 chain-of-thought 在 grounding 上不 work 甚至掉点，这种与人的实际思考同构的方式才有效；SFT 训多样推理路径还给 RL 留足探索空间，避免 policy collapse。意外收获：RL 后模型长出了训练时没教过的新指令角度、还会融合多个角度。
   - **数据管线（navigation）**：任务生成、轨迹合成、拒绝采样的常规骨架之外，重点是一个 action 级（而非 trajectory 级）的细粒度 judge：正确轨迹里的坏步骤会被筛掉，错误轨迹（占比很高）也不整条丢弃，而是保留其最大正确前缀，数据利用率显著拉高。
   - **Online RL**：纯端到端，在动态手机/docker 环境里 rollout 与策略更新交替迭代。用 GRPO + 轨迹 reward，针对模型爱陷进重复动作循环的毛病加了重复动作惩罚，任务按课程学习挑选。支撑长轨迹的工程手段是大量并行 + 把图像分辨率降一半（手机 icon 密度低，降一半对性能无影响，降到 1/4 才明显掉点）；并行环境数做到 500 以上，可支持的最大交互步数从 15 提到约 50 步。选端到端的理由讲得很直白：infra 门槛虽高，但链路一旦打通，整个训练是 computation driven——只要喂 query 和算力就能 scale。
   - **端云协同**：端侧模型同时承担执行与 monitor 两职（不另设 monitor agent，避免冗余），检测到轨迹偏离用户意图时把历史轨迹上下文加错误原因总结一起上交云端大模型接管纠错；轨迹记忆组件放在端侧存上下文，兼顾隐私。monitor 训练时专门学了「提取当前轨迹为什么错」，只给云端历史而不给错误信号时纠错明显更难。
   - **MCP 与用户交互**：给 GUI agent 接上 MCP 工具调用（高德、GitHub、arXiv 等），把长链 GUI 操作压成单步 API 调用、减少 error propagation，还能让 mobile agent 解锁桌面级任务；用户交互机制让 agent 主动识别指令的歧义与信息缺失、向用户提问（危险操作如支付、登录验证也要确认），回复再组织进上下文继续执行。
3. **实验/实战**：AndroidWorld 上 235B 模型在技术报告发布时达到 76.7 分的 SOTA（字幕），超过 Gemini 与 Seed 1.8 一代模型，四个尺寸均为同尺寸 SOTA；ScreenSpot Pro 上 32B 模型 73.5 分，超过 Gemini 3 Pro。Grounding 复现研究里，Gemini 3 Pro 报告 72.7 分、直接复现只拿到约 39 分，加上 zoom-in 才回到报告水平——zoom-in 提升大但对放大倍数等超参敏感且引入延迟，是个 trade-off；test-time scaling 也能显著激发 grounding 能力，但如何低代价激发还是开放问题。端云协同在 AndroidWorld 上相对纯端侧模型有约 33% 的相对提升，云端调用比例减少超过 40%，约 40% 的任务可完全在端侧完成；错误总结模块单独带来约 6.9 个绝对点的提升。
4. **结论/观点**：讲者的未来清单很能说明这支团队的判断优先级——agent RL 的训练效率与长程优化（500+ 环境下训约 100 个任务仍然很慢）、credit assignment、RL 从 benchmark 到真实场景的泛化、环境 scaling、harness 之争（谁先做出被接受的 harness 谁就拿到回流数据，模型跨 harness 泛化差还得逐个 SFT 适配）、benchmark 与真机之间的 gap、安全隐私、持续学习与 memory。这些排序本身就是结论：瓶颈在 infra 与数据回流，不在再想一个新算法名。

## 关键数字

| 指标 | 基线 | 结果 |
|---|---|---|
| 模型尺寸覆盖 | — | 2B / 8B / 32B / 235B MoE，均为同尺寸 SOTA（发布时） |
| AndroidWorld（235B） | Gemini、Seed 1.8 一代模型 | 76.7 分，发布时 SOTA |
| ScreenSpot Pro（32B） | Gemini 3 Pro | 73.5 分，反超 |
| Gemini 3 Pro grounding 复现 | 报告 72.7 分 | 直接复现约 39 分，加 zoom-in 后回到报告水平 |
| 开源 grounding 数据指令质量问题占比 | — | 接近 1/4 的数据存在指令质量问题 |
| Online RL 并行环境 / 最大交互步数 | 15 步 | 500+ 环境，约 50 步 |
| 端云协同 vs 纯端侧（AndroidWorld） | 端侧 MAI-UI | 相对提升约 33%，云端调用减少 40%+，约 40% 任务全端侧完成 |
| 错误总结模块消融 | 无错误总结 | 绝对提升约 6.9 分 |

## 可迁移

- 对 coding data / RL infra 工作的直接可试点：「错误轨迹保留最大正确前缀 + action 级 judge 过滤坏步骤」可以直接搬到 agent 轨迹数据的清洗管线——整条丢弃失败轨迹是巨大的数据浪费，前缀保留加步级过滤是现成的数据利用率杠杆。UI-Ins 的思路也可迁移：与其给模型套通用 CoT，不如把「任务本身的描述维度」做成 reasoning 格式。
- Infra 视角：端到端 online RL 的判据值得借鉴——infra 先行、链路打通后训练变成「喂 query + 堆算力」的 computation driven 模式，才谈得上 scaling；同时他们的经验是并行环境数（500+）和可支持步数（15→50）这类系统指标对最终性能的影响不亚于算法改动。端云协同里「小模型兼任 monitor 并产出结构化错误总结再交大模型接管」，是一个可直接试的低成本纠错拓扑。

## 疑问 / 下一步

- UI-Ins 在 RL 后涌现的新指令角度具体长什么样、是真推理还是格式模仿，字幕只给了定性描述，想看 paper 里的样例与失败分析；另外 zoom-in 的超参敏感性与 test-time scaling 的成本曲线，团队博客里应该有更细的复现 recipe，值得对着自己的 grounding 评测复现一遍。

## 原文金句

> 「我们提供 query 之后，整个 online RL 的流程就很简洁地执行起来了……以 computation driven 的方式，它是会更有 scale 潜力的。」

> 「谁能够优先推出一个被大家接受的 harness……你很有可能拿到这个 harness 里面的大量回流的数据。」
