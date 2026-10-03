# CUA — 面向动态环境的 computer-use agent 自主持续学习（ICLR 2026 CUA Workshop）

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/CUA-continual-learning.html

## 元信息

- 专题：ICLR 2026 CUA Workshop（青稞合集 sid=7971490）
- 标题：面向动态环境的 computer-use agent 自主持续学习：摆脱人工数据依赖
- BV：BV17Yo5BjEs1
- 时长：约 29 分钟（1718 秒）
- 观看/提炼日期：2026-10-02
- 讲者：字幕未自报姓名（自述为该工作所在课题组成员，此处不猜）
- 相关工作：ACULL（字幕口径，官方拼写未核）、CUA-Judge（字幕口径，构建于 WebJudge 之上）
- B站链接：https://www.bilibili.com/video/BV17Yo5BjEs1/
- 字幕原文存档：本地 `transcripts/CUA-CONTINUAL.txt`（677 条，带时间戳）
- 说明：基于 B站 AI 字幕整理；工作代号（字幕作 ACULL、酷阿 Judge 等）按读音转写，未逐一核对官方拼写。

## 一句话总结

benchmark 分数与真实部署之间隔着两个未被覆盖的维度——环境的多样性和动态性；讲者组提出 ACULL（字幕口径）：让 CUA 自己探索环境、合成任务、按课程调难度、用 CUA-Judge 自评，再迭代 RL 训练，全程零人工数据，在目标环境最高涨约 29% 且不发生灾难性遗忘（字幕口径，未指明是百分点还是相对涨幅）。

## 核心

1. **背景/问题**：OSWorld、WebArena 上 agent 涨势很快（OSWorld 一年前约 30 分已算高、如今开源五六十、闭源 70 多），但 benchmark 只覆盖冰山一角（OSWorld 12 个环境、WebArena 6 个站点，字幕口径）：Claude 3.7 在 OSWorld 近 40 分、换 ScienceBoard 只剩约 10 分；Online-Mind2Web 扩到 136 个网站后性能最高骤降约 60%。第二个维度是动态性：版本更新、UI 大改、平台迁移、分辨率变化都会掉分，平台迁移最高可达约 50%。人工标注不可 scale，所以问题变成：把 agent 放进目标环境，让它 on-the-fly 自主持续学习。

2. **定义与三大挑战**：continual learning = 在特定或多个环境上序列化学习不同分布的 task set、持续提升且无灾难性遗忘；分 intra-environment（同环境、任务分布变）与 cross-environment（多环境序列化）两种场景。挑战有三：任务生成（CUA 的多模态环境知识不在预训练里，简单 prompt 生成不出高质量任务，且任务必须绑定具体 context）、评估（多解、长轨迹、上百步截图输入，无法比对 ground truth）、基础设施（每个环境是一台重型虚拟机，要在服务器上稳定部署成百上千个）。

3. **方法 ACULL（字幕口径）**：
   - **任务合成**：agent 先自主探索环境收集 environment experience，再从 web 爬取软件初始状态（context），据此生成多样化任务；引入 curriculum——按当前正确率调制：简单题（正确率 70% 以上）加难、中等题换表面形式保逻辑与技能不变、难题分解为子任务/子技能。随着 iteration 增加，agent 完成任务所需步数变多，侧面证明能力与难度同步上升。人工抽检约 150 个（字幕「100 接近 150」）合成任务，94% 有效。
   - **评估 CUA-Judge**：在 WebJudge 基础上改造：先抽取任务的 key points（如「100 美元以下、骑士队、季后赛」），判定时要求显式输出是哪一步的 evidence 支撑每个 claim，压制幻觉；再加初始状态与最终状态的差异分析（改颜色、画曲线这类任务只需比首尾）。在 1400 条 OSWorld 轨迹上与 WebJudge 对比：agreement 更高、false positive 更少；另抽样约 300 条轨迹人工核对，94% agreement with human（均为字幕口径）。
   - **基础设施**：统一环境管理 server（一个 IP 批量创建环境、崩溃自动重启）；环境初始化单步要 5–10 分钟、GPU 会空转，于是做 environment preloading 与异步——policy update 时预加载下一步环境；按文档可在自有服务器或 AWS 部署 100 多个环境。
   - **训练**：合成任务 → CUA-Judge 给 reward → 迭代 RL，循环往复。

4. **实验与分析**：6 个环境（3 个 office、1 个 email、2 个专业软件如 KAlgebra），base agent 为 UI-TARS 与 Qwen3-VL（字幕口径）。intra-environment 上两模型均随 iteration 持续提升、目标环境最高约涨 29%，overall 不降甚至更高；对动态变化（版本/平台/分辨率）同样随训练缓解；cross-environment 序列化学习性能持续提高，且出现正迁移——没在 Thunderbird 上学过，其他环境学完后它也涨了。内部参数分析定义 sparsity（显著变化的参数比例）：LLM 与 vision encoder 都只有约 20% 参数显著改变，讲者认为这解释了 RL 为何不遗忘；且 vision encoder 的 sparsity 随层深增加（浅层对视觉决策更关键），与 LLM 各层较均匀的 pattern 不同。失败分析：主因是重复动作与环境知识不足，task misunderstanding 只占少数；case study 显示 grounding 出错后 agent 仍自称完成、缺乏 self-correction（填表下拉失败仍 claim 完成），以及死循环（删不掉就不断新建 session）。

## 关键数字

| 指标 | 数字 | 口径 |
|---|---|---|
| 未见环境性能塌陷 | OSWorld 近 40 分 → ScienceBoard 约 10 分；网站扩到 136 个最高降约 60% | 字幕（[05:25]、[06:09]） |
| 动态变化掉分 | 平台迁移最高约 50% | 字幕（[07:21]） |
| 合成任务有效性 | 约 150 个抽检，94% 有效；课程阈值：正确率 70% 以上加难 | 字幕（[16:00]、[17:00]） |
| CUA-Judge 准确率 | 约 300 条轨迹人工核对，94% agreement | 字幕（[19:19]） |
| 环境初始化成本 | 每步 5–10 分钟（靠 preloading 隐藏） | 字幕（[21:08]） |
| intra-env 提升 | 目标环境最高约涨 29%，overall 不降 | 字幕（[22:24]，百分点/相对未明） |
| 参数 sparsity | LLM 与 vision encoder 均约 20% 参数显著变化 | 字幕（[25:23]） |

## 可迁移

- 做 agent 自改进循环时，reward 先行：要求 judge 显式输出「哪一步的 evidence 支撑哪个 key point」并比首尾状态差异，能显著压 false positive——这套判定格式可直接移植到 coding agent 的自动验收。
- 任务课程按实测正确率三段调制（易则加难、中则换皮、难则分解），是零人工数据下维持学习信号密度的实用配方。
- 环境初始化 5–10 分钟量级的重型 sandbox 必须 preloading + 异步，否则 GPU 空转吃掉全部收益；部署 100+ VM 环境的 server 化管理方式对 RL infra 同样适用。

## 疑问 / 下一步

- 「只有约 20% 参数显著变化所以不遗忘」目前是相关性解释；换个阈值结论是否还成立、与 SFT 对照的参数变化分布讲者未给，值得对原文核实。

## 原文金句

> 「benchmark 和真实世界还是存在一定的 gap 的。」
> （讲者在给出 OSWorld→ScienceBoard 40 分→10 分的落差后；全场论证的出发点。）
