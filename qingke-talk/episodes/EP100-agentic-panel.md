# EP100 嘉年华 · Agentic 专题圆桌（字幕实录版）

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP100-agentic-panel.html

## 元信息

- 期号：EP100 嘉年华专题（青稞Talk 第 100 期特辑分会场实录）
- 标题：Agentic 专题｜2025 “青稞” AI 嘉年华
- BV：BV1rriSBNEyh
- 时长：约 90 分钟（字幕末条 5404.0 秒）
- 提炼日期：2026-10-02
- 提炼方式：B站 AI 字幕实录（2481 条，`transcripts/EP100-AGENTIC.txt`）。发言人归属以字幕自我介绍 + EP100 官网嘉宾名单互证；字幕识别误差按名单与上下文订正，无法确定归属处写「嘉宾」
- 主持人：张桂彬（新加坡国立大学博士生，导师颜水成；官网名单）
- 嘉宾（官网名单 + 字幕自述互证）：张绍磊（中国人民大学助理教授，DataAgent / DeepAnalyze）、徐海洋（阿里通义实验室高级算法专家，Mobile-Agent 系列）、王鸿儒（爱丁堡大学博士后，Theory of Agent / self-evolving agent 综述）、潘家怡（UC Berkeley 博士，agentic coding 一线训练）
- 活动综述见 [EP100-ai-carnival.md](EP100-ai-carnival.md)

## 一句话总结

四位嘉宾对 2026 的判断高度收敛：能力会持续从 inference-time 的 prompting/framework 内化进模型参数，但 agent 作为「落地的最后一公里」不会与 LLM 简单划等号；scaling 的真正难点不是单维度堆量，而是 reasoning 与 acting、agent 与 environment 的联合 scale，以及 reward 的定义与规模化。

## 核心（按议题）

### 开场：四位嘉宾在做什么

- 张绍磊：从大模型算法转到 agentic training，做数据科学场景——把数据分析、编码、质量评估等能力用 RL 原生注入模型（DeepAnalyze 一类工作），目标是企业级海量数据的自动化分析与洞察。
- 徐海洋：多模态 GUI agent（Mobile-Agent 系列）。选 GUI 是因为它是日常生活里真正刚需多模态、且跨 PC/mobile/web/车机的开放长程场景；现在看到的买外卖、回消息还只是浅任务。
- 王鸿儒：做 agent 的理论（Theory of Agent）——关注三件事：scale reasoning 与 acting、scale agent 与 environment、构建 self-evolving agent（以 memory 为桥梁、前提是 self-awareness）。他判断 2025 上半年是 scale of acting（K2 为代表）、下半年是 scale of environment 爆发。
- 潘家怡：上半年在学校把 SWE、computer use、RL、reward model、多智能体都玩了一遍，下半年去业界做一线模型训练，主攻 agentic coding。她认为 2025 最大的事是大家开始系统化思考「怎样让模型变成 native agent」，而 RL 是今年才成熟的关键技巧。

### 议题一：LLM 未来会与 agent 划等号吗

- 潘家怡：基本已经是了。能在 prompting 层做出来的，说明基座有这个能力；转到 training time 做只会更好。framework 层的技巧没有 scale 保证，但训练可以持续投 compute、算法与数据，是更稳定的方法——通用能力（memory、planning、reasoning）会持续内化进参数。
- 王鸿儒：短期（半年到一年）可以划等号，长期不会。基础模型像基础教育、agent 像进入社会后的分工与成长；self-evolving 更可能先发生在 agent 层。且 agent 的市场（落地最后一公里）远大于 LM 本身，前提是好 LM 是好 agent 的前置条件。
- 张绍磊：别忘了 LLM 之外还有 LLM system——Cursor/Claude Code 受欢迎约一半来自模型外的系统与交互设计。agentic 能力会内化进模型，但 agent 的形态、与人的交互仍在模型之外（memory、infra）；更远期若模型能自己决定自己的 infra 与 memory 构建，那才是 AGI 形态，目前还远。
- 徐海洋：框架、agent 模型、基模三者相辅相成且交替迭代：agent 模型靠 workflow 回流数据、再迭代回下一代基模；GUI 这类场景还要端云协同、大小模型协同等系统级调度，离统一范式还很远。

### 议题二：Agent scaling law 最值得探索的方向

- 王鸿儒：三维度递进——先 scale reasoning 与 acting（单独都已被验证，难在同时 scale）、再 scale agent 与 environment 的共演化、最后是时间维度上的 self-evolving。三者同时 scale 上去，就离 AGI 不远了。
- 徐海洋：environment 至今没有明确定义（一个 App 算不算一个 environment？），定义不清就谈不上 scale。当前想做的是 mobile/PC/web 三端联动解题。数据是 GUI 的最大瓶颈：query 设计难、verify 难（工具调用好坏、路径好坏、反思好坏、泛化好坏都要判），纯靠人标无法 scale；动态环境里的弹窗、验证码、登录、推荐流都是真实难点。
- 潘家怡：不必神化——agent training 本质还是机器学习，无非算法、参数、数据（多样性 + 每类数据量）几个工具；系统性做事时工具箱全都要用，区别只在当前短板在哪。
- 张绍磊：各种 scaling 本质上都在 scaling reward。code 类 agent 领先，正是因为环境清晰、reward 易定义；其他领域要先解决「如何定义 reward 并让它通用」，包括用 world simulator 充当 reward 模型或直接充当 environment。

### 议题三：垂类 agent 会被通用大 agent 取代吗

- 潘家怡：agent 轨迹（调工具、等反馈、带 memory 的 loop）在预训练语料里天然 OOD ，比 chat 更吃专有数据。短期有好数据 + 好 infra 的垂类模型提升明显；长期（AGI 量级）可能不需要 specialized agent。
- 徐海洋：分两个维度看——金融/医疗这类「领域」垂类会被基座取代；coding、GUI、math 这类「原子能力」垂类会与基模长期并存，因为要求极高、需专项优化。且真实应用永远要算成本，1T 模型部署不到端侧，领域小模型有生存空间。
- 张绍磊：时间维度上一到两年垂类数据还有价值，五年或 AGI 之后会被替代。但垂类积累不是浪费——prompting 时代的 workflow 攒下的数据正是现在 agentic 模型的冷启动来源，ReAct 这类方法也在新范式里延续。
- 王鸿儒：做个类比——即便通用 LLM 取代垂类 LLM，通用 agent 取代垂类 agent 所需时间还要高一个数量级，因为 agent 的数据壁垒、复杂度、获取难度都再上一个台阶。内化的本质是减少与外部世界的交互次数而不降任务成功率（以 search 为例），但该调几次工具恰恰由领域专家知识决定，非专家拿不到工具、也调不对次数。

### 议题四：Self-evolving agent 会收敛到统一范式吗

- 王鸿儒：2026 年内不会收敛。evolving 可发生在 tool、memory、model、trajectory、skill 等多个模块，哪个先收敛还看不清；且收敛有前置条件 self-awareness——先知道自己的 knowledge boundary 在哪，才能有的放矢。他预判 2026 会看到更多 self-awareness 研究（包括 agent 层面的：该调哪个工具、调几次）。
- 徐海洋：自主进化涉及 agent 最本质的问题群：step-level verify 与定位真正的知识缺失点、把静态总结的 knowledge 用进动态环境、学习范式（context / RL / SFT 只有这几条路但怎么学住且可迁移未定）。以 OSWorld 为例，300 多个任务轨迹都公开了，大家仍刷不满——「学会 knowledge」本身极难。
- 张绍磊：收敛的前提是这套范式在各领域被验证过、有足够广的 benchmark 覆盖。历史上 reasoning 范式收敛靠的是 solid benchmark；self-evolving 目前只在 math/code 等好 verify 的任务上跑通，预计先在单领域内达成共识，再跨领域迁移，最后收敛成几大类方法；真正统一只有 AGI 时才可能。
- 潘家怡：一年内收敛很难，但可能出现「版本陷阱」式淘汰——大家弃用某些技巧、聚焦少数更通用、更易进生产系统的方向。

### 议题五：Agent benchmark 的不足与未来

- 徐海洋：定义问题比解决问题难。好 benchmark 三要素：定义清楚解决什么问题、与时俱进（GAIA 的答案已满网可搜、失去测评意义）、度量真实问题。AndroidWorld 公信力下降就是因为 App 不真实、环境是自研仿真，测不出弹窗验证码这类真实卡点；真实 query + 云手机/真机评测反而更可信。
- 潘家怡：agent benchmark 要求可复现环境 + 稳定且不可 hack 的分数，这本身就极大抬高构造难度。很多时候直接和模型交互得到的信息量比看 benchmark 分还高。
- 王鸿儒：environment scaling 可能带来本质解法——把 agent 本身当 world model / 环境：既能模仿人做 role playing，又能主动产生动态变化，静态 benchmark 的缺陷有机会被绕开；他们已在做把文本环境统一训成一个 text-based world model。
- 张绍磊：现有 benchmark 以模型为中心，但真实使用是人机交互共同完成（如 vibe coding）；未来需要把 human 引进评估——用 LLM 模拟人类用户与 agent 交互、看最终任务是否解决。

### 议题六：2026 展望（各自最想投入的）

- 张绍磊：自主性与自治性——让模型真正 autonomous 地作为生产力工具提效。
- 徐海洋：摆脱固定 workflow 模板，展示像人一样的问题解决（试 coding、试 search、试工具、甚至问人）。
- 王鸿儒：acting 与 environment scaling 继续推进；更期待同时 scale 多个维度的范式（agent–world-model 共演化），以及 agent 在各行业的落地、个性化与安全问题。
- 潘家怡：看更原生、更聪明的 agentic model 在各种 workload 上铺开——现有技巧还没用满，仅推进已知技巧就能走很远。

## 关键数字（圆桌口径，均为字幕）

| 数字 | 出处语境 |
|---|---|
| 2025 年 AI agent 市场规模约为 2024 年两倍、进入百亿量级 | 主持人开场（字幕口径，未核对） |
| K2 被理解为第一个 scale 到 1T 参数、原生支持 agentic tool use 的模型 | 主持人议题二引入 |
| OSWorld 只有 300 多个任务、轨迹已公开但无人刷满 | 徐海洋谈「学会 knowledge」之难 |
| agent 轨迹常达 50–100 步，放进 context 难以 follow | 徐海洋谈 step-level verify |
| Cursor 类产品的好感度约一半来自模型外系统 | 张绍磊估计（明确为个人感受口径） |

说明：本场为观点型圆桌，数字多为嘉宾经验口径，未做外部核对。

## 共识与分歧

共识：① 能力内化是大趋势（prompting → training），但 agent 层不会消失；② scaling 的难点在多维度联合与 reward 定义，不在单维度堆量；③ self-evolving 短期不收敛，self-awareness 与 benchmark 是前置短板；④ benchmark 必须真实、动态、防污染，人机交互应进入评估；⑤ 垂类 agent 短期有价值、长期被通用模型吸收，但积累的数据与方法会沉淀下来。

分歧：LLM 与 agent 的关系上，潘家怡最乐观（基本已划等号、内化即可），王鸿儒给出明确的短期 yes / 长期 no；垂类存续上，徐海洋按「领域 vs 原子能力」二分给出并存论，潘家怡更强调数据 OOD 决定的短期优势；对 scaling 的归因，张绍磊收束到 reward，王鸿儒强调维度间的共演化关系。

## 可迁移

- Agent 数据飞轮的既定路径已在本场被三方复述：prompting/workflow 攒轨迹 → 冷启动 agentic 模型 → 迭代回基模。新项目可直接照此排期，别指望一步到位训出 native agent。
- 评测先行：与其刷公开 benchmark，不如用真实 query + 真实环境（云手机/真机/真实 sandbox）搭小而可信的内部评测，并把答案防污染当成设计约束。
- GUI/多模态 agent 的第一瓶颈是 verify 而非 policy：query 难度分布、step-level 校验、动态环境异常（弹窗/验证码/登录）先有方案，再谈训练规模。
- 内化程度可以用「交互次数」量化：在任务成功率不降的前提下最小化工具调用次数，就是在最大化模型内部能力——可直接做成训练目标或评测指标。

## 疑问 / 下一步

- environment 的定义与计数单位（一个 App？一个任务分布？）没有共识，environment scaling 在此之前无法横向比较各家工作。
- self-awareness（知道自己的 knowledge boundary）被点名为 self-evolving 的前置条件，但会上没人给出可落地的训练或评测方案，值得跟踪王鸿儒团队的后续。

## 原文金句

> 「本质上大家都是在 scaling 这个 reward，就是在 scaling 激励。」—— 张绍磊

> 「发现和定义问题其实比解决问题更难。」—— 徐海洋（转述与 benchmark 同行的共识）

> 「（垂类 agent）你可能非专家的话需要调十次，但专家可能就只需要调一次……我画一条线可能只需要一美刀，但我知道在哪里画这条线。」—— 王鸿儒
