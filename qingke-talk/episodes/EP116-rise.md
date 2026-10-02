# EP116 — RISE：组合式世界模型，在"想象"中实现机器人策略的自主进化

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP116-rise.html

> 「我觉得 world model 应该是这两年对 physical AI 最有用的一个技术路线。」——讲者在展望部分的判词

## 元信息

- 期号：116
- 标题：RISE：组合式世界模型，在"想象"中实现机器人策略的自主进化
- BV：BV1rnXhB9Ekf
- 时长：01:11:16（讲授约 50 分钟 + Q&A 约 21 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：任嘉智（香港中文大学 MMLab 博士二年级；本工作完成于李宏洋老师的 OpenDriveLab 团队；字幕中主持人称「任嘉智博士」，学衔以本人自述为准）
- 相关论文：RISE: Self-Improving Robot Policy with Compositional World Model（字幕原文作 "rise self improving robal policy with compositional word model"，正式题名以论文页为准）
- 相关代码：以论文项目页为准（讲者口述实验视频在项目网站）
- B站链接：https://www.bilibili.com/video/BV1rnXhB9Ekf/
- 字幕原文存档：本地 `transcripts/EP116.txt`（1404 条，带时间戳）
- 编号说明：B站视频实际内容为官网 EP116《RISE》，台账 B 行旧标题（KLong 相关）与视频不符，纪要按实际内容。

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP116，已存档）清洗提炼，信息来源为 AI 字幕原文（经人工清洗；个别专名可能有识别误差，如 world model 在字幕中通篇作「word model」、Genie Envisioner 作「jenny invasion」、value 基座作「派零五 / 判定舞」，均以论文与公开资料为准）。凡讲授口径与论文版可能不同，以下数字来源均标注（字幕）。

## 一句话总结

为了让机器人策略的 RL 不必在真机上付串行交互的代价，RISE 把世界模型因式分解成两件可独立升级的组件——像素空间里想象未来帧的 dynamics model，与逐帧打分的 value model（窗口首尾 value 之差即 advantage）——再在想象中做 offline 约束 + online 改进的策略优化；关键设计包括强迫 batch 关注动作而非背景的 action-centric batching、用 progress loss 加 TD loss 的混合价值学习，以及 60% offline / 40% online 的配比，在双臂长程任务上胜过 RECAP 与 DSRL（字幕口径）。

## 核心

### 问题：VLA 的 exposure bias 与真机 RL 的三重代价

现在的 VLA 模型在真机上常「差一点」：抓取差一点、方向偏一点，而且偏了之后不会恢复。讲者把根源归为 imitation learning 的 exposure bias——训练只见专家轨迹，推理时每步都不完美，误差单向累积。标准解法是 RL（在交互中学会纠错），但真机 RL 有三重结构性代价：交互是时间的线性过程、无法并行；需要人力全程看管防机器人进入坏状态；每个 episode 结束要人工 reset 场景。多数实验室也没有几十台同样的机械臂可供并行。

世界模型的老传统（World Models、Dreamer、TD-MPC，字幕口径）都是小模型（几十到几百 M 参数），只在仿真与简单任务上验证；近两年的「新时代」靠生成模型爆发（Genie、UniSim、Navigation World Models）把想象质量拉上来，但复杂 manipulation 仍未解决。RISE 的问题意识即在于此：用什么样的系统设计，让世界模型能用到真正复杂的操作任务上。

### 核心设计一：把世界模型拆成 dynamics 与 value 两半

RISE 不训一个端到端的世界模型，而是因式分解：dynamics model 只负责给 state 与 action 想象未来帧，不带任何评估信号；value model 再对想象出的每一帧逐步打分。讲者给了三条理由：

1. **两件事本质不同，可以各用最合适的架构与先验**：想象适合用预训练好的生成模型，打分适合用 language model / 机器人 policy 一类带知识的模型，两边的 loss 也可分开设计。
2. **不存在一个现成架构能同时把生成质量与 value 预测都做到最好**，硬耦合只会互相拖累。
3. **两条技术线都在快速迭代**，组合式架构能直接吃到任一边的进展——讲者举了 RISE 之后有人把新的 interactive world simulator 与 TOPReward 拼起来就得到理想 value 曲线的例子，并称这种「拼装红利」正是组合式的价值。

### 核心设计二：为什么在像素空间想象，而不是 latent 空间

这是讲者着墨最多的方法论判断（Dreamer 作者在 DreamerV4 里也从 latent 转向像素式想象，被他引为旁证）：

1. **能吃生成模型的预训练先验**：像素空间建模直接站在视频生成模型的肩膀上；latent 空间方案既没有机器人知识、也没有世界知识，基本从零训。
2. **可验证（verifiable）**：像素想象能直接看生成的未来视频，判断模型学好了还是学崩了、是交互不对还是背景不对；抽象 feature 空间里你无法判断训练走到哪一步——那些把 feature 再 decode 成图来检查的工作，最后看的还是像素，而 decode 出的图质量通常很低。
3. **慢不是真瓶颈**：像素生成确实慢，但生成加速（如 self-forcing 一类工作）进展很快，会被领域发展解决；latent 的轻量优势不构成长期壁垒。

讲者也承认像素里有大量对任务无用的信息，但反问：我们没有可扩展的办法预先界定哪些像素有用（不同任务要看桌面、桌周、桌下、纹理各不相同），一刀切地丢掉只会引入人的偏见——不如全建模，再靠可验证性去迭代。

### Dynamics model：基座、action 注入与一个关键的 batch 设计

实现上，dynamics model 选 Genie Envisioner 为基座（字幕作「jenny invasion」；其本身基于 LTX-Video 架构，一到两秒生成一段视频），做两处改造：从 text 条件改成 action 条件，并扩展到 multi-view 输入（机器人 policy 是多视角输入，这是刚需）。原基座没有 action 条件、纯快速生成版画质也一般，所以要「创造」action following 能力并提升画质：

- **Action 注入**：删掉 text 条件，加一个约 40M 参数的小 transformer 把 raw action 编码成 token 送入 DiT（Q&A 口径）。
- **训练语料与对齐**：在两个大规模机器人预训练语料上训（字幕专名作「AHIBOKBOARD / 格莱 XIA」，以论文为准）；不同 embodiment 的 action 维度对不齐就尾部零填充（与 π0 一类混合机器人训练同款做法）。输入是几帧历史观测加一个 50 步 action chunk，输出 25 帧一段的未来视频。
- **Action-centric batching（讲者点名的重要设计）**：常规随机组 batch 时，一个 batch 里塞满不同 scenario，loss 被背景与场景差异主导，模型忙着优化背景、action following 收敛很慢。改法是强组 batch：同一个 batch 只放少数 scenario、每个 scenario 放更多不同时刻的 action chunk——背景几乎相同，未来想象的差异只能来自动作，网络被迫学会「动作不同、结果不同」。讲者展示对照：无此设计时抓取方向完全偏掉，有此设计时方向基本对正。

预训练完做下游适配的成本很低：新 embodiment（实验用松林双臂机器人，字幕口径）只需 500–1000 条轨迹就能把预训练知识迁过去。而且这个世界模型 success 与 failure 都能渲染（演示里有夹爪没夹紧、拉链被弹开的失败想象），这是后面拿它当评价器的前提。

### Value model：progress 估计的病与 TD 的药

Value model 不用通用 VLM 零样本（讲者内部实测：现有公开 VLM 没在机器人数据上训过时，预测的 value 基本不对、最终还得喊停微调），而是以一个预训练机器人 policy 为基座（字幕作「派零五 / 判定舞」，专名以论文为准），它自带 robot-centric 知识且原生支持 multi-view 输入。每帧得到 value 后，advantage 取窗口末帧减首帧之差，即有 ， $A = V_{\mathrm{end}} - V_{\mathrm{start}}$ ，的口径（字幕）。

价值学习的关键观察针对 progress estimation（用「这一帧在 episode 中的位置」当 value 的 ground truth，0 到 1 单调）：

- 它只能喂专家轨迹（假设了起点 0、终点 1、单调上升），拿带扰动的数据推理时会输出一条非常光滑均匀的上升曲线——没有意义、没有判别力、极易过拟合。
- 讲者的解法是加一个传统 RL 里的 TD learning loss 与 progress loss 联合训练：progress 是强约束（正则），TD 在这个约束上学会让关键步的 value 该升升、该降降，从成功与失败两种轨迹里学判别。只用 TD 时曲线判别力强但抖动大；只用 progress 时平滑但迟钝；两者合用得到对关键动作敏感、又不抖的曲线。讲者说这个「过平滑」的毛病是在传送带任务（一个 episode 里要往盒子里放约四个积木）上实测发现的，加 TD 后解决。

### Policy improvement：offline 约束下的 imagination 内在线学习

策略改进分两段。**Offline 段**：把 advantage 作为条件输入（讲者称参考了 Decision Transformer 一类设计，字幕专名不清），先把策略的动作收敛到有效分布内——否则 online 阶段从随机动作出发探索，大量探索没有意义。**Online 段**：policy 输出 action chunk，组合世界模型想象未来并打分，得到 observation-action-reward 数据回训 policy；其中两个机制被讲者单独强调：

1. **双 policy 与 EMA**：产生数据的 behavior policy 与被优化的 policy 不用同一份权重，用系数 0.999 的 EMA 把 behavior policy 的权重缓慢同步给后者（字幕作「raw policy」，即 target 侧）。
2. **想象状态再利用**：世界模型想象出的未来 state 可作为新的起点继续 rollout，让 policy 见到「略微偏掉的 state」该怎么纠——但因误差累积，最多复用 2 次。

实验层面的两条 finding（字幕口径）：其一，online 学习必须有 offline 数据约束，**60% offline + 40% online 的配比效果最好**（讲者注明比例可能因任务而异）——原因是世界模型对特别坏、特别怪的 action 渲染不准，没有 offline 约束，policy 会输出怪动作、再被不准的 reward 带偏；其二，想象 state 复用（≤2 次）对结果有实际提升。

### 结果与定位（字幕口径，无逐项成功率数字）

在双臂机器人任务上，RISE（最右蓝色）胜过 RECAP 与 DSRL（RECAP 被讲者说明为 π0.6 的算法在 π0.5 上复现实现，字幕口径）。讲者对 DSRL 的评价很具体：它在抓、放这种短程任务上 RL 提升明显且快，但任务 horizon 一拉长（几十秒的动作）就很难收敛——因为 rollout 里需要同时有好动作和坏动作，长程任务初期几乎采不到好动作，数据分布只会越来越坏。RISE 的想象内训练不受真机串行约束，正是冲着长程去的。

### 边界与下一步

讲者在正片与 Q&A 里给出的边界相当坦白：

- **世界模型必然有偏**：它学在离线数据上、又是神经网络，对分布外 action 的想象不保证准；RISE 的全部 offline 约束设计就是为这件事兜底。物理错误方面，他称实验任务上未见穿模（生成基座的大规模预训练约束住了物理合理性），但小动作、没见过的 action 会糊。
- **不可预测环境（如人类行为插入）当前无解**：其他 agent 的行为独立于 ego action，极难预测；可探索的方向是显式给周围 agent 也加 action 条件一起训（他在自动驾驶里的经验：周围车行为同样难预测）。
- **下一步三件事**：sim 与 real 联合训练世界模型以扩 state-action coverage（仿真里可任意造状态与动作，把「坏动作对应坏未来」的能力迁回真实）；给 reward/value 做大规模预训练（点名 RoboMeter 一类工作，称其泛化强到对假机械臂视频也能给出合理 reward 曲线；RISE 里 value 还没有这层预训练）；从 task-specific 后训练走向 multi-task 的 generalist post-training（类比 LLM 的通用后训练）。
- **对世界模型的整体判断**：它可以贯穿 physical AI 的全链路——生成数据做 SFT、当 policy 表征（backbone 加 action head）、充当免真机的 evaluator（只需 CPU）、做 planning（采多个 action 选最优）、以及 RISE 做的后训练；被讲者称为摆脱真机依赖的可行路径之一。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| Dynamics 输入/输出 | 历史观测 + action chunk | 50 步 action chunk → 25 帧未来视频 | 字幕 |
| 基座生成速度 | Genie Envisioner（LTX-Video 系） | 约 1–2 秒生成一段 | 字幕 |
| Action 编码器 | raw action → DiT token | 约 40M 参数小 transformer | 字幕（Q&A） |
| 下游适配数据量 | 新 embodiment（双臂机器人） | 500–1000 条轨迹即可迁移 | 字幕 |
| Offline / online 配比 | online 学习配 offline 约束 | 60% offline + 40% online 最好 | 字幕 |
| 双 policy 同步 | behavior policy → target 侧 | EMA 系数 0.999 | 字幕（Q&A） |
| 想象 state 复用上限 | 误差累积约束 | 最多 2 次 rollout | 字幕 |
| 训练数据配比（专家 : policy rollout） | 世界模型训练 | 约 5 : 1（讲者称不必精调） | 字幕（Q&A） |
| 基线对比 | RECAP、DSRL | RISE 在双臂长程任务上更优（无逐项数字） | 字幕 |

## 可迁移

- **「想象环境内 RL」是 infra 形态的替换，而非算法替换**：RISE 真正换掉的是交互的物理载体（串行真机 → 可并行、可 reset 的像素想象），策略优化侧仍是 offline 约束 + online 改进的老结构。做 agentic RL 时同理：把环境换成可并行模拟的沙盒、并用真实轨迹做分布约束，是把 rollout 成本降下来的通用招式；60/40 的真实/想象配比可作起始参考，但要按任务复测。
- **Action-centric batching 是可直接搬的训练技巧**：当模型学不动某个因果变量时，先检查 batch 构成是不是让 loss 被无关变量（这里是背景）主导；强组 batch（固定无关变量、放大目标变量的差异）比改架构更便宜且常常更管用——同样的思路适用于任何「模型在学捷径」的场景。
- **Progress + TD 的混合价值学习**：单调 progress 是强先验但无判别力，TD 有判别力但抖——「强先验正则 + 自举修正」的组合，与在 LLM 侧用规则 reward 约束、再用模型 reward 细化的做法同构；且讲者明确说只喂成功轨迹的 progress 监督会训出「什么都给光滑上升曲线」的假 value，这一点对一切 progress 型 reward 都是警告。

## 疑问 / 下一步

- 双臂任务上 RISE vs RECAP/DSRL 的逐项成功率字幕未给（讲者为赶时间跳过了对比视频与结果细节），需查论文原文。
- Value 基座专名（字幕作「派零五」）、dynamics 训练语料专名（字幕作「AHIBOKBOARD / 格莱 XIA」）、offline 段参考的条件化策略设计专名，字幕识别均不可靠，以论文为准。
- 世界模型对分布外 action 的想象误差如何量化、何时该触发真机回采，讲者只给了「加 offline 约束」的定性兜底，没有给出误差监测指标——这是把 imagination 内 RL 做成生产系统时绕不开的一环。
- 讲者称 value 侧还没做大规模 reward 预训练（点名 RoboMeter 为方向）；RISE 后续若补上这层，60/40 配比与 TD 的必要性是否会变，值得追踪。

## 原文金句（1-2句）

> 「pixel space 这个建模是 verifiable 的……我们能通过它生成的 future video，判断出这个网络学得怎么样，是学好了还是学崩了。」——讲者论证像素空间想象的核心一条（字幕口径，按干净口径转写）

> 「我觉得 world model 应该是这两年对 physical AI 最有用的一个技术路线。」——讲者展望（字幕口径）
