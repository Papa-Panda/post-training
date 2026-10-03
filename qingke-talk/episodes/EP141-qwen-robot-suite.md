# EP141 — Alignment unlocks Scaling：Qwen-Robot Suite 的设计原理和背后的思考

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP141-qwen-robot-suite.html

> 「在目前我们应该解决的一个问题——align the action space，然后你才去做 scale。」——讲者对三篇工作的总括

## 元信息

- 期号：141
- 标题：Alignment unlocks Scaling：Qwen-Robot Suite 的设计原理和背后的思考
- BV：BV114GV6LEMB
- 时长：01:23:25（讲授约 52 分钟 + Q&A 约 31 分钟）
- 观看/提炼日期：2026-10-02
- 相关论文：Qwen-Robot Suite 三份技术报告（RoboManip / RoboNav / RoboWorld，讲者口径，具体版本以报告页为准）
- 相关代码：未开源（Q&A 口径：企业客户可经阿里云对接内部测试）
- B站链接：https://www.bilibili.com/video/BV114GV6LEMB/
- 官网预告：https://qingkeai.online/blog/Qwen-Robot-Suite
- 字幕原文存档：本地 `transcripts/EP141.txt`（1546 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP141）清洗提炼。字幕识别误差较多，专名按上下文校正：RoboManip（字幕作「robot mania」)、RoboNav（作「BON/VON」)、RoboWorld（作「robot word」)、VLN、π0.5（作「派0.5」）、RoboTwin / RoboCasa / LIBERO-Plus / EBENCH（e-bench）、Qwen3-VL、Qwen-Omni、Aloha。讲者姓名与单位从字幕自述与主持人致谢中提取（主持称「熊辉博士」；讲者称底层用 Qwen base、企业客户经阿里云对接），正式头衔以论文作者页为准。数字均为字幕口径。

## 一句话总结

这期讲 Qwen-Robot Suite（一套三个模型：操作 RoboManip、导航 RoboNav、世界模型 RoboWorld）背后的统一猜想：LLM 之所以能靠 scale 吃下异构数据，是因为语言有统一的 token 接口；机器人没有——joint、末端执行器（EEF）、waypoint 的数值语义互不相同，不同本体的同一数字含义也不同，所以异构机器人数据混在一起预训练并不天然产生 1+1>2。讲者的解法是**先对齐 action space，再谈 scale**：操作侧做表征/运动/行为三层对齐（约 80 维统一动作槽位、相机坐标系下的 EEF 表达、结构化 prompt），导航侧因 waypoint 天然对齐而改对齐 input（语言 prompt + 语义参数），世界模型侧则把控制信号翻译成语言做粗粒度对齐。配套结论是一条评测纪律：预训练的价值只有在 OOD 设定下才测得出来，IID 仿真榜单上不用预训练也能超过 $\pi_{0.5}$ ，这类数字不能证明预训练有效。

## 核心

### 出发点：VLM 有物理直觉，但「会看」不等于「会动」

讲者先立一个观察：今天的 VLM 在预训练配方并不专门考虑具身数据的前提下，对物理世界已有相当的 sense——能认物体、判方位、做 grounding、看视频大致推断球接下来往哪滚。短板在另一端：给定 instruction 做精细动作控制（比如输出一个关节角那样的精确数值）天然难做。讲者的归因是接口问题：语言模型的知识是在统一 token space 里预训练出来的，预测目标自洽，所以能力能涌现；机器人的预测目标是控制量，而控制空间不存在一个统一的 space——哪怕是同一含义的量，不同本体、甚至同一本体的不同副本，跑出来的数值都不一样。在这种 space 里把多个数据源混起来训，你很难相信会像语言那样出现「合在一起促进 emergence」的效果。整个 Qwen-Robot Suite 就是沿这个猜想展开的三个验证。

### RoboManip：把 action 对齐做成三层工程

操作模型 RoboManip 只用开源数据（包括 human-to-robot 的 ego 数据），讲者拆出对齐的三个维度：

1. **表征对齐（representation alignment）**：预先给所有本体的控制量分配槽位，组成约 80 维的统一动作空间，每个槽位语义固定——不让一个维度同时预测 joint 和 EEF，不让一个维度同时预测手和夹爪，全部拆开。这样任何数据源在相同控制信号下给出的梯度所对应的「学习意图」是一致的，梯度不冲突，scale 才可能成立。
2. **运动对齐（motion alignment）**：数值层面的对齐。同一关节数值在不同本体、不同关节上含义不同；他们把原始 EEF 坐标系转换成相对于某个相机参考系（如腕部相机）的 EEF 相对变化。直觉是：相对于「我看到的画面 + instruction」，EEF 接下来该怎么动，这个关系跨本体有强一致性，从而减少不同数据集之间的系统性差异、提高预训练效率。多视角时以哪个相机为准由架构指定。
3. **行为对齐（behavior alignment）**：加一小段历史 action chunk 作为 context，让模型通过 history 的 state-action 迭代获得本体感知；再加结构化 prompt，把 embodiment、instruction 以及 speed、FPS 等不好对齐的信息先用语言写进去。

数据侧两件大事。一是 **human-to-robot（H2R）管线**：讲者明确说，从基模厂视角看，ego 数据成不成，决定了这件事该不该由基模厂来做——ego 若能作为预训练源，就意味着不依赖本体、不依赖在线交互也能涨能力。他们的管线不是 video generation 路线，而是偏传统的：标注手部关键点、做 retargeting、取 depth、分 layer、把人手抠掉、用目标本体的 URDF 把运动渲染回视频图层。背后还有一套清洗管线托底。二是**清洗原则**：instruction、video、action 三者的信息必须对齐（instruction 和 video 讲同一件事，video 和 action 做同一件事），否则混训时梯度互相打架。最终 recipe 是 robot 数据 + 更多 H2R 数据的混合，预训练后直接 post-train 部署，没有更复杂的关系。

### 评测观：IID 榜单测不出预训练，要用 OOD 尺子

这是讲者自认最想传递给社区的部分，标题口号是 "this is the real scaling test"。他们早期照前人惯例在 LIBERO、RoboTwin clean/random 等设定上评估，发现一件尴尬的事：**不做预训练、直接在 benchmark 上 post-train，超过 $\pi_{0.5}$ 并不难**；但他们很清楚拿 Qwen3-VL 直接真机部署就是不如 $\pi_{0.5}$ 。榜单和体感对不上，说明测法有问题——仿真测评不是因为 sim-real gap 而不对，而是训练集与评测集本质 IID，这种设定测不出预训练的价值。他们把 RoboTwin 的 clean-to-random 设定重新拎出来（训练只用干净桌面数据，部署时场景、物体、背景全扰动）: $\pi_{0.5}$ 能直接冲到将近 50 分，开源里宣称更强的模型只有 10 分量级，他们不带预训练也只有 20 出头。这把尺子与业界口碑对齐之后，他们扩出一组 OOD 评测：LIBERO-Plus、RoboCasa 等 unseen-sim、自建 instruction-following bench（随机场景布局 + 随机指令，训练集仍是标准 clean 数据）、cross-embodiment transfer（在 RoboTwin 的双本体训练集上训，去 UR/Franka/方舟等没见过的本体上测）。

在这把尺子下的结论：对齐 + 预训练后，OOD 各指标显著超过 $\pi_{0.5}$ ，IID 指标则基本不动；EBENCH 的 breakdown 里，涉及 background/instruction/object 泛化与组合泛化（所有扰动叠满）的设定下，他们是「基本唯一不掉点」的方案；双臂任务与 pick-and-place 的优势最明显，恰好对应开源数据里最多的两类（双臂与 pick-and-place 为主导），讲者认为这说明预训练能力与后训练能力之间存在很强的对应关系。另一个被讲者称为惊喜的现象：把跨本体数据按比例混进 RoboTwin clean 的 post-training 时，用统一 EEF 训则性能随训练不掉甚至上升，**不用统一 EEF 直接「炸掉」**；cross-embodiment 真机上 Aloha 类本体间迁移也成立，而 joint 空间下看不到明显差异（ $\pi_{0.5}$ 的 EEF 反而不 work）。最后他们用 held-out 数据集验证 scaling：MSE 这类代理指标看不出框架差异，但换成下游 sim 评测，随着数据变多，clean-to-hard 设定的性能斜率越来越陡；easy-to-easy 则完全看不到 scaling——**不用 OOD 设定，你会以为放多少数据结果都差不多**。

讲者还展示了一个「无法提前 overfit」的 demo：让 Qwen-Omni 充当测试员，实时根据盘面生成不可预知的测试指令、语音转文字后直接喂给端侧部署的机器人执行，连续测了约 6–7 分钟，把物体挨个抓了个遍（个别物体只做过单物体抓取的 post-training，环境整体未见过），成功率相当高，主要翻车反而来自测试员模型自己判错对错。

### RoboNav：action 天然对齐，那就对齐 input

导航（VLN）与 manipulation 相比有一个被简化掉的问题：不管什么本体，移动最终都可以用 waypoint 表示 action chunk，所以 action 侧天然对齐，是一个「更干净的验证 scale 的 setting」。它的异构性在 input 侧：自驾要安全合规行驶，跟踪/找物要响应语言指令；不同任务的相机数目、关注视角、历史帧权重都不同。他们的做法刻意保持简单（简单的东西才 scaling，也方便日后被 agent 当 tool 调用）：用语言 prompt + 可描述的明确语义参数，告诉模型当前任务该关注什么信息，不同任务配不同的 language tag 与 context 参数，把市面上主要的导航任务 all-in-one 进一个 VLN 模型。结果上他们最关注 EQA 与 instruction following 两项（EQA 基于 agent 框架做出），相对此前以端到端为主的方案有明显 gap；instruction following 的 OOD 性能随训练量单调变好，model size 的 scaling 趋势也干净。端侧模型只能做到 4B 量级，于是有了下一步。

### RoboCloud：把 VLN 当 tool，让 VLM 当大脑

讲者借 OpenClaw 的设计原则搭了 RoboCloud: VLM 负责复杂任务，VLN 作为底层指令执行的 tool，加上 markdown 形式的 memory notebook 与一组辅助 tool（如自写的 look-around，因为宇树官方狗没有环视）。现场 demo 是一段 20 分钟一镜到底、四十轮左右的真实交互：机器狗在咖啡厅里找一把遗落的绿色雨伞，途中弹道偏左撞了墙，agent 能识别「走到的地方不对」、转身回头继续搜，最后定位目标、double check 后向用户汇报。讲者的定性是：agent 框架下 error recovery 是很自然的，上层智能把端侧小模型的短板兜住了。

### RoboWorld：用语言做 action 的粗粒度对齐

世界模型这一篇直观上是一个 video generation model，但技术意图仍在 action 对齐：把 manipulation、driving、navigation、人类视频里的控制信号，通过一条 workflow 系统性改写成语义化的 action 语言——没办法在 output 的 action 侧对齐，就在 input 侧做粗粒度对齐，看这样训出来的视频生成能不能忠实反映 instruction 的特征与物理特征。评测相应地不拼画质，拼 instruction following、物理对齐、时序 consistency 与多视角一致性。对开源 robot/general 世界模型效果理想；与 Seedance 这类最强闭源视频模型比，讲者明说「不确定能比它好」——基模规模仍是硬道理。

### 总纲与 Q&A 中的边界

三篇报告想回答的是同一个问题，讲者把它抬到「具身领域的一个 first principle（至少是当下该先解决的）」: **align the action space，然后才 scale**；manipulation、navigation、world 三个方向用不同技术手段验证，目前都是正信号。Q&A 里讲者的几个明确判断值得记录：

- **数据规模现状**：他们只用了约 5 万量级的数据，离外部流传的 50 万、500 万还差很多；讲者认为现阶段数据 scaling 是更有收益的方向（与模型 scaling 相辅相成）。预训练是否吃干榨尽数据，以「下游 post-train 的 OOD 评测是否收敛」为观测指标，但也承认这可能只是 infra/architecture 的上限而非数据的上限。
- **触觉**：非常重要，很多问题加了触觉会简单得多；但还没有能稳健大规模使用的数据与硬件，暂时保持关注，判断未来一两年可能有相对好的方案。
- **latent alignment**：并非没想到——ego 数据他们试过 latent space 方案，做 demo 可以，放 OOD 不鲁棒，所以选了显式表征对齐；latently 对齐足够好他同样认可，未来可能走通。
- **进度/state 感知**：明确表示不太相信能 scale up，带一点上一代 dense reward / process reward 最后没成为标准方案的历史 bias；memory 类任务他倾向于交给 agentic 框架解决，而不是塞进模型。
- **云与端**：最高智能还是要依赖云端大模型，除非有不能上云的敏感信息；端侧只放高频的 system-1 式输出（导航托底也作为 agent 系统的 input）。
- **内外参**：RoboManip 需要提供相机内外参才能把链路转起来；不走内外参就退化为 joint 或普通 EEF 控制，性能「看起来也还好」。
- **工程杂项**：底层基座是 Qwen base（「推荐大家都用」）；部署时有一层 reformatter，把任意人类指令改写成底层模型熟悉的指令格式，太 OOD 的场景仍有问题；next-token prediction 与 action 两个输出头混训目前是兼容的，语言数据还能帮模型保住基础能力；gripper 与灵巧手拆成两个槽位（hand 转 gripper 容易，反过来会引入 bias）；移动操作短期靠两个模型拼、长期想靠 80 维空间里的 base 维度混训两路数据；diffusion action head 试过一次，没有突破性结论，沿用传统方案。

## 关键数字

| 指标 | 基线 | 结果 | 来源 |
|---|---|---|---|
| 统一动作空间维度 | 各本体异构控制量 | 约 80 维固定语义槽位 | 字幕（RoboManip 方法部分） |
| clean-to-random（RoboTwin 设定） | $\pi_{0.5}$ 约 50 分；宣称更强的开源模型约 10 分；自研无预训练约 20 出头 | 对齐 + 预训练后各 OOD 指标显著超过 $\pi_{0.5}$ | 字幕（实验部分） |
| 组合泛化（扰动全叠加）设定 | 此前方法均掉点 | 自研方案基本不掉点（字幕口径近乎唯一） | 字幕（EBENCH breakdown） |
| 跨本体数据混入 post-training | 不用统一 EEF「直接炸掉」 | 统一 EEF 下性能不降反升 | 字幕（实验部分） |
| 不可预知指令真机 demo | 单物体 post-training、环境未见过 | 连续测试约 6–7 分钟，整体高成功率（字幕口径） | 字幕（demo 部分） |
| RoboCloud 真机长程任务 | 一镜到底真机交互 | 约 20 分钟、约 40 轮 agent-本体交互，完成找伞并汇报 | 字幕（demo 部分） |
| RoboNav 端侧模型规模 | 复杂任务需大模型 | 端侧约 4B 上限，复杂推理上发 VLM/agent | 字幕（RoboNav 部分） |
| 预训练数据量级 | 外部流传 50 万 / 500 万量级 | 自研仅约 5 万量级（字幕口径），自认差距大 | 字幕（Q&A） |

> 注：字幕未逐项给出各 benchmark 的精确分数，表中只录讲者明确给出的量级描述；精确数字以三份技术报告为准。

## 可迁移

- **评测先行于 scale**：若你也在做机器人/具身预训练，先复查自己的 benchmark 是否 IID——讲者的经验证据是，IID 榜单上「超越强基线」可以在完全不预训练时取得，这种胜利对预训练配方毫无信息量；把训练集设为 clean、评测注入场景/物体/背景/指令扰动，scaling 曲线才显形。这一条对任何「大数据预训练是否有效」的消融都适用：代理指标（MSE 类）看不出差异时，下游 OOD 评测才算数。
- **统一接口先于堆数据**：80 维固定语义槽位 + 相机系 EEF 的做法，本质是给异构数据源定义一个不会产生梯度冲突的目标空间，然后混训才不「炸」。迁移到任何多源训练（不同机器人、不同标注体系、甚至不同任务）时，先审一遍预测目标的数值语义是否跨源一致，比先调配比更优先。
- **对齐不齐的东西先用语言兜住**：不好对齐的信息（FPS、速度、本体描述）先写进结构化 prompt，RoboWorld 更进一步把控制信号整体翻译成语言——语言是当下最现成的统一接口，代价是精度（讲者承认），收益是跨域可混训与可被 agent 调用。
- **系统分工**：端侧小模型做高频执行 + 上层 VLM/agent 负责记忆、纠错与任务分解，error recovery 从框架里免费获得；需要 memory 的任务优先外置到 agent，而不是指望单模型内化。

## 疑问 / 下一步

- 约 80 维动作槽位的具体划分、相机系 EEF 的归一化细节与 RoboWorld 的 action-language 改写 workflow，讲者都指向技术报告，本期未展开；尤其「多相机以谁为准、内外参以什么形式进模型」只给了原则回答。
- 讲者对 latent alignment 的否定结论来自半年前的内部实验；在显式对齐已被证明可 scale 的前提下，latent 路线若由更强的基座重做是否翻案，值得跟踪社区后续工作（讲者提到已有外部团队在 world model 上尝试类似方向）。
- 「预训练只用了约 5 万量级数据」是字幕口径，单位（小时/条数）未在字幕中点明，与行业 50 万+ 小时口径的对比需回报告核对后再引用。
- 组合泛化「唯一不掉点」、clean-to-random 各家分数均为讲者口径陈述，EBENCH/RoboCasa 等第三方复现数字待查。

## 原文金句（1-2句）

> 「严格来说是我们的一个猜想：那是不是如果我们能像语言空间这样把 action 的 space 也 align 起来，那是不是也能实现像语言空间这样的 scaling？」

> 「你如果不考虑 OOD 的 setup，其实你依然看不见这条曲线——你会看到好像我怎么放这个 data，最后结果都差不多。」（字幕口径整理）

## 提炼方式说明

本纪要基于 B站 AI 字幕实录提炼；字幕中讲者的口头禅与重复已清理，英文专名按上下文与公开资料校正，主持人问答环节只保留信息增量。凡数字未标「论文/报告口径」者均为字幕口径。
