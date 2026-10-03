# EP100 嘉年华实录 — LLM/MLLM 专题圆桌

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP100-llm-panel.html

> **主旨引文**（薛复昭）：「一切用其他东西换 FLOPS 的东西，都是短期性 solution；一切用 FLOPS 换其他东西，都是长期 solution。」

## 元信息

- 期号：EP100 专题实录（2025 “青稞” AI 嘉年华 · LLM/MLLM 专题；嘉年华总览见 EP100-ai-carnival）
- 标题：LLM/MLLM 专题｜2025 “青稞” AI 嘉年华
- BV：BV1pziSBiE7f
- 时长：6005 秒（约 01:40:05；海报时段 18:00–19:30）
- 活动日期：2025-12-28（线上直播）
- 观看/提炼日期：2026-10-02
- 主持：程家乐（清华大学博士生，师从黄民烈教授；字幕自述作「陈佳乐」，姓名以 EP100 海报名单为准）
- 嘉宾（以字幕自述与 EP100 海报名单互证）：
  - 薛复昭（Google DeepMind Research Scientist，预训练团队；做 pretraining scaling law、data-efficient scaling、蒸馏与模型架构，入职 DeepMind 一年余。此前长期做 MoE 相关工作）
  - 刘子纬（新加坡南洋理工大学副教授；团队近期做原生多模态方向的工作，包括 NEO 等开源工作）
  - 谢天宝（香港大学博士在读；过去一年主要做 computer use、多模态 agent 方向的工程与 engineering research）
  - 曹宇（字幕中主持人称「曹宇」，自述在某厂从事多模态大模型后训练与 RL 相关工作；EP100 预告名单中本场仅列刘子纬、薛复昭、谢天宝及一位「神秘嘉宾」，曹宇未见于预告名单，与「神秘嘉宾」席位的对应关系未确证，见「疑问 / 下一步」）
- 相关论文：无单篇对应论文（圆桌讨论）
- 提炼方式：字幕实录版（B站 AI 字幕，`transcripts/EP100-LLM.txt`，2647 条）。AI 字幕识别误差原样保留在转写文件中；本纪要中的人名以 EP100 海报名单与嘉宾自述互证为准（字幕对薛复昭、刘子纬、谢天宝、曹宇均有多种误识），引文按字幕原文摘录、个别明显 ASR 错字在引用时以语境还原。数字均为字幕口径。

## 一句话总结

一场后训练 / 预训练 / 多模态三方视角的圆桌：长期看用参数换 FLOPS 的 MoE 会让位于 dense、预训练并未撞墙而是转向 data-bound、原生多模态与 LM/VM 统一是长期方向但短期仍受 infra 与组织制约、mid-training 的质量很大程度决定 post-training 的质量；而 RL 这边最缺的不是算法 trick，而是一条可外推的 scaling law、稳定的环境与可科学迭代的评测闭环。

## 核心（按议题）

### 议题一：模型架构——MoE、稀疏注意力与 dense 的长短期之争

- **薛复昭**（自述此前长期做 MoE，但长期看空 MoE）：MoE 本质是「用 memory、用 parameters 换 FLOPS」，而长期看 FLOPS 永远是最便宜的——芯片进步再慢也比数据来得快，所以一切用其他东西换 FLOPS 的都是短期 solution，用 FLOPS 换其他东西的才是长期 solution。他预测模型会逐渐变回 dense，甚至像 diffusion 那样变成 super dense；但这是渐变过程，不会一夜之间把 MoE 全干掉。他还引 OpenAI 在 GPT-4.5 发布时的公开表述：整个行业已从 compute-bound 变成 data-bound，这是社区共识。
- **曹宇**唱了个温和的反调：这个 tradeoff 更多是预训练阶段的事，而且今天的预训练已经不是「纯人类数据」的预训练——像 Gemini 这类模型可能从一开始就引入大量合成数据，等于已经在用 FLOPS 换 data 了。MoE 在当前带宽受限的硬件体系下是对大量用户友好的架构解法；他真正在意的，是预训练的内涵已经变了。
- **刘子纬**：长期稍站 dense 一边，但两者都有价值；现阶段要考虑部署场景，MoE 有其具体特点。更远的变量是学界在探索的下一代计算体系（存算一体、流水线重构），三到五年若有进展，会重新影响 dense / sparse 的选择；短期内仍是现有 GPU 架构，按应用场景取舍。
- **谢天宝**：模型架构研究已进入百花齐放（MoE 是很古早的话题），工业界 MoE 基本成熟，但架构选择应该跟着 learning paradigm 走——linear attention 本身就是受 learning paradigm 影响的产物；当我们重新思考 learning paradigm 时，会反过来重新设计架构。

### 议题二：原生多模态——pixel 直接进模型，还是 encoder 桥接

- **薛复昭**：原生多模态（直接把 pixel 变成 token、早期就混图文数据）从 infra 角度更干净：与 text token 几乎无区别，纯语言模型的 infra 不用大改。而 encoder 方案有两个麻烦：一是 encoder 用什么 target 训才能保证无损（captioning、CLIP、detection、pixel reconstruction 都不显然）；二是负载不均衡——image / video 出现的位置和密度不均，sequence parallelism 下 vision encoder 很难 shard，前段 image 密集的 worker 负载大、后段又在 idle，reshuffle 与 recompute 对 MFU 伤害很大。他倾向干净的模型结构。
- **刘子纬**（自述站 native 一边，团队在做 NEO 等开源工作）：现在 native 做得少，主要是缺一个所有人能用的 open codebase 或 checkpoint，主流模型都是 encoder / 桥接结构。native 的三个优点：数据上可以全局考虑、不用先训 ViT 再接语言；early fusion 的 interleaved 数据可能在预训练期就带来 emergability（他提到 Gemini 3 发布时有「重训了原生版本」的传闻，真假存疑）；infra 上一个干净架构可以到处复用语言模型的已有 infra。判断：现在关注度不够，但往后走是更 promising 的曲线。
- **谢天宝**：native 从 2023 年就在讨论，语境一直在变；各种 recipe 被反复捣腾，很难评判优劣。最终留下来的是工业上更好控制、可维护性与可拓展性更强的那个，还要看各家手里的数据；工业界工程能力还没跟上想法，没尝试够，现在下不了结论。
- **曹宇**（后训练视角）：对做后训练的人来说，多模态模型更多是 operator、是工具，很难影响到 test-time learning 这种非常早期的设计；但原生与否背后有个更本质的问题——未来模型的发展跟着数据的可获得性走。粘合式多模态需要的 text-vision 配对数据远远不够、获取成本又高，原生架构有更好的数据亲和性。他同时提出一个定义之争：输入端原生（text、vision、audio）大家没争议，输出端是否必须包含 vision 输出则各执一词——人除了受过训练的画家并没有 native 的视觉输出能力。他倾向更数据 native 的架构：数据上限决定能力上限，对后续 post-training 与 agentic 任务更友好。

### 议题三：LM 与 VM 为什么分开做，统一的收益在哪

- **薛复昭**：先提醒闭源模型是黑盒，未必真是一个模型。就算是一个模型，把 vision encoder 和 language decoder 放同一颗芯片上 serve 时，没有 image 的 query 会让 vision 参数的 memory 白占；更好的做法是 encoder 与 decoder 分开 serve、做 pipeline，language 部分可以跨模型共享，image 部分按需调用。结论仍是「会让整个问题更复杂」，但 serving 层面三方（原文如此，指可解）可解。
- **谢天宝**：分开做的原因和 coding、工具调用等能力分开做一样，是研发过程使然：原理上人人都想直接掏出一个全能 AGI，但路线图上必须先分开研究、再合版；合版研究去年推进了很多（如何把不同 capability 与 modality 合进一个模型、如何做 test-time learning）。现在处于中间版本，分开开发、分开发布，是工程上的滞后性。
- **曹宇**：VLM 与 LM 分开只是一个短暂过程。模型本身不 care token 代表的是 text 还是 video；组织上大模型研发已经不是单兵作战，成熟公司里所有模型都经历 branch out、合版验证再合并。被追问「统一上限是否更高」时他明确回答「一定」：真正有智能的操作对世界的理解不会局限于单一模态。现在之所以卡在单一模态，是因为 Transformer 在离散信号上的建模与数据实践远比连续信号（图像、audio）成熟。他判断输入模态的原生化没有问题（audio、text、image 乃至 long video），但输出端该包含哪几个模态，他自己还非常挣扎、没想清楚。
- **刘子纬**：补了两个现实因素：组织架构与工程因素之外，AGI 的 benchmark 仍然偏文字，很多事不太需要 vision、文字上就能达到不错的体感，所以合的动力不足；过去两年追求的更像 symbolic 的智能形态。往后从数字世界走向 physical AI 时天然需要多模态融合与 long-horizon 决策，具身任务上已经观察到不合的劣势。他用进化打比方：人是先有运动与操作能力、最后才进化出语言，我们做 AGI 是反过程——先把语言做好，再回去解 manipulation、action、planning 这些「低层」任务，而这些恰恰需要更多 sensory input。短期两边并行，长期走向统一。

### 议题四：数据——合成数据、mid-training 与多模态数据的可扩展性

- **曹宇**（先讲 mid-training）：他认为 mid-training 的划分至今不严谨，没人明确定义它到底包含哪些内容。早期雏形是把达不到 SFT 黄金标签质量的数据往前训练过程里扔，美其名曰提升指令理解；DeepSeek-R1 之后，大家才认识到 mid-training 高质量数据对 reasoning 与 agentic 任务的作用。他的核心观点：mid-training 在很大程度上是在做 world model 的建模——大量合成数据的训练方式（中间状态转移可见、loss mask 与 SFT 不同）本质上是在 given state 下建模现实的概率转移。RL 负责对已有知识技能的组合与泛化应用，原子技能必须先在 mid-training 阶段被模型理解和体会；所以 mid-training 的质量在很大程度上决定 post-training 的质量。
- **薛复昭**（先讲合成数据）：合成数据是个宽泛概念（RL 的 rollout、corpus rewrite、agentic 与环境交互的数据等），最核心的两个挑战：一是用当前模型生成的数据蒸馏下一代或更大的模型，gain 能否 up-transfer，直觉上和实践上都不好说；二是把 post-training / serving 阶段生成的高质量数据带回更早的训练阶段，会不会牺牲 post-training 的 gain，我们并没有很好的 justification 证明这些数据一定好、或者我们真的需要那么多高质量指令数据。模型足够大、memorization 足够快时，合成数据几乎不可避免地带来 model collapse 式的自我重复，对 scaling 是危险的。这些问题都需要很 careful 的 study。
- **刘子纬**：合成数据与 mid-training 和预训练脱不了干系——每代模型不停迭代，一部分 mid-training 的数据下一代就进到预训练里了，这是一个大的系统工程。他更感兴趣的是：如果下一代模型是完整的原生多模态，mid-training 会进来什么不一样的 signal。关于 mid-training 是否带来新知识，他引学界在小 scale 上的实验（包括 NeurIPS best paper runner-up 一类工作）：预训练基本奠定了 knowledge 的大分布，mid-training 只是把帮助往外推了推；但若 continual learning 变得更重要，mid-training 应不应该具备 learning-to-learn 这类不同能力，会反过来改变数据与训练目标的设计。
- **谢天宝**：给出一个流传很广的共识框架——post-training 是把预训练 / mid-training 里已经存在的 solution 拉回来，把 pass@64 拉到 pass@1（无论是 MC 采样还是 RL 强化）；如果前面阶段没见过足够好的数据，后面硬学只会损模型的分布。两个工程推论：post-training 能放的数据量有窗口上限，所以需要在 mid-training 阶段把能力补回来再反复迭代；数据该放哪个阶段、哪一类数据该不该要、该去找回哪一类数据，现在很多还是人在判断，缺一套科学的分析方法——active learning 的概念很早就有，但和工程师手搓主流模型的那套东西严重脱节。
- 多模态数据不可扩展的问题（主持：QA / caption 类数据难 scale、还带幻觉）：曹宇给出偏激进的看法——低等动物没有 text、没有 token 也在世界上活得很好，VLM 类模型可能不需要那么多数据打底，未来的学习应更多发生在与环境的交互中（RL 或 reward-free 的 learning from experience），但这是具身 / 小脑智能的场景，omni model 的路径他仍举棋不定。刘子纬主张回看十年前何凯明一代人做的纯视觉自监督：那套方法已被证明不能直接 work，但其设计 inspiration 可以替换或增强现在 pipeline 的某个环节；他举了自己团队做 visual jigsaw（打乱图像 / 视频再恢复顺序）的尝试，能同时激发 low-level 特征与 fine-grained recognition、视频识别、空间智能等 high-level 能力，而空间智能恰是 GPT-5、Gemini 这类强模型也做不好的地方，它需要的不是 text-image pair，而是关系（relation）的建模。薛复昭非常 buy vision-centric 预训练，但不完全认同小动物的类比：人和小孩学视觉感知很大一部分来自 physical touching，这是机器很难拿到的 signal，也是具身的 pain point；他绕开文本依赖的思路是 pixel reconstruction、next-frame prediction 这类视觉自监督，只是现在还不够 work，本质仍是 data-bound——算力再翻十倍一百倍，text-image pair 一定跟不上，长期他看好直接用 vision 硬搞。谢天宝表示这个问题超出自己的 scope，不做过多评价。

### 议题五：后训练与 RL——训出顶尖模型最重要的因素是什么

- **谢天宝**（从工程全链路讲）：RL 要具备的要素链条极长——首先要有一个好的 environment（硅谷有大量提供 environment 与 sandbox 的公司，需求差异极大，有的甚至激进到需要 replicate 一个 internet）；要有比较好的仿真（原文指基座）模型，mid-training 与 SFT 的 cold start 要做得还可以，模型本身还要有适合 RL 的「调性」和 foundation；再往下是 training infra：agent 任务的 context 显著比 IT 类任务夸张，inference infra、context management、agent workflow 架构、memory 设计、模型架构都要来回互相迭代；最后才轮到 RL 算法，而算法设计又要回头改 training infra 与 environment（如支持 trace 谱系之类的东西）。这些都做完也未必成功：调完 branch 出去，合版还可能不成功、还要再调。结论：这是很难在学术界整体研究的庞大系统，工业界的 know-how 也很难一次性获得，只能做、调、在实践里试。
- **曹宇**：调侃前面是「做 RL 的血与泪」，自己讲一个点：RL 从控制理论看是带着反馈信号的闭环控制，信号链路里每个环节都重要，而 RL 与预训练最大的区别在于问题选取——选哪些任务、怎么选、配比多少、任务间矛盾怎么解，这些目前依然是 human-centric 的，还没有进化到预训练那种什么任务都能 scale 到很大的程度。他认为 DeepSeek-R1 在过去一年是完全的高光时刻，成因很大程度上是问题选择与挑战定义之清晰。另外他预判 RL 正在快速专业化：做 coding 的 RL 你自己得是很好的领域专家，金融、法律、医疗同理，RL 从业者的背景不可能再只有算法。这与前几年「只有算法背景就能做 RL」已经完全不同。
- **薛复昭**（自述不是 RL 专家）：RL 一个蛮大的问题是没有一条非常好的 scaling law（至少开源角度没有）。开源发布的 7B、14B、70B 等模型往往是 train 好甚至 over-train 的成品，不在一条可以预测下一档的 scaling curve 上；小模型训完不知道大模型会发生什么，只能依赖最终的 run，又慢又贵，算法迭代与 debug 都很费劲。加上 environment 的 messy、训练不稳定（loss 上蹿下跳非常抖）、某个包更新就改变结果、异步更新引入的不确定性，即使 fix 了随机种子 RL run 也不是 deterministic 的，结论噪声很大。他希望的解法是：把整个东西科学地建一个 leaderboard、稳定地 improve，迭代自然会变得更容易、更好、更快。
- **刘子纬**：三点。一，RL 与 pretrain 分不开，很多 RL 现象只在某一类 pretrained model 上成立，想科学地研究 RL 必须把「怎样的 pretrain 导致怎样的 RL 特性」作为 assumption 放进来。二，infra 极其重要：做 self-reflection 类实验时，搭环境、并行采 reward 与 feedback、虚拟机、同步异步、不能拖累梯度回传，整件事需要大量 engineering effort，它本质上是一个 engineering 的大 framework，而不是设计一个 loss function。三，过去半年 reasoning 变种百花齐放，最近开始去伪存真（如 JustRL 一类工作）：很多 trick 不是那么有用，最朴素、最干净的那套把超参调通、trick 用足，就能达到最好的效果；经过半年的 bubble，大家能分清哪些进展是真的、哪些只在特定 setting 下 work。

### 议题六：开放式问题与 RL 的泛化之争

- **曹宇**（先声明不想站队，因为直播切片容易引发无结论的争吵）：他的基本感知是「RL 训什么有什么」，但不代表训了 A 其他方面就提升。一个流行的解释是 RL 只更新了大量参数中的很小一部分，甚至极低的 low-rank 更新就能打平全参调优。他更深一层的看法是：RL、SFT 与预训练本质上没有区别，都是策略梯度下降更新权重，没有道理说某种方法天生更能改权重。他观察到的是另一件事：现在的 benchmark 模型为 benchmark 牺牲了很多泛化性，在这个条件下观测到的 RL 泛化差被牵引着往一条路上走；但和 SFT 相比，RL 在很多实践场景里泛化性还是要好（SFT 本质是 offline 的一种 imitation）。他真正的保留是：RL 的 scaling 还远远没达到 promise 的量级，权重变化仍处在较小扰动范围，所以很难像预训练那样自然地把一个任务的能力迁移到其他领域。
- **刘子纬**：RL 的 task 与 environment 相比预训练的规模远远不够，还没到互相学习与涌现的程度；learning-to-learn 在 2018 年 meta-learning 那一波也没真正 work 过，没找到一般化的可学习梯度。他的要求不高：不追求泛化到完全 unseen 的 task，能在已知 task 矩阵内有一定的内插能力，就已经足够 impressive。
- **谢天宝**：verifiable reward 这块到底挖掘干净没有，他觉得没有，可验证与不可验证两个集合并起来也不是全集，每个集合内部都还没 explore 清楚，按他手头的工作估计至少还要一两年。他提到 Ilya 在 talk 里提的 value model：model-based RL 再加上 value model，明年可以再看一看。
- **薛复昭**：泛化是个相对定义——若以 task 为单位划分，RL 每个 task 要 sample 非常多的 trace（64 条、1024 条 sequence），这些 sequence 还彼此很像；从 FLOPS 与 token 的角度看，RL 在每个 task 上花的算力多得多，所以「泛化差」部分是对比口径问题。直觉上他认为 RL 对指定 task 的 generalization 应该比监督 / 自监督更强，真正的瓶颈是我们拿不到 trillion / billion 级别的 environment。

### 议题七：工具调用与 agentic 的下一步

- **谢天宝**：过去一年最大的突破是从短时任务走向多步长时任务，但决策质量还是很低——仔细看 trajectory，很多关键性 decision 是错的：deep research 的报告没有反映事实真相，coding 引入一个 bug 再修一个 bug、来回 debug 很多次，与人最大的区别就在这里。第二是 context / memory 问题：记忆如何表示、管理、利用，现在仍然是 2022–2023 年的 symbolic memory 范式；当 token 足够多时，在无损信息的前提下靠 context management 存储管理 memory 会撞上上限，这直接决定 agent 的用户粘性与可落地的场景。未来一两年要解决它，可能需要改 learning paradigm、改架构、改 agent 本身。按他的阶段判断，agent 从「可以决策」走到了「可以长程决策」，但决策质量是当前的核心短板。
- **刘子纬**：补充三点。tool use 的输出空间越来越长，要么靠原生模型安全地解一个 long sequence 的 output space 问题，要么靠 memory management（如何构建、update、调用 memory）。每一步出错概率随长度变大，self-reflection 与出错后纠错恢复的能力，是 tool use 走到真实世界的关键点——不可能每步概率都到 95%、99% 以上。第三，tool use 的泛化可能与 tool 的 functionality 而非 task 相关；未来几年大机会是 physical AI，在进 physical AI 之前要先把 digital AI 的 agent 解掉，而 computer use 里的图标本身吸收了现实工具的抽象，人能快速上手，agent 在数字世界学到的 skills 有没有可能 transfer 到真实世界、让机器人举一反三地用工具，是非常有趣的方向（大部分 robot 现在不会用工具）。
- **曹宇**（产品视角）：2025 最 impressive 的 agentic 软件产品是 Claude Code——长程 agentic 能力与各种想象不到的用法让他感觉是现象级产品，大部分 coding 服务商都把 Claude Code 作为对标，由此能窥见 Anthropic 对 agentic 与 tool use 的理解水位。硬件产品他举豆包手机：前几天很惊艳（能帮他刷抖音极速版赚几毛钱、再用微信转给朋友），过几天又变无聊（打不开微信、不能在拼多多和淘宝之间比价）——这正是 agentic AI 的技术水平与国内厂商对开放政策举棋不定之间的矛盾。他讲这两个例子的用意是：environment 在数字世界里 scaling、training 本就难，与其在 simulated environment 里无休止地训，不如把 agent 做成硬件或软件产品直接推向世界，让世界给它真实的 reward signal；不过从结果看 Claude Code 的 reward signal 好得多，豆包手机的大部分 signal 是负的——生活中的使用场景并不总像想象中那样 work。
- **薛复昭**：自述不是这块的专家，只讲偏好：他喜欢从 Claude Code 这类普通 agentic tool use 出发往外围走，gaming 是迈向 physical AI 比较重要的一步——Minecraft、魔兽乃至 GTA 这类很真实的 simulated 游戏里有大量交互，或许能从 3A 游戏 generalize 到真实世界的 physical task。

### 议题八：持续学习、长上下文与 agent 的最终形态

- **谢天宝**：continuous learning 现在要解决的是如何从很小的 batch、很弱的 message signal 里学出有效的梯度下降，再加上灾难性遗忘与恶意攻击等传统问题，它可能要求 learning paradigm、模型架构与算法一起创新。过去的设定是模型训完就 freeze、inference 花了大量算力却从不更新它——这中间有庞大的差别，是他认为非常大的一个方向。
- **曹宇**：从理论逻辑上，scaling 一定是用 FLOPS 去 trade 其他维度（FLOPS for intelligence、for 成功率、for continual learning），听起来自然且值当；但今年他自己踩了 continual learning 的坑：第一问是拿什么学（LoRA 还是 sparse 的 update 机制，变量非常多），还牵扯 serving——持续学出一个新模型，难道真能做到 production model 的 serving mode 吗？非常困难。但困难会变成有意思且值当的研究领域。他自承对架构的 sense 比较差，希望 2026 年能真正搞清楚哪些架构或方式能把 test-time training 做出来，而非停留在纸面推导与 toy example 的涨点上。
- **刘子纬**：continual learning 越来越重要，原因有三层。数据层：无论 text、agent 还是多模态数据天生都是长尾的，低频部分应该不动、只更新高频部分，这种更新对整个 dynamics 的影响需要研究。架构层：现在大家把大模型看作一坨 homogeneous 的参数，但它可能更像一个 OS、有不同区域——二三十年前就有人提出 slow weight / fast weight，有些参数慢更新、有些快更新、以不同层级 arrange，如何与现在的大模型范式相容很有趣。应用层：不同领域对 long-tail knowledge 的重视程度不同，有些场景必须在模型与数据层面处理长尾，它会反向追问整个领域的进展。
- **薛复昭**：lifelong learning 分 parameter-based 与 context-based 两条路。parameter-based 无非是搞个 LoRA、给 MoE 加一个 expert、或加一个 embedding bank，本质没区别；context-based 就是把 history 写成长 memory 再 retrieval。他直言第一种方法有点 crazy：模型越搞越大、stability 越来越差，还总在用很小的 batch 对它做反向传播，风险很高，时间长了用户体验反而出问题。更 safe 的是 prompt-based / context-based：唯一缺点是 context 比较长，但 context 长归根到底还是 FLOPS 问题。真要更新模型，反正几个月就会出一个新模型，周期性重训即可，不必搞那些花里胡哨的。

### 2026 展望（收尾）

- **薛复昭**：一，希望预训练出现比较大的 paradigm shift（这不是某一家能决定的，是整个社区一起决定）；二，个人想多学 RL 与 agentic 的东西，长期看这些方向重要、他也感兴趣。
- **刘子纬**：一，多模态会迈向更 unified、更一体化、更 native 的模型，NLP、vision、infra 出身的人都能贡献进来，形成更好的生态；二，相信会有一个新的大范式出现，可能从 language-centric 走向 vision-centric、多模态-centric 或 interleaved-centric，带来不一样的惊喜。
- **曹宇**：2025 对做后训练的同学挑战很大，在算法与 infra 之间反复横跳，也是一个快速学习成长的过程；2026 总体继续坚守把 RL 做起来。另外两个心愿：一是希望 DeepSeek 春节期间不要卷，不然春节又报销了；二是希望能多做一些更接地气的研究——通过产品本身与更接地气的实践去探讨 continual learning 这类问题（parameter-based、context-based 还是 embedding-based），比纯理论讨论更有价值。
- **谢天宝**：比较具体：把手头的东西接着往下推，research 上更偏一点，先做小范围实验与 follow-up，还是围绕 self improvement 与 continual learning，以及手头多模态 agent 那一堆杂事 concrete 地做下去。

## 关键数字

本场以定性观点为主，数字多为发言人举例时给出的示意量级（均为字幕口径）：

| 数字 | 内容 | 发言人 |
|---|---|---|
| pass@64 → pass@1 | post-training 的作用：把已存在的 solution 的通过率拉满 | 谢天宝 |
| 64 条 / 1024 条 sequence | RL 每个 task 采样的 trace 数量级（讨论泛化口径时举例） | 薛复昭 |
| 95% / 99% | agent 每步决策概率难达到的水平（出错不可避免，需纠错能力） | 刘子纬 |
| 几个月 | 新模型发布周期（周期性重训即可，不必在线更新参数） | 薛复昭 |

## 共识与分歧

**共识**

1. 预训练没有撞墙：头部仍在稳定前进，行业瓶颈正从 compute-bound 转向 data-bound（薛复昭主述，谢天宝、刘子纬未异议）。
2. 原生多模态与 LM/VM 统一是长期方向，但短期受 infra、组织与 benchmark 偏文字的制约，两边仍会并行（四人一致，曹宇给出「统一上限一定更高」的明确判断）。
3. RL 目前最大的短板不在算法 trick，而在环境、infra、评测与 scaling law 的缺失；朴素干净的方法把超参调通往往比堆 trick 更有效（谢天宝、薛复昭、刘子纬一致）。
4. 持续 / 终身学习是绕不开的大方向，但实现路径未定（四人一致认为重要）。

**分歧**

1. MoE 的长期命运：薛复昭明确看空（用参数换 FLOPS 是短期解），刘子纬长期稍站 dense 但强调部署与下一代硬件变量；曹宇认为 MoE 是当前硬件约束下的合理架构，争论的前提（预训练内涵）已经变了。
2. 持续学习用什么学：薛复昭明确反对 parameter-based 在线更新（风险高），主张 context-based；曹宇认为三种路径（parameter / context / embedding）都应通过产品实践去验；刘子纬提出 slow weight / fast weight 的分层参数安排。
3. RL 泛化差的归因：曹宇认为主要是 benchmark 导向与 RL scaling 不足（权重扰动太小）；薛复昭认为很大一部分是对比口径问题（每个 task 消耗的 FLOPS 多得多）；刘子纬认为 task 与 environment 的规模还没到涌现的程度，不必追求强泛化。
4. 多模态数据怎么 scale：曹宇认为应转向与环境交互中学习（偏具身场景），刘子纬主张复兴视觉自监督的关系建模，薛复昭看好 vision-centric 硬搞但强调绕不开物理接触信号的缺失。

## 可迁移

- **「mid-training 决定 post-training」的配方观**（曹宇）：RL 是对已有技能的组合与泛化，原子技能必须在 mid-training 阶段先被模型「理解和体会」——做 coding data / RL 数据时，可迁移的做法是把能力建模（中间状态转移可见的数据）前移到 mid-training，而不是指望 RL 阶段现学新技能。
- **RL 迭代的基础设施清单**（谢天宝、薛复昭）：environment / sandbox、适合 RL 的基座「调性」、training infra（context management、并行采 reward、同步异步）、以及一条可外推的 scaling 曲线与稳定 leaderboard，缺了这些算法迭代只能靠最终 run 又慢又贵地碰运气——这是搭 RL 训练框架时可以直接对照的 checklist。
- **agent 产品先行验证 reward**（曹宇）：把 agent 做成产品直接推向真实世界拿 reward signal，比在模拟环境里无休止训练更能暴露真实问题——对 agentic 评测的启发是优先在真实任务流里采反馈，而不是只维护模拟 benchmark。

## 疑问 / 下一步

- 曹宇的身份未确证：字幕中主持人称「曹宇」，自述在某厂做多模态后训练与 RL；EP100 预告名单本场为刘子纬、薛复昭、谢天宝 + 神秘嘉宾，曹宇与「神秘嘉宾」席位是否对应，待主办方后续材料核实。纪要中引用其观点时保留字幕称谓，不做身份断言。
- 薛复昭转述的 OpenAI「compute-bound 变 data-bound」说法出自 GPT-4.5 发布时的公开表述（字幕口径），未回原文核对；刘子纬提到的 Gemini 3「重训原生版本」传闻，他本人已注明真假存疑。

## 原文金句

> 「一切用其他东西换 FLOPS 的东西，都是短期性 solution；一切用 FLOPS 换其他东西，都是长期 solution。」—— 薛复昭（谈 MoE 的长期命运）

> 「post-training 是把预训练或 mid-training 里已存在的 solution 拉回来，把他的 pass rate 从 pass@64 拉到 pass@1。」—— 谢天宝（谈后训练的作用）

> 「我们先把语言给做好了，然后再回去去解那些我们之前觉得很 low level 的 task。」—— 刘子纬（谈 AGI 的反进化路径）
