# EP125 — 重探 On-Policy Distillation（OPD）：三类典型失败以及修复路径

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP125-revisiting-opd.html

> 「对于 long horizon 的 LLM post training，我们需要保持 OPD 是一个 local 的信号，此外也不要让它再是一个 sampled token 或者 one token 的 loss。」——讲者在总结中的判词

## 元信息

- 期号：125
- 标题：重探 On-Policy Distillation（OPD）：三类典型失败以及修复路径
- BV：BV1CMLg6rENd
- 时长：01:20:30（讲授约 50 分钟 + Q&A 约 30 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：傅宇谦（中科院自动化所；研究方向：大模型智能体后训练与强化学习；字幕中自述姓名，正式头衔以论文作者页为准）
- 相关论文：Yuqian Fu, Haohuan Huang, Kaiwen Jiang, Jiacai Liu, Zhuo Jiang, Yuanheng Zhu, Dongbin Zhao, *Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes*，https://arxiv.org/abs/2603.25562 （2026-03-26 提交，v2 2026-04-27）
- 相关代码：已开源（讲者 Q&A 中给出 GitHub 二维码；repo 地址待从论文页补录）
- B站链接：https://www.bilibili.com/video/BV1CMLg6rENd/
- 官网预告：https://qingkeai.online/blog/revisiting_opd-talk
- 字幕原文存档：本地 `transcripts/EP125.txt`（1381 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP125，已存档）清洗提炼，信息来源为 AI 字幕原文（经人工清洗；个别专名可能有识别误差，如金明达老师的名字在字幕中作「吉米巴/金鸣霸」、verl 框架作「VR」、AlfWorld 作「阿尔法word」，均以论文与公开资料为准）。论文（arXiv:2603.25562）仅作交叉引用；凡讲授口径与论文版不同，以下标注（字幕口径）。

> 关联：本期是 OPD 线的"失败模式与修复"一讲；154 期《Rethinking On-Policy Distillation》（THUNLP，arXiv:2604.13016）是同主题的后续系统化工作，且在其参考文献中引用了本篇（2603.25562）。两篇互补：125 回答"标准 sampled-token OPD 为什么失效、怎么低成本修"，154 回答"OPD 什么时候成/败、机制签名是什么"。本仓库 `opd-reading-list/papers/2026_revisiting-opd/` 收有本篇的阅读条目。

## 一句话总结

讲者把标准 OPD 还原成一个「目标—估计器—实施」三层问题，指出业界默认的 sampled-token 实施方式只保留了序列级 reverse KL 的一个脆弱代理：它在长 rollout 上同时存在信号失衡、教师反馈不可信与 tokenizer 不匹配三类失败模式；讲者通过引入教师 top-k 局部支持匹配（带重归一化与梯度截断），配合采样端 top-p 控制与特殊 token 掩码，把 sampled-token 与全词表 KL 之间的这一层监督重做扎实，并用单/多任务实验验证了其相对 sampled-token 的实际提升。

## 核心

### 引入：先分清 OPD 在优化什么

讲者按三层拆解 OPD：**目标**（整条回复分布的 reverse KL）、**估计器**（sequence-level vs token-level）、**实施**（sampled token / 全词表 / top-k 匹配）。关键点在于：业界默认实现用 sampled-token 的 log-ratio 代替序列级目标，一个位置只回传一个采样 token 的信号，长 rollout 上这个代理会失去教师监督的整体信息。

OPD 近期受关注也有现实原因：多家技术报告用它合并多个专家模型的能力（multi-teacher / MoPD），且 on-policy 训练对既有能力破坏小，后续有向持续学习迁移的价值。讲者同时点明 OPD 并不新（最早可追溯至 MiniLLM 一系的工作，Gemini 2.5 的技术报告也提到类似用法），近年的热度更多来自 post-training 流程对低成本、防遗忘融合方法的需求。

结论先行：token-level OPD 相比 sequence-level 反向 KL 并不等价——它是另一个（有偏）估计器；这一差异在短 rollout 上无大碍，在长 horizon rollout 上会放大成训练不稳定。

### 三个设计约束：为什么 OPD 不能只是"更认真地算 KL"

讲者在 talk 中直接列出约束关系，作为后续方法设计的前提：

1. **更新必须局部化**。序列级反向 KL 的梯度权重是整条轨迹 log-prob 差的累积，token $t$ 的更新会被未来 token 的奖励耦合，样本一长，方差随轨迹长度爆炸。token-level 局部更新在方差管理上更可控。
2. **教师监督的信息量不能太少**。每位置只给一个 sampled token，教师分布的其余信息全部丢弃；全词表监督最完整，但在 infra 成本上并不总成立。
3. **推理/训练的infra成本不能无限上调**。纯全词表 KL 对 infra 的要求很高，讲者点名 DeepSeek V4 是靠极强 infra 才把全词表塞进训练流程的；对普通团队而言，sampled-token 的成本优势是真实存在的，问题是这个代理丢掉的信息不可替代。

这三条约束决定的不是"选哪个目标"，而是"在固定算力内如何保留尽可能多的教师信息"——这是讲者后续方法的设计坐标。

### 理论分析：方差上界与未来耦合问题

讲者给出 gradient 方差的渐近界对比：token-level 更新的最坏情况方差约为 $O(T^2)$ ，sequence-level 约为 $O(T^4)$ ——注意这不是说 token-level 无偏，而是说它在方差可控性上结构性胜出。轨迹长达 16K 甚至 128K token 时，sequence-level 梯度的不可控性成为首要矛盾。

讲者引入折扣因子 $\gamma$ 把两种框架连续统一起来： $\gamma=1$ 时退化为完整的序列级估计， $\gamma < 1$ 时未来奖励的 coupling 被截断，token-level 更新可视作其在 $\gamma \to 0$ 的极限。toy 研究验证： $\gamma$ 越大梯度方差越高， $\gamma = 1$ 时轨迹漂移显著。结论有一定的工程含义：实际实现里没有免费的全序列信用分配，token-level 的"近视"是有意方差控制，不是 bug。

### 三类失败模式：sampled-token OPD 是怎么坏的

讲者在 slide 与总结中反复盘点的三类失效（Q&A 中他再次按这三条归纳，顺序一致）：

1. **单个 token 信号失衡**。sampled-token 监督在每个位置只看采样到的一个 token，正负 reward 的比例在实践中严重失衡：大部分采样点都落在「学生概率大于教师概率」的区域，loss 以负信号为主，优化实质依赖少数局部正样本回传的梯度，而那些正样本常常只是填充词或连接词。这意味着优化器在大部分步沿错误方向被稳定推挤。
2. **学生前缀上教师反馈不稳定/不可信**。学生 rollout 一旦进入重复循环或格式游离出教师的熟悉分布，教师仍会在局部给出较高概率——讲者原话是猜测教师「发现前面已经有错误前缀之后就开始摆烂了」。这不是教师"笨"，而是前缀漂移后教师 next-token 分布本身不可靠。这类失败会随序列变长和累积误差显著加剧，讲者用 mismatch 与序列长度的关系实证了这一条。缓解路径包括：训练初期用 SFT 或 warm start 缩小师生分布 gap（讲者列举了英伟达 Lightning、自 OPD 等做法），或改用 sequence-level loss 让当前 token 对后续轨迹负责。
3. **Tokenizer 与 special token 不匹配**。同一段文本，学生与教师的切分可能不同（如 think token 的切分差异），教师会给这类 token「非常负的 reward」；更硬的情形是该 token 直接不在教师词表里。讲者给出的基础设施解法是 masking，同时提醒：硬训下去 tokenizer 切分有时也能被"搬回来"，但过程不稳定。

三类失败共性：都是**实施层**的偏差，不需要新目标就能改，也不需要什么玄妙调参——mask、top-p 控制与支持集选择都能各自压掉一块。

### 修复方法：将监督从 sampled token 换成 teacher top-k 支持集

讲者的方法（teacher top-k local support matching）核心替换只有一处：不再用采样出的 token 回传监督，而是在每个前缀下取教师分布 top-k 支持集，要求师生概率都限制在该集合内、各自重归一化（renormalization）后再做 KL 比较。

为什么选 **teacher top-k 而不是 student top-k**——这是 talk 中最有取舍感的设计论证：从 policy gradient 的角度，teacher top-k 内的 token 基本都带正信号（概率被教师认可），监督的方向明确；student top-k 则以负信号为主，只告诉学生"别把概率放这里"，缺少正向牵引。top-k 以外的 token 不参与 loss、直接截断、不分配梯度——讲者在 Q&A 中明确承认这条实现细节在正片里讲漏了，但它至关重要：学生被鼓励把概率质量往教师 top-k 内迁移，而不是在 top-k 之外乱跑。（一位提问者用 student top-k 或未截断方式复现时，直接 entropy 爆炸 + 训崩；讲者的解释是 top-k 之外的梯度处理方式决定了这条边界。）

重归一化在消融中是不可删除项：「去掉之后训练很快崩溃」。也就说明 teacher top-k 的关键不只是"看前 k 个 token"，而是把师生分布先约束进同一概率空间再比。讲者顺带提及金明达团队的相关工作：用 top-k + 长尾蒙特卡洛可得到无偏 KL 估计，更新版论文有更全的实验，附录里也有对该 KL 形式的复现。

三个实现小点被讲者单独列为实践要点（比算法本体更容易被低估）：

1. **重归一化不可去掉**（见消融结论）；
2. **学生采样端控制 top-p**，先降低采到教师不熟悉样本的概率，再让教师反馈（讲者最后给的经验值是 0.95 或 0.9）；
3. **特殊 token 与切分不匹配的 token 先 mask**，非 special token 同样建议先试 mask，仍不行再手工切到同一 token 空间。

### 实验：数字全部按字幕口径

实验分两部分：单任务数学推理（math reasoning）与多任务多教师（math + agentic 两域合到同一模型）。

| 观察项 | 字幕给出的口径 | 备注 |
|---|---|---|
| 单任务，AIME（without mask 配置） | 约 +10 点 | 相对 sampled-token 基线 |
| 单任务，平均分（without mask 配置） | 约 +5 点 | 多个数学基准平均 |
| 多教师设定，数学平均分 | 从 sampled-token 的 30 多分涨到 40 多分 | 跨过 40 分关口 |
| 多教师设定，AlfWorld | 约 +5 点 | agentic 域 |
| 训练过程 | accuracy 渐升、grad norm 无跳变 | 讲者用来论证训练稳定性 |
| top-k 经验取值 | 64 或 128（一百左右） | Q&A：太小退化回 sampled token、太大收益长尾 |
| top-p 经验取值 | 0.95（或 0.9） | Q&A 中以学生采样控制为准 |

注：旧纪要引用论文综合数字「+19.8%」——讲者在 talk 中没有给出这个数字，故已从表中移除；该数字属论文口径，talk 未展开。

消融提醒同样来自字幕：top-k 不要孤立评估——「top-k 单独拿出来不一定好很多」，它要配合更稳定的 rollout（top-p 控制、分布对齐）才完全发挥；把 sampled token 混进教师 top-k 支持集在单任务里没有正收益，在多任务里甚至会拉低效果（原始 top-k 支持集更好）；如果 infra 足够强、能直接做全词表 KL，讲者说那样「没问题」，top-k 主要是成本与稳定性的权衡。

另外两个小实验（字幕提及、未展开）：EMA top-k KL（附录实验，把 sampled token 也纳入支持集后可做无偏 KL 估计）、WebShop 等 agentic 任务。

### Q&A 要点（含讲者最坦白的边界）

- **多专家只学优势怎么做？** 讲者说自家的 multi-teacher 设定假设每领域一个 teacher，任务间冲突尚无解；建议要么在训练任务构成时先算好配比，要么引入稀疏但真实的环境信号帮模型选 teacher。
- **闭源模型也能当 teacher 吗？** 两条已知路线：把开源模型训成闭源模型分布的近似，再做 logit 级 OPD（董立/魏福如等人的 black-box OPD，字幕音译）；或让闭源模型逐 token 给学生 rollout 打分（那更像 rubrics 而非严格 OPD）。
- **关键分叉节点怎么定位？** 讲者先把问题引向 RLVR 领域：entropy 与 uncertainty 可作为分叉判据；原理上可对高不确定位置加重损失——这不在本工作范围内，但方向成立。
- **teacher 应视作 reward model 吗？** 讲者坦白：两者区别「我理解没有区别」（字幕口径），不可验证型 reward model 也面临同样的 reward hacking 与分布偏离问题；当前 OPD 的稠密信号优势，就是 RL 里训练 reward model 远比组织 OPD 流程更费力。
- **学生模型怎么选？** 至少要有 SFT 之后的 instruct 级指令遵循能力；纯 pre-train 阶段模型无法被教师给出有效监督。
- **框架经验**：讲者说不考虑生产落地的建议是直接用 RL 框架实现 OPD（可以从 policy gradient 推导，只需替换 reward 信号；他们自己用 verl），全词表塞入对 infra 要求大。
- **训练崩溃排查经验**：把出现异常 step（loss 突刺、entropy 飞）对应的 rollout 样本和对应 reward 全部下载逐条看——教师是否在训练前期就对长推理重复样本给出错误信号，是 debug 的第一检查点。
- **讲者最坦白的边界**（在总结与 Q&A 中重叠出现，构成对整个方法的核心质疑）：**教师分布是否能直接对应任务成功率，后续实验发现两者并不匹配**。这是讲者自己列出的未决假设——如果教师本质上只是一个「不考虑 dynamics 的世界模型/reward model」，用它的分布做优化信号，能否真的提升任务完成率，尚无保证。他对真实 reward 信号引入 OPD 的探索（类 MiMo MOPD 的稠密+稀疏加权）目前「暂未见好收益」（粒度不够细、比例难定），环境信号与稠密 OPD 的混合比例是后续需要系统回答的问题。他的未来三点是：自适应支持集（top-p / uncertainty-aware 选集）、「何时用教师匹配、何时必须引入环境奖励」的判定方法、把 OPD 抗灾难性遗忘的特性接入持续学习。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| 梯度方差最坏情况上界（token-level） | sequence-level $O(T^4)$ 对比 | $O(T^2)$ | 字幕（理论部分） |
| 分析的轨迹长度上限 | — | 16K–128K tokens | 字幕（理论部分） |
| toy 研究结论 | $\gamma = 1$ （序列级） | $\gamma$ 越大方差越高， $\gamma = 1$ 时轨迹明显漂移 | 字幕（理论部分） |
| 单任务 AIME（without mask） | sampled-token OPD | 约 +10 点 | 字幕（实验部分） |
| 单任务平均分（without mask） | sampled-token OPD | 约 +5 点 | 字幕（实验部分） |
| 多教师数学平均分 | sampled-token OPD（30 多分） | 40 多分 | 字幕（实验部分） |
| 多教师 AlfWorld | sampled-token OPD | 约 +5 点 | 字幕（实验部分） |
| top-k 经验取值 | — | 约 64 或 128 | 字幕（Q&A） |
| 采样 top-p 经验取值 | — | 0.95（或 0.9） | 字幕（Q&A） |
| 旧纪要引用的综合数字 | sampled-token OPD | +19.8%（论文口径，talk 未展开） | 论文 arXiv:2603.25562 |

## 可迁移

- 对 RL / post-training 基础设施的直接清单：实施 sampled-token/log-ratio 式 OPD 之前先查三件事——师生 cut（tokenizer / special token）是否对齐、采样端是否做了 top-p、教师支持集以外的 token 是否截断梯度。讲者的实验显示这些实现细节比换教师、换序列长度更先决定成败。
- 用 Infra 语言表述 teacher top-k 方法的等价物：它是在「每个位置只回传一个采样动作」与「整条序列全词表分布匹配」之间插入一层支持集约束的信用分配手段——监督带宽上调、但授权范围仍严格受限于教师置信区域。
- 调试纪律（来自 Q&A）：OPD 训练出现 loss/entropy 异常，先把对应 step 的 rollout 全量拉出来逐条看教师打分，不要先调超参；这是讲者团队验证有效的 debug 顺序。

## 疑问 / 下一步

- 官网提纲里的「后训练流程中的两层 Gap」具体指什么，讲者在 talk 中没有点名展开；他自列的三处不足（support-set KL 非全词表、rollout/training policy 训推差异、教师逼近不是任务成功率的完美替代）可能对应其中部分，但他自己说后者「并不是一个常见问题」（字幕口径）。
- Teacher top-k 中 K 的选取与序列长度、任务复杂度的关系：讲者给的经验数是 64/128（一百上下），但只在小规模实验里验证过；大规模、多领域场景下的收益递减拐点未给出。
- 字幕里提及但没展开的三个点：金明达团队 EMA top-k KL 的无偏估计实验（在附录）、WebShop 的 agentic 实验、二者与本工作的完整消融对比。
- 教师分布与任务成功率脱钩后，训练目标该怎么改写：讲者说未来可能把支持集换成 top-p 或 uncertainty-aware 集合，但还没给出判定框架；这是与 EP154「teacher-aligned prompts / OOD 保熵」思路可以直接对照的交叉点。

## 原文金句（1-2句）

> 「对于 long horizon 的 LLM post training，我们需要保持 OPD 是一个 local 的信号，此外也不要让它再是一个 sampled token 或者 one token 的 loss。」——讲者总结判词（字幕 48:59–50:16，按干净口径转写）

> 「教师模型或许能够看成是一个 reward model，或者是……不考虑 dynamics 的一个世界模型。那这个世界模型是不是真的能反映任务的成功率，这个是需要考量的。」——讲者在总结中对本工作核心假设的自我质疑

> 「大模型要 work 一定得每个细节都解决掉。」——Q&A 末尾被追问「三类失败哪个最重要」时讲者的回答：实际工程里每个实现细节的优先级都一样高
