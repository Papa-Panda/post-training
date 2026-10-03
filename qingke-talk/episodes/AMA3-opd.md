# AMA3 — OPD 专题（青稞 AMA 第 3 期）

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/AMA3-opd.html

## 元信息

- 专题：青稞 AMA 第 3 期 · OPD（On-Policy Distillation）专题（六位嘉宾圆桌 + 观众 AMA）
- BV：BV1qKVd6XEA3
- 时长：约 02:20:30（圆桌约 112 分钟 + 观众问答约 28 分钟）
- 提炼日期：2026-10-02
- 主持：付宇谦（中科院自动化所博士生，导师赵东兵、朱元恒；Revisiting OPD 作者，字幕自述）
- 嘉宾（均据字幕自述）：
  - 叶天竺（微软亚洲研究院研究员；黑盒 OPD、on-policy context distillation）
  - 顾玉贤（清华大学博士生；MiniLM 一作，被多位嘉宾指为最早的 OPD 工作之一）
  - 何炳祥（清华大学计算机系直博生，导师刘志远；JustRL、PRIME、MiniCPM RL）
  - 李亚轩（上海科技大学本科生，清华 NLP 实习；Revisiting OPD 作者，与何炳祥合作）
  - 杨晨旭（中科院信息工程研究所博士生，导师林正；多模态后训练，OPSD/IOSD）
- B站链接：https://www.bilibili.com/video/BV1qKVd6XEA3/
- 字幕原文存档：本地 `transcripts/AMA3.txt`（2538 条，带时间戳）
- 说明：基于 B站 AI 字幕整理；英文术语（如 logits、reverse KL）以可辨者为准，工作代号（字幕作 COPT、IOSD、EXOPD 等）按读音转写，未逐一核对官方拼写。

## 一句话总结

六位 OPD 一线研究者的一致判断是：OPD 的算法本身已接近成熟、改良空间在收窄，它真正的未来位置是后训练流水线里的「合板」工具与持续学习的载体；能否超越 teacher 取决于是否有外部信息注入（多 teacher 互补、环境/context 反馈、与 RL 结合），而全词表 vs top-k、context 泄露、跨词表、长上下文噪声，是眼下最实际的四个坑。

## 开场：每位嘉宾为什么做 OPD

- **叶天竺**（微软亚研院）：组里有蒸馏传统（MiniLM 一脉）。他去年想通一件事：pretraining 本质是 off-policy learning，很难见到 0 到 1 的突破；on-policy learning 以及它背后的东西——self-improve、continual learning——可能都离不开 OPD。去年做黑盒 OPD，今年做 on-policy context distillation。
- **顾玉贤**（清华）：最早的 OPD idea 出现在 2022 年冬天 ChatGPT 刚火时，mentor 问「能不能拿 RL 来做蒸馏」，后来演化出 reverse KL 的形式；MiniLM 论文 2023 年 6 月挂出，「比 Google 那篇 on-policy 蒸馏早了九天，把他们 scoop 了」。当时的最大 insight 是 on-policy 的重要性，但没人买账——那时大家觉得 RL 太贵、都爱做 DPO。黑盒蒸馏当时想做没做出来（想 hack GPT 的 logits 没成功），后来被叶天竺实现了。
- **何炳祥**（清华）：数据出身（UltraFeedback、SFT/偏好数据、PRIME），去年被 Thinking Machines Lab 博客里的一个现象吸引：被 RL 训过的模型能被 OPD 迅速恢复 teacher 的性能。他自己的 JustRL 用 32 卡训了两周、三四千步，正好想试试能不能用 OPD 把性能全部恢复回来，由此入坑。
- **李亚轩**（上科大本科）：10 月听何炳祥聊起 Thinking Machines 的博客——相比 RL 效率高、相比 SFT 不遗忘——两人复现时「踩了非常多的坑，训了很多 OPD 失败的情况」，最后沉淀成 Revisiting OPD。
- **杨晨旭**（信工所）：多模态视频理解出身，今年初一大波自蒸馏工作涌现后才跟进。他的体感是自蒸馏「把它做有效，条件非常苛刻」，不同领域、不同数据可能需要不同方法。

## 圆桌问答

### Q1：OPD 2023 年 6 月就出现了，为什么三年后突然又火了？

主持人先抛此问。多位给出的归因高度一致，可以合并为五条：

1. **时代背景**：2023 年大家对「后训练」还没有清晰概念，顶多 SFT 加简单 RLHF/DPO，没人做大规模 RL；现在后训练已成显学（顾玉贤、叶天竺）。
2. **Thinking Machines Lab 的博客点火**：他们把 OPD 从 dense reward 的角度讲、和 RL 连起来，正好击中了大家对 sparse reward 的不满——好像 RL 的老大难有了一个新解法（叶天竺：「有一定偶然性，他们不一定想到能有这么火」）。
3. **RL 基建成熟**：早期没有 verl、ms-swift 这些框架，自己做 on-policy 很难；基建到位后能做的东西变多了（杨晨旭）。
4. **开源模型追上来了**：几年前开源模型远逊闭源，现在能拿到足够强的开源模型的 logits，白盒蒸馏才成为可能（何炳祥）。
5. **真实需求出现**：MiMo 的 MOPD 报告开源后，各大厂开始用 OPD 做多领域合板；加上大家希望模型能 self-evolve、从环境和历史经验里学——OPD 看起来正好能满足（杨晨旭、何炳祥）。何炳祥还补了心路：过程奖励（PRM）路线 24 年底碰壁（易 hack、数据难收集），DeepSeek-R1 把大家带回 outcome reward，做了一年又开始 rethink，dense reward 路线于是重新吸引注意力。

### Q2：蒸馏的上限被 teacher 锁死，OPD 能不能超越老师？

主持人的观点先行：可能不是对单个 teacher 的超越，而是 multi-teacher 之间存在互补（他在做多任务 RL 时观察到过领域泛化现象）。

- **杨晨旭**：EXOPD 一类工作已在 benchmark 上超过 teacher。想让学生超过老师，需要 OPD 提供一个正确的优化方向——teacher 给的方向对但幅度不够，手动把幅度放大就可能超过；但为什么能超，各家都还停留在猜想阶段。
- **顾玉贤**：HOPD 有意思，是在插值上做文章；参数插值/外推可能带来超越原始几个模型的能力，但原理不清楚，现象确实存在。
- **叶天竺**：如果 teacher 固定、一直训到收敛，学生最后只会变得和 teacher 完全一样。但存在一些中间状态：学生初始在某方面比 teacher 好一点，on-policy 的过程中可能保留自己的好处、同时学到 teacher 的能力——「不是很有保证，更多是实践中调到了这样一个点」。另一条是 context/self distillation：环境或人类反馈作为特权信息给 teacher，学生学完再当 teacher，这时已不存在固定 teacher，其实已超出最早 OPD 的范畴。
- **李亚轩**：纯 OPD 不太可能超过 teacher；要超过就得注入新信息——自蒸馏提供额外信息、OPD 与 RL 结合都能让模型走得更远；RL 和 OPD 串行穿插（RL 把熵降下去、OPD 再拉一下）能超过只做 RL。多个不同领域 teacher 互相指导、配合合适的 OPD 步频，能达到更高的综合平均（他提到团队的 COPD 工作：学生与教师共同进化）。
- **何炳祥**：更悲观一些。在比较广的 setting 上，SFT 或 OPD「甚至很难接近 teacher 50% 的性能」；目前只在「学生自己 RL 之后的模型当 teacher」时能接近 100% 恢复，multi-teacher 时不同能力不太冲突、偶有促进。只靠输出端 logits 分布对齐是不是一种高效的学习信号，他存疑。
- **顾玉贤**补了一个实验观察：拿模型自己蒸自己（A 当 teacher 蒸 A）也能涨点——本质是 on-policy 本身在涨，不是蒸馏在涨。主持人接：相当于 pass@k 不变，只是把 pass@k 变成了 pass@1，把自己的置信度拉高就能涨点。

### Q3：DeepSeek V4 用全词表 OPD，普通研究者有没有平替？top-k / top-p 够不够？

- **何炳祥**：他们在数学和代码上对比过 sampled token 与 top-k（k 尝试到 128）：数学上 sampled token 似乎已能达到接近 top-k=128 的效果，16K 长度内 k 的变化影响不大。他的保留：top-k 监督到词表里靠后的 token，可能偏离 on-policy 的本意——teacher 自己在那些 token 上都不确定，信号带噪；而且靠后 token 的 advantage 天然弱一到两个数量级，不 dominate 优化过程。实现细节上 teacher 的 log p 加权还是 student 的 log p 加权会带来优化方向不一致。
- **李亚轩**：Revisiting OPD 里算过——KL 的定义本身就决定了概率极小的 token 乘上去值也极小、advantage 近零、几乎不产生梯度，所以「top-k 勉强够用了」。top-p 的实现难点是维度不定（有时 10 个有时 20 个），张量维度不好确定。他的猜测：DeepSeek 用全词表是因为他们观察到 k 调小后训练不稳定，在 agentic 这类长程任务上不稳定会被放大，干脆全词表且 infra 写得好，直接绕开了这个问题。李亚轩还提到付宇谦他们发现：不稳定场景下把 top-k 拉高非常有帮助，但拉到一定程度收益有限。
- **顾玉贤**：力挺全词表。从优化目标上，全词表的期望严格等于 Thinking Machines 那个公式且方差严格更小，「是一个不亏的事情」；实现上存 hidden（几千量级）比存整个词表 logits 省得多，再加上 offload。对研究者他的实际建议是 JSD：它介于 forward 与 reverse 之间、有界、训练更稳；top-k 训久了容易崩（无论 forward 还是梯度都是有偏的）。如果工业级 infra 支持得好，还是全词表（reverse KL 或 JSD）更有道理。
- **叶天竺**：认同全词表在期望等价、方差小上的数学地位（Thinking Machines 用的其实是 K2 形式的估计，梯度与全词表 reverse KL 的 advantage 等价，是无偏估计），自己实验里全词表明显好于单点估计（收敛速度与最终性能皆然）；每个 token 的 head 向量都参与目标计算、都训到，可能也有帮助。
- **杨晨旭**：学术取向——top-k 性价比高，全词表显存要求高，这一块不发表更多观点。

### Q4：token-level 还是 sequence-level？折扣因子要不要非零？（credit assignment 怎么给）

Thinking Machines 的 loss 是折扣因子为零的特例，本质问题是当前 token 要不要为后续轨迹的成败负责。

- **顾玉贤**：试过 discount 非零，「没用，可以很明确地告诉大家」，知乎上也有人试过同样结论。机制上：只有紧邻的第一个 token 能推出等价的全词表形式，长程的估计方差太大，会抵消收益。他还观察到 reverse KL 数值在降、但 benchmark 不涨，只能作罢。
- **杨晨旭团队**推过上界：sequence-level 的方差会越来越大；但当 teacher 不完美、会在某个 token 后开始给错误反馈时，sequence-level 理论上能一定程度缓解——只是性能上没拿到明显提升。
- **何炳祥**：token-level 相当于一个有偏但方差紧得多的估计；上下文十几 K 时两者差别不大，到上百 K 的 agentic 场景「这件事直接决定了它有没有用」。Revisiting OPD 的另一个观察是：成功 run 里 advantage 主要落在 student 与 teacher 的 overlap token（双方都高概率的区域）上——相当于 KL 本身已对非 overlap token 做了隐式抑制，再显式加 discount 因子，收益比想象中小很多。

### Q5：闭源模型怎么做黑盒 OPD？

- **叶天竺**（自家探索）：student rollout 之后额外训一个 discriminator（类 GAN）区分 student 与 teacher 的 response，用它的分数当 student 的 reward。在偏 chat/数学的 setting 上确实能涨。但稳定性问题很经典：discriminator 太容易按 response 长度分辨两者，于是出现长度震荡（变长了拉短、变短了拉长）。再往深一层：这是 sequence-level 的稀疏 reward；想按 token 训 discriminator 给逐 token reward，「很容易训崩」。他的定性判断：做法 make sense，但「如何做好黑盒 OPD 可能还有更多的探索空间」，不是最终结果。
- **顾玉贤**：context distillation 本身就可以做黑盒 OPD——把特权 context 给模型自己（相当于让 Gemini 生成一段 prompt 再塞回去），再走 OPCD，就是黑盒的，自认为可以试。
- **杨晨旭**：提到微软一篇与最近的 Rubric OPD——用 teacher 当 rubricator，指出 teacher 与 student response 的具体差异、给权重，再让 teacher 当 verifier 打分加权。但他担心这条路「很容易发展到类似于 RL 里 reward model 的感觉」。付宇谦总结：怎么构造一个好的 rubric / context 空间，让老师能把能力传给学生，可能是黑盒的关键。
- **何炳祥**：把问题推广开——白盒里怎么用 logits 已有很多探讨（top-k、全词表）；更激进的问题还有跨 family、跨词表怎么对齐，甚至用 teacher 内部 attention 激活信号做蒸馏（block 的 hidden size、depth 都不一样时怎么办），这方面工作很少。核心目标始终是让学生无限接近、甚至超越 teacher。

### Q6：跨 family 的 OPD 怎么办？词表不对齐与分布差异怎么缓解？

- 付宇谦抛砖：NVIDIA 的 Lightning OPD 与 Revisiting OPD 都提到，分布差异大时先做一轮 warm-up SFT 让学生逼近，再做 OPD；但词表对齐这一块还没看到新工作。
- **顾玉贤**：很值得探索。HuggingFace 那个 GKD（字幕作「go的」）试过普通词表 overlap，「试完之后发现不是很 work，普通词表的 overlap 确实太小了」。他提到自动化所前段时间有一篇做这件事（李亚轩接话：就是 Revisiting OPD，「我们的做法比较 trivial，直接把（不匹配的部分）mask 掉了」）。
- **杨晨旭**：担心不同 family 的 thinking pattern 差距太大，蒸馏效果可能反而不如直接 SFT——他本人也不确定，想听大家意见。
- **何炳祥**：即使同词表，不同尺寸上 OPD 也不太 work。李亚轩展开了 Revisiting OPD 的出发点：1.5B 学生用自己 RL 过的同族 1.5B teacher 做 OPD，能以很低的成本恢复 80% 以上性能；换成同族、性能更高的 7B teacher（DeepSeek 蒸的 7B），OPD 过程却「一直停滞」，学生几乎学不到。换一组 pair（7B 学生对另一个 7B teacher、或 64B 量级）现象一致。这是他们想解释的核心谜题。
- **何炳祥**补刀：30B 以上的规模还没试过，结论可能有局限；而且 Qwen3 技术报告称其小模型都是 SFT+OPD 得到的，跨尺寸在工业实践里似乎又有效——这也是他的一个疑惑。

### Q7：Multi-teacher OPD 会不会专家冲突？与 mix-RL 比有什么本质区别？

两种 multi-teacher 要分开：每个领域一个 teacher 的多任务式，以及单领域多个 teacher 同时学。

- **付宇谦**：无论 multi-task RL 还是 multi-teacher OPD，teacher 之间的冲突或促进都很难在训练前预测，缺少一个可量化的先行指标，「目前也是对此比较困惑」。
- **叶天竺**：multi-task 式冲突弱一些（不同任务直接调对应 teacher 混在一个 loss 里训，正是 MOPD 要解决 mix-RL 效果不好的问题）；同一任务多个 teacher 冲突会更严重——同一条轨迹上 pattern 互相打架。而且单任务直接用最好的那个 teacher 就行，意义不大。
- **杨晨旭**（COPD 实验）：同一批多领域数据上，MOPD 确实优于 mix-RL，MiMo 的报告也提到 mix-RL 存在冲突。SDFT 这类先做一个任务、再做另一个任务的顺序式做法，遗忘也更少。他的解释是 OPD 更适合持续学习。
- **叶天竺**：合板的关键是利用 OPD 学得快——单独 RL 串得越多、忘得越狠；OPD 能用大约 RL 10% 量级的 steps 把能力学回来，训得少、遗忘就少。Thinking Machines 博客最后也是这个 setting：先 RL 再 OPD 回去。
- **RL 与 OPD 的本质区别**（叶天竺）：博客把 OPD 当 RL 讲、从 dense reward 切入，「更多是一个帮大家好理解的方式，或者说比较容易蹭到热度」。仔细看，OPD 更像在模仿一个分布，与 RL 的训练目标整体不一样。

### Q8：OPD 与 RL 在一个 pipeline 里怎么串？谁先谁后、怎么结合？

- **何炳祥**：工业界放法不一（NVIDIA 把 OPD 不一定放收尾，有人放最后合板）。他的统一看法：SFT、RL、黑白盒蒸馏本质上都是在对 policy 的分布做某种拉扯——SFT 更关注长尾 token（mode-covering），OPD/RL 是 mode-seeking，专挑高概率 token 不断放大；差别只是每个 token 概率该按多大比例放缩、乘什么因子。若基座容量与数据足够，各方法的收敛点可能都一样。
- **杨晨旭**（自蒸馏 + RL 的结合）：在多模态里直接用 OPSD 会撞上信息泄露（生成到后面不断复述特权信息，但推理时根本没有特权信息）。他的解法是解耦：方向用 RLVR 的正确答案把关，OPD 只改更新幅度——把加法改成乘法，用 evidence ratio 去缩放 advantage，缓解两路信号冲突。直接加权效果不好（两路梯度量级不同、超参难调）。
- 付宇谦帮何炳祥回忆：他们试过把 0/1 reward 直接加进 OPD（无任何加权），效果并不特别好。

### Q9：in-context OPD / 自蒸馏这条线怎么做？特权信息泄露怎么解？

- **杨晨旭**：为泄露问题想了很多办法都解决不好。SDPO 在附录里给过方案（生成时 mask 掉前几个 token），他们试下来还是泄露；改 top-k 等其他方法也一样——「至少在这个多模态推理场景下，它无论如何都会去泄露」。最后才走解耦路线：让 OPD 不直接改 student 的分布，而是间接参与改变幅度。
- **顾玉贤**：in-context OPD 若看成 continual learning 的一种形式，他站「训进参数派」——Claude 靠管理 skill 已实现一定程度的不训参数派持续学习，但训进参数这条路里 self-OPD/OPCD 是最有可能的路径。它解决的核心问题是「如何利用模糊的奖励信号」：RLVR 需要一个数值信号才能动，而 OPCD 可以把模糊的文本反馈通过 context 变成稠密信号。
- **叶天竺**：context distillation 适合拿不到 0/1 reward 的 setting（题目没那么 verifiable、信息以 textual feedback 返回时）。他澄清命名：很多叫 self-distillation 的工作其实和 context 没关系（先 RL 再 OPD 回来其实学的也是 context 里的信息），这个名字叫得不准。Context 为什么容易 hack？因为你假设「teacher 加了 context 分布就一定变好」，未必。具体经验：把 context 提炼成更高层、更抽象的 item（他们称 experiential knowledge——把学生轨迹先总结成高层指导意见）会明显更好；直接把学生乱七八糟的答案扔到前面，模型没被训练过去 follow 这种东西，teacher 诱导出的分布就是歪的。保证「context+teacher 构成的分布是有效的」，是 context distillation 与普通 OPD 的关键区别。
- **何炳祥**接话：这个问题可能不是 OPD 的问题、而是 prompt 的问题——存在 automated prompt optimization 这个领域（他提到今年 ICLR 有个叫作 GEPA? 字幕作「PORO叫JP」的工作）：优化出的 prompt 不能太 case-by-case，抽象的 prompt 效果反而合理，与叶天竺的经验吻合。

### Q10：研究 continual learning 缺环境/benchmark，小模型上的研究能迁移到大模型吗？

杨晨旭抛出困境：找不到好环境、好 benchmark、好数据集来研究 continual learning。

- **何炳祥**的观察：小模型在 context 利用上先天不足——把很详细的 context、完整实验 log、冗长的工具调用轨迹塞给小模型，信息越多反而越差（注意力分散），不给 context 反而可能做得更好；换 Gemini 3 Pro 这类大模型则做得很好。于是学术界在小模型上研究出的 dynamics 能不能迁移到前沿模型，「答案可能是否定的」。
- 叶天竺补充分类：一种是给一本书自学的 in-context learning 式「continual learning」，另一种是序列任务越做越好的参数学习式，两者并无严格定义之分；OPD 较适用第二种（一直有环境给反馈、持续交互）。
- 何炳祥把 taxonomy 再扩到三种：个性化（per-user LoRA 这类，OpenAI 的 memory 可能已够、是否需要进参数有问号）、知识注入（近几个月的新知识更新，前一阵子的老话题）、以及部署后吸收全新任务/OOD 数据（coding 是代表）；他认为第三种最有趣也最缺公认 benchmark，大公司内部怎么搞外人不知道。
- 共识收束：continual learning 的 bench 分数很大程度与模型的 context utilization 能力相关，开源与闭源模型在 follow context 上差距显著——换个强模型做同样的 in-context 方案，分数可能直接更高。

### Q11：OPD 接下来会怎么发展？（每位嘉宾展望）

- **叶天竺**：pipeline 里大概率是合板工具（RLVR 很难超越 oracle reward）；另一头是蒸小模型。学习方式上最看好 self-OPD/OPCD 处理回流数据：部署模型与真实用户交互收集的 transaction 很难完整 rollout 到 0/1 reward，而 OPD 可以在 partial trajectory 的中间就用 teacher（或 context-conditioned teacher）给信号——这是 OPD 可能超越 RL 的地方。
- **李亚轩**：OPD 已是成熟框架，MiniLM、Qwen3 的工业实践都在，能做的空间在收窄；他看好两个方向：一是杨晨旭那类 RL 与 OPD 结合的工作，二是合板（多 teacher 能力合进一个 student，COPD 那条线）。剩下多是小修小补解决具体困难。
- **顾玉贤**：押注 OPCD 对 continual learning 的帮助，本人是训进参数派；OPD 算法本身的改进空间不大（稳定性、刷任务都已见顶），更多是新场景应用（合板、持续学习）。
- **杨晨旭**：学界好做的是自蒸馏一类故事性强的工作；业界难啃但影响力大的是黑盒 OPD 与跨词表——做成了大家都会跟。
- **何炳祥**：先讲 limitation——dense reward 不是免费午餐：response 越长、prefix 越靠后，teacher 信号越带噪，teacher 对该 token 的熵本身就高时让学生去学反而把梯度学坏；长上下文 OPD 需要算法改进（他猜 Cursor 2.5 在 agent 场景做的是局部的 self-OPD，以避开长程的 reward drift）。未来落点仍在已有知识的传播而非未知边界的拓宽：工业后训练合板（目前看效率高于 mix-RL）、小模型的 SFT+OPD 可能成为标准配置。另外通用域没有 verifiable reward 时，能不能像 context distillation 那样把参考答案、环境反馈拼进 context 让模型学好，是一个开启新维度的可能。
- **付宇谦**（主持人补充两点）：一是文本反馈这个应用点——Cursor 在 2.5 之前没人展示过在工业场景把 self distillation 用好，Cursor 用它调整用户交互与模型行为，「我自己有个想法」：他们完全可能把用户在 Cursor 里的 Tab 接受/拒绝等行为，用 self distillation 蒸进 Composer 模型。二是效率——他们观测到只对 response 前 10% 做 OPD，效果或许就能接近全程 OPD：可能前 10% 就决定了后续 pattern；也启发 RL 在 long-horizon agentic 任务里是否只需 roll 前 10%、再让 OPD 估后续 reward 即可训练。

## 观众 AMA（第二环节）

**Q：PG-style OPD 与 GKD-style OPD 怎么看？KL 是放在 loss 里传梯度，还是放在 reward 里走 policy gradient？**
顾玉贤：MiniLM 很早就结合过这两种 loss。他的理解是两件事在期望上等价——无论走 loss 还是走 reward，sample 都是从 student 里 roll 出来的，仔细推 GKD 的东西就是等价于走 reward 的那条路；区别在于变种：reverse KL 有这个等价性，forward KL 与 JSD 没有，只能在全词表维度上做经验改进。

**Q：自蒸自己时 KL 等于零怎么退化？采样分布和训练分布是什么关系？**
顾玉贤：只是期望上等于零，实际采样出来的东西本身就带信号；训着训着就变成 off-policy 了。Thinking Machines 也做过这个实验：模型自己 roll 一堆数据再 SFT，越训越差，解释一致——采样出来的数据有信号，训几步后它就 off-policy 了。

**Q：OPD 要不要在 rollout 之后做拒绝采样，只训 reward 为正的轨迹？**
现场无人有正面结果可分享。李亚轩反而报告了一个反直觉的副产物（Revisiting OPD 期间、未及发表）：做 strong-to-weak 时换了个含很多难题的数据集（Demystify? 字幕作「DEMESS」，取难度高的题做 OPD）效果反而不好，用简单 prompt 做 OPD 效果更好；他不确定这能否泛化。

**Q：有没有一个可以在训练前就衡量「是否可 OPD」的指标，预估收益？**
何炳祥：定量预测「能恢复多少」还没做出来；初步指标是 overlap ratio——student 与 teacher 各自 top-k 的重叠程度、以及 overlap 里的概率质量和，能大致看出两者的 thinking pattern 接不接近。他想 claim 的结论是：学生与老师的 thinking pattern（分布）越接近，OPD 的提升越明显；具体提升幅度还需要更细的定量分析。

**Q：多专家 OPD 的主流训练方式是 mini-batch 混合还是顺序训不同 domain？**
主持人代答：应为 mini-batch 内的混合，不是顺序训练。

**Q：VLM 的 OPD 有没有好的实践或论文推荐？**
杨晨旭：可以直接用他们开源的 IOSD 脚本去复现；代码方向 SDPO 主要聚焦 code；纯文本/数学的可以试 EXOPD 的仓库。

**Q：OPD 的训练数据应该和老师、学生训练用的数据用同一套 query，还是不一样？**
李亚轩：NVIDIA 的 Lightning OPD 建议用同一套 query，Revisiting OPD 里的一个 reset（实验）也显示相同 query 效果会好一点；但有一个有意思的现象——相同 query 的熵会非常低，后面再做别的 post-training 不太好，所以建议混入一些 OOD 数据。

**Q：VLM 任务的 OPD 与 LM 的 OPD 相比，有没有什么不一样的发现？**
杨晨旭：没有很明显的差异。主要是生成长度——VLM 推理长度现在还比较短，纯文本长得多；有些方法在短生成上更生效，长了之后把上下文放进 prompt 的 OPD 方式效果会因长度变差。他自己的实验发现：在 VLM 上用轨迹作特权信息和直接用答案，最后做 OPSD 效果差不多，纯文本上则不太一样。「我个人感觉主要的差异还是在生成长度上。」

## 共识与分歧

达成的共识：

- OPD 这波热度是后训练显学、Thinking Machines 的 dense-reward 叙事、RL 基建（verl/ms-swift）成熟、开源模型可及性四者叠加的结果，算法本身 2023 年就已成形。
- 纯 OPD 训到收敛就是复刻 teacher；观察到的「超过老师」都要靠外部信息（多 teacher 互补、自蒸馏/context、与 RL 串行），或只是 on-policy 把 pass@k 压成 pass@1 的置信度效应。
- 全词表在期望与方差上严格占优、是工业级的正确解法；学术平替首选 top-k（性价比）或 JSD（稳定性），sampled token 在数学等短中程任务上也够用。
- token-level（折扣为零）足以胜任十几 K 上下文；上了百 K 的 agentic 场景，方差与信号衰减会成为决定性问题。
- OPD 的产业位置已基本锁定：多领域合板 + 小模型蒸馏 + 用 partial trajectory 处理回流数据的持续学习；与 mix-RL/model merging 相比，它合板效率更高、遗忘更少。

保留的分歧与未知：

- RL 与 OPD 是不是一回事：叶天竺认为博客的 dense-reward 讲法更多是叙事，两者目标本质不同；也有嘉宾（何炳祥）认为各方法只是对分布做不同方向的拉扯，容量与数据足够时收敛点可能相同。
- 强 teacher 反而教不动学生（7B 教 1.5B 停滞、同族小 teacher 却能恢复 80%+）的原因未有定论，与 Qwen3 报告的工业经验表面冲突，30B 以上未验证。
- 黑盒 OPD 没有公认解法：discriminator 路线有长度震荡与训崩问题；context distillation 被多人看好，但 context 怎么构造（抽象化、经验化）还没有方法论，只是经验之谈。
- 跨词表/跨 family 基本空白：词表 overlap 太小、thinking pattern 差异大到蒸馏可能不如 SFT，目前只有 mask 掉不匹配部分这类 trivial 处理。
- continual learning 缺公认 benchmark，且小模型上的实验结论可能根本迁移不到前沿模型（context 利用能力是隐含变量）。

## 原文金句

> 「他们不一定想到说这个能有这么火，有一定偶然性。」——叶天竺评 Thinking Machines 博客对 OPD 热潮的作用

> 「基本上把这个 OPPO（OPD）当作 RL 去讲、从 dense reward 去思考，更多是一个帮大家好理解的方式。」——叶天竺

> 「我感觉他确实也不是一个免费的午餐。」——何炳祥评 dense reward（长尾处 teacher 信号带噪、熵高）

> 「我们的做法比较 trivial，直接把（跨词表不匹配的部分）mask 掉了。」——李亚轩谈 Revisiting OPD 的跨词表处理

> 「主要的差异还是在生成长度上。」——杨晨旭评 VLM 与 LM 的 OPD 之别

*说明：本纪要由 B站 AI 字幕整理，字幕对英文术语、工作代号与人名有较多识别误差；嘉宾姓名均以各自字幕自述为准，术语与工作名按可辨范围转写，未确证拼写者已在正文标注。*
