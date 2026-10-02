# EP85 — OpenMoE 2: Sparse Diffusion Language Models
> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP85-openmoe2.html

> 「diffusion 和 MoE 能够互相增益，并不会互相损害，所以他们是一个 double win。」——讲者对 OpenMoE 2 核心论点的概括

## 元信息

- 期号：85
- 标题：OpenMoE 2: Sparse Diffusion Language Models
- BV：BV1rg1sBQESS
- 时长：01:06:03（讲授约 57 分钟 + Q&A 约 9 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：倪锦杰（NUS AI researcher；主持人开场介绍，字幕中其姓名另有「李俊杰」等识别误差，正式头衔以论文作者页为准）
- 相关论文：OpenMoE 系列（第一代为自回归 MoE 架构探索）；讲者团队关于「数据受限时 diffusion 语言模型超过自回归模型」的 Super Data Learners 工作；*Any-Order GPT*（讲授中引用其实验图）；OpenMoE 2 本体技术报告讲授时尚未放出（具体编号与版本以论文原文为准）
- 相关代码：将随最终 scaling 模型一并开源（checkpoint、训练日志、技术报告与代码，讲者结尾确认；以项目 GitHub 为准）
- B站链接：https://www.bilibili.com/video/BV1rg1sBQESS/
- 字幕原文存档：本地 `transcripts/EP85.txt`（1362 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP85，已存档）清洗提炼，信息来源为 AI 字幕原文（经人工清洗；个别专名可能有识别误差，如「A2」应为 AR、「EXPETRACE」应为 expert-choice、「MMMU」应为 MMLU、「apple」应为 epoch，均以论文与公开资料为准）。OpenMoE 2 的最终 scaling 实验在讲授时尚未训完，文中数字均为在研口径；凡讲授口径与论文版可能不同，以下标注（字幕口径）。

## 一句话总结

这期为 OpenMoE 2 做架构论证：讲者主张在「数据有限、算力相对过剩」的 regime 下，掩码 diffusion 语言模型（diffusion 损失 + 双向注意力）因为 any-order 建模、可调的训练/推理 FLOPS 扩展与天然的数据增广视角，会随多轮重复训练持续超过自回归模型；而 MoE 与 diffusion 恰好互相补短——diffusion 给 MoE 补上多轮训练所需的计算量、并让 expert-choice 路由第一次变得可行，MoE 则给 diffusion 补上参数扩展与 FLOPS/参数比的自由调节——六组模型的消融显示自回归 + MoE 在数据重复下是最差组合，diffusion + MoE 则如预期落在两个 dense 参照之间。

## 核心

### 出发点：为什么还要挑战自回归

讲者先给自回归（AR）记功：它成为主流不是偶然——训练时每个位置都拿到交叉熵信号、一次反向传播覆盖全序列，signal-to-FLOPS 比极高；推理时因为已生成的 token 不再改变，KV cache 让生成长度的服务开销线性增长，再加上 continuous batching、PagedAttention 一类对 token 级切分友好的技术。decoder-only AR 是多年探索收敛到的局部最优。

但它的两个性质也正是天花板：因果掩码强加了一个从左到右的归纳偏置，且参数量与计算量的比值几乎不可调（最 dense 的形态就是 decoder-only 本身，再往上只能靠 loop 类结构）。讲者的判断是：高质量数据总量有限（他估计约 10T–20T token 级别），数据增长近似线性而算力多维增长，未来必然是多 epoch 重复训练的局面——前沿实验室最大的模型实际也要训 5–10 个 epoch 才可能榨干数据价值——这时要比的就不是「单遍训练谁高效」，而是「重复训练谁先撞墙」。

### 候选架构：掩码 diffusion 语言模型

OpenMoE 2 本体是 decoder-only Transformer，组件（RoPE、RMSNorm 等）与主流 AR 模型基本一致，只换两处：把 AR 的交叉熵换成 diffusion 损失（简化后仍是交叉熵形式，但带重加权、对应一个 ELBO），以及把因果注意力换成完全双向注意力。训练时随机遮盖输入序列，只在被遮盖位置算损失；采样时可以任意顺序、任意位置并行生成，不必从左到右。

讲者给了 diffusion 在数据受限时更强的三个解释：

1. **any-order 建模**：双向注意力 + diffusion 损失等价于在建模同一序列的任意排列，自然数据（尤其代码、数据库、生物序列这类非因果生成的数据）本就不是严格从左到右产生的，AR 的因果偏置是额外加的建模难度。引用的 Any-Order GPT 实验图也显示：左到右建模自然数据仍然最好，但反方向建模的 loss 远不是随机水平——数据不是完全因果的。
2. **训练与测试时 FLOPS 扩展**：diffusion 达到数据上限大约需要多得多的训练 FLOPS（粗略估计一百多倍量级），上限也更高；每个生成步都在用双向注意力 refine 全部位置，且这个计算量通过 diffusion 步数可调可控——对照之下 AR 的 CoT 式计算扩展不可控（无法命令它何时停止思考）。
3. **数据增广视角**：一条长度为 $N$ 的序列可按遮盖方式切出 $2^N$ 种不同的训练样本，有限数据集被扩成天文数字量级，这解释了它为什么不怕多 epoch 过拟合；但讲者强调这一条只解释过拟合场景，前两条是整体性的架构优势。

团队前作的实证结论是：在数据量固定、多轮训练的设定下（且超参数还沿用对 AR 有利的常用值），diffusion 会在某个时间点后在验证 loss、下游任务与生成式代码基准上全面超过 AR；常用的抗过拟合手段（dropout、输入加噪）救不回 AR。另外两条工程性质：推理延迟低——diffusion 天然是多 token 并行预测（MTP 程度随步数可调，而 AR 上 speculative decoding 的接受率最多约 3–5 个 token），batch size 为 1 时延迟可以比 AR 低数倍到数十倍，商业 diffusion 模型已有 1000–2000 token 每秒量级的吞吐报道；这对 coding、robotics 这类低延迟场景是结构性优势。

### 核心消融：为什么 diffusion + MoE 是双赢，AR + MoE 不是

为什么给 diffusion 加 MoE？讲者的答案是两者正交且互补，并用一组六模型实验把这件事钉死。设定是数据受限、多 epoch 训练，六个模型分别是：AR 1B dense、diffusion 1B dense、AR 与 diffusion 各一个 MoE 版本（总参数 8B、激活 1B，分别与 1B dense 做 FLOPS 对齐、与 8B dense 做参数对齐）、以及相应的 8B dense 参照。预期是 MoE 的表现应落在两个 dense 参照之间。结果：

- **AR 侧预期破产**：AR 的 MoE 是三者中最差的，甚至不如 AR 1B dense。原因在 FLOPS：MoE 保持总参数不变却把每 token 计算量砍低，数据受限时本就吃不饱的模型更吃不饱，过拟合反而更严重；同理 AR 8B dense 在这个设定下还不如 AR 1B dense——数据固定时把 AR 模型做大只会更早过拟合。
- **diffusion 侧符合预期**：diffusion 全面超过所有 AR 模型，其 MoE 版本在两个基准与验证 loss 上都稳稳落在两个 dense 参照之间。

双赢的机制拆解：从 MoE 一侧看，diffusion 补上了多轮训练下 MoE 缺的 FLOPS；且 diffusion 一次前向处理一整批 token、没有因果掩码，**expert-choice 路由**（由专家来挑要处理哪些 token，而不是每个 token 挑专家）在 AR 上根本不可行（生成时一次只前向一个 token 无从路由），在 diffusion 上天然成立——它带来自适应计算（重要 token 分更多专家、不重要 token 少分甚至丢弃）、天然完美负载均衡（不需要 load balancing loss 与 dropless kernel），并避免最热专家成为吞吐瓶颈，讲者估计约有 15% 量级的吞吐收益。从 diffusion 一侧看，它的多 epoch 收益曲线几乎看不到递减（团队曾把 1B diffusion 在 1B token 上训到 500 个 epoch、等效 500B token 仍在涨），上限太高、收敛太慢；MoE 让它更快逼近上限，同时让 FLOPS/参数比可以自由调节——diffusion 步数管「参数不变、FLOPS 扩展」，MoE 的激活参数量管稀疏度——从而按下游任务是知识型（如 MMLU，吃参数）还是推理型（如 HellaSwag，吃 FLOPS）选密度。可行性另有单遍预训练设定的一组实验确认：diffusion MoE 同样落在两个 dense 参照之间，MMLU 更靠近 8B（MoE 的增益集中在知识），HellaSwag 离 8B 较远（推理任务要的是 FLOPS 而不是参数）。

### MoE 架构消融逐项（OpenMoE 2 的选型依据）

- **expert-choice vs token-choice**：OLMoE 在 AR 上的结论是 expert-choice 远差于 token-choice，OpenMoE 2 在 diffusion 上严格控制实验后发现两者相当，最终 scaling 版本选用 expert-choice。其归因有二：加 shared expert 后 expert-choice 的 token dropping 被兜底解决（OLMoE 恰好漏了这一点）；diffusion 的双向注意力让各位置激活分布更同质，路由没有 AR 里那种随位置漂移的偏置（如 attention sink）。顺带结论：shared expert 本身无论在 AR 还是 diffusion 上都有收益（把两个 routed expert 换成 shared expert 的一组显著更好），DeepSeek 用它不是没有道理，尽管 expert parallel 下 shared expert 要在每个 rank 复制、有额外开销。
- **专家粒度**：专家越多长期越好（256 个专家的曲线在 TensorBoard 原始曲线上下降最陡，平滑图会误导），但最终 scaling 只选了 64 个——专家数太多会让 expert parallel 的 all-to-all 通信带来约 2–3 倍量级的开销下降，得不偿失。
- **upcycling**：从训过 100B token 的 dense 模型复制 FFN 初始化 MoE，初期起点更高，但会被从零训的版本反超，结论是不值得做（与 AR 上的已有证据一致）。
- **前两层 dense（DeepSeek 技巧）**：在 diffusion 上没有看到收益，MMLU 反而明显变差（少了两层的 MoE 参数），最终所有层都用 MoE。
- **scaling factor**：shared expert 与 routed expert 的输出 norm 不均衡会互相埋没，用一个系数去平衡在原理上对长程 scaling 有价值，但在当前训练量下没有测到明显收益（讲者提醒这不能证伪其在 1T–10T token 量级的价值）。
- **batch-level vs sequence-level expert-choice**：batch 级全局负载更好，但会让不同用户的请求互相竞争，真实训练里若在乎请求独立性就不能用。
- **路由函数 softmax vs sigmoid**：性能相当，softmax 在 MMLU 上略好（diffusion MoE 口径）。

### 状态与 Q&A 要点

OpenMoE 2 的最终 scaling 模型在讲授时还在训练，架构消融已齐；讲者承诺训完后把全部 checkpoint、日志、技术报告与代码开源。Q&A 中他对「是否要等 AR 撞墙才会结构性转向」的回答是：数据撞墙其实已经发生（高质量数据只有约 10T–20T token），一旦想多 epoch 榨数据就必然要面对架构问题，而且这个问题不限于语言模型——其他模态数据更少、重复训练更普遍。diffusion MoE 与 AR MoE 在架构上的最大区别就是路由方式（expert-choice vs token-choice）；他判断 diffusion 的 FLOPS/参数密度更高、MoE 负责把这个密度调灵活，这是与 AR MoE 的本质差别。另外他明确说 MoE 层本身除 expert-choice 相关改动外没有大改，token dispatch、通信等与 AR MoE 相同，真要极致优化速度可以针对 expert-choice 再做。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| 六模型消融（数据受限、多 epoch） | MoE（总 8B / 激活 1B）预期落在 1B 与 8B dense 之间 | AR-MoE 三者中最差、甚至低于 AR 1B dense；diffusion-MoE 如预期落在两个 dense 参照之间 | 字幕 |
| AR 规模效应（数据受限） | AR 1B dense | AR 8B dense 反而更差（更大更过拟合） | 字幕 |
| diffusion 多轮耐力 | 1B token 数据 | 训至 500 epoch（等效 500B token）仍无收益递减 | 字幕 |
| 高质量数据总量估计 | — | 约 10T–20T token；前沿最大模型实际约训 5–10 epoch | 字幕 |
| diffusion 达到数据上限的训练 FLOPS | AR | 粗略估计一百多倍量级，上限也更高 | 字幕 |
| MTP 对照 | AR speculative decoding 接受率约 3–5 token | diffusion 天然多 token 并行、步数可调；商业模型约 1000–2000 token/秒量级 | 字幕 |
| expert-choice 吞吐收益 | token-choice + dropless 约束 | 约 15% 量级（避免最热专家瓶颈） | 字幕 |
| 专家数选择 | 256 个专家长期曲线最陡 | 最终选 64：专家过多使 EP 通信出现约 2–3 倍量级开销 | 字幕 |
| upcycling 对照 | 从 100B token dense 初始化 | 初期占优，随后被从零训练反超 | 字幕 |
| scaling factor 验证范围 | 当前训练量下无明显收益 | 讲者称可能需 1T–10T token 量级才显现 | 字幕 |

## 可迁移

- **先判 regime 再选架构**：单遍训练、算力受限时 AR 的信号效率仍是王；数据受限、要多 epoch 或数据本身非因果（代码、结构化数据）时，双向 + 任意顺序建模的边际收益系统性变大。给 RL 后训练选基座或做数据重复策略时，应把「还要把同一批数据过几遍」当成一阶变量，而不是默认一遍。
- **MoE 评估要同时报 FLOPS 对齐与参数对齐两个参照**：只看参数对齐会把「计算量被砍」误判成「稀疏结构不行」；数据受限场景下 MoE 的失败先查每 token FLOPS 是否吃饱，再谈路由设计。AR-MoE 在重复训练下垫底就是这笔账没算清的例子。
- **路由方式的选择受生成范式约束**：expert-choice 的自适应计算与免负载均衡在并行解码（diffusion、blockwise 生成）下才成立；若推理栈是逐 token 自回归，就不要为训练时的漂亮性质引入推理期不可行的路由。shared expert 则是低成本的通用兜底（token dropping、共享信息），两种范式下都值得默认打开。
- **密度可调性是一种产品能力**：知识型任务吃参数、推理型任务吃 FLOPS，能独立调节 FLOPS/参数比的架构（步数 × 稀疏度）可以按部署目标选模型形态，而不是训完才发现密度配错。

## 疑问 / 下一步

- OpenMoE 2 最终 scaling 模型的规模与基准结果讲授时未出（还在训）；diffusion + MoE 的优势能否从 1B/8B 量级保持到更大规模，是这条路线最关键的待验证点，等技术报告与 checkpoint 开源后核对。
- 「diffusion 全面超过 AR」的证据主要来自团队自家 Super Data Learners 设定的多轮重复训练；单遍、数据充足 regime 下两者的相对位置讲者只给了定性说法（AR 信号效率更高），没有同口径对照数字。
- 推理侧的账只算了延迟：diffusion 每步全序列双向前向的总吞吐与单位成本相对 AR + KV cache 的实际对比未给；15% 吞吐收益、2–3 倍通信开销也都是讲授口径的量级估计，需论文复核。
- expert-choice 的「重要 token 多分专家」在语言任务上按什么重要性定义、与 sequence-level/batch-level 的请求独立性冲突在真实 serving 中怎么处理，讲授未展开。

## 原文金句（1-2句）

> 「diffusion 和 MoE 能够互相增益，并不会互相损害，所以他们是一个 double win。」——讲授约 42 分钟处对双赢论证的收束（按干净口径转写）

> 「高质量数据也就 10T、20T 这个样子……当数据用完了以后，我们必然想去多（过几）个 epoch，那个（时候）必然就会考虑架构的问题。」——Q&A 中对数据撞墙的判断（按干净口径转写）
