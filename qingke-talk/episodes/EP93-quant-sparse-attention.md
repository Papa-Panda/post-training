# EP93 — 通过量化与稀疏性实现高效注意力机制

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP93-quant-sparse-attention.html

> 「AI Infra 的本质其实就是把所有的运算都变成 compute bound。」——讲者在讲 FlashAttention 为什么有效时给出的判词（字幕口径，下同）

## 元信息

- 期号：青稞Talk EP93（B站合集第 93 期实际内容）
- 标题：通过量化与稀疏性实现高效注意力机制（SageAttention 系列 + SpargeAttention + Sparse Linear Attention）
- BV：BV1d8UfB2EpP
- 时长：02:00:47（讲授约 66 分钟 + Q&A 约 54 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：张金涛（清华大学计算机系博士生，导师朱军、陈建飞；字幕自述分享时在加州大学伯克利分校访问、参与 vLLM 推理框架相关工作）
- 相关论文（均为讲授点名，具体版本以论文原文为准）：
  - SageAttention（ICLR 2025；INT8 量化注意力）
  - SageAttention2（INT4/FP8 更激进量化；技术报告首版 2024-11-07）
  - SageAttention2++（MLSys 2025 workshop 短文；FP8 矩阵乘 + FP16 累加器再提速 25%）
  - SageAttention3（首个 FP4 注意力前向 + 首个低比特注意力反向/训练探索）
  - SpargeAttention（算子级动态稀疏注意力）
  - Sparse Linear Attention（SLA：稀疏 + 线性注意力混合分解）
- 相关代码：SageAttention 已开源（ `pip install sageattention==2.2.0` ，一行替换 PyTorch SDPA）；SpargeAttention 已开源；SLA 的 SageAttention 扩大版讲者称将开源（分享时尚未发布）
- B站链接：https://www.bilibili.com/video/BV1d8UfB2EpP/
- 字幕原文存档：本地 `transcripts/EP93.txt`（2928 条，带时间戳）
- 台账说明：仓库台账旧登记本期为《Masked Visual Actions：自监督视觉信号学习》，与视频实际内容不符，已过时；本期按视频实播内容（量化与稀疏注意力）立目。

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP93，已存档）清洗提炼，信息来源为 AI 字幕原文（经人工清洗；专名识别误差较多：SageAttention 在字幕中作「SATTENTION/CATTENTION」等、SpargeAttention 作「Spider/Sparge」、FlashAttention 作「FLATTENTION」、FP16/FP8 在字幕中常作「IP16/IP8」、compute bound 作「computer bd」，均按公开资料订正）。凡硬件跑分、加速比与生态采用数字皆为讲授口径，下文关键数字总表统一标注来源「字幕」；与论文正式版可能有出入，以论文为准。

## 一句话总结

注意力是 Transformer 中唯一随序列长度平方增长的运算，在视频生成里占 80% 以上运行时；本期系统讲了在工业落地的两条正交加速路线——低比特量化注意力（SageAttention 1/2/3：从 INT8 到 FP4、从推理到训练）与稀疏注意力（SpargeAttention、SLA）——核心洞察有三：量化精度问题的根源是 K/Q 的 channel-wise outlier，而沿列减均值可在不改变 softmax 结果的前提下把它「免费」抹平；注意力图可以分解为「少量大值、高秩、适合稀疏计算」与「大量小值、低秩、适合线性近似」两部分，混合处理能在 30K 序列上做到 95% 稀疏度无损；量化与稀疏正交可叠加，商业上线模型中已实现注意力约 70 倍加速。

## 核心

### 背景：为什么单优化注意力

注意力运算的两个矩阵乘复杂度均为 $O(N^2 D)$ ，是 Transformer 中唯一与序列长度成平方关系的运算；序列一长，它必然吃掉绝大部分 latency。讲者给的视频生成例子：一次生成中注意力占 250 秒、其余运算（linear、norm 等）共 50 秒，注意力可被压到 5 秒而质量损失很小（字幕口径）。这也决定了优化收益的边界：端到端收益取决于注意力在目标负载中的时间占比——长 prefill、短 decode 的 LLM 场景与全程 dense attention 的视频/图像生成场景受益最大，短序列则不必。

两条路线的加速上限不同，这是讲者反复强调的选型逻辑：低比特量化的上限由硬件 tensor core 位宽决定（以 5090 为例，FP16 约 209、FP8 约 2 倍、INT8 约 4 倍、FP4 约 8 倍，字幕口径），且实测很难打满（SageAttention3 约 5 倍）；想再往上走，只能靠稀疏性真实减少计算量。线性注意力目前没有单独在工业场景落地的例子，通常要与 full/sparse attention 混合，加速被其中的稠密部分封顶。

### 先修课：FlashAttention 与 compute bound

讲者用两层 GPU 模型（HBM 全局显存 + SM 计算单元）解释 naive 实现为什么慢：三行写法要把 $N \times N$ 的 $S$ 、 $P$ 矩阵反复写回、读回 HBM，序列 100K 时既爆显存又把计算单元拖成等存储的 memory bound。FlashAttention 的解法是分块读入 Q/K/V、在 SM 本地用 online softmax 增量更新， $S$ 、 $P$ 从不落 HBM，整个算子回到 compute bound。讲者的判词是：AI Infra 的本质就是把所有运算（包括通信）变成 compute bound，让硬件利用率打满。本期所有后续工作都建立在 FlashAttention 式分块 kernel 之上。

### SageAttention 1：INT8 量化 + Smooth-K

低比特量化加速注意力此前没人做成，讲者归纳两个原因：一是注意力算子与硬件强耦合，必须给出高效 kernel 实现；二是注意力对精度异常敏感（FlashAttention3 的 FP8 版本误差大到不可用，字幕口径）。

SageAttention（ICLR 2025）的做法：Q、K 按 FlashAttention 分块粒度做 per-block INT8 量化后用 INT8 tensor core 算 $S$ ；P、V 不量化，但讲者发现 $P$ 行和为 1、与 $V$ 相乘结果不溢 FP16 表示范围，于是给 PV 乘法换 FP16 累加器（而非 FP32），在 4090/5090 上再快一倍。

真正的精度杀手是 K 的 channel-wise outlier：K 的每一列围绕某个固定大值分布，但量化必须沿 token 方向分组（矩阵乘的归约维度不可被量化分组切断，否则无法反量化），组内大小值悬殊导致量化不准。解法 Smooth-K 非常「优雅」：先把 K 每列的均值向量 ， $\bar{K}$ ，逐列减掉，K 立刻平滑到零附近，量化误差消失；而 $Q(K-\bar{K})^T$ 等价于给 $S$ 的每一行减去同一个常量，不改变 softmax 的输出分布——即数学上严格无损，只需在算子最前面加一行。结果是 kernel 约 2 倍于 FlashAttention（字幕口径），语言、视频、图像、分类模型均无精度损失（小数点后 2–3 位无差异）；短序列如 CogVideo（约 1.7 万）端到端也有约 35% 加速。

### SageAttention 2：INT4/FP8、Smooth-Q 与两个硬件级发现

第二代更激进：Q/K 用 INT4（或 INT8）、P/V 用 FP8。INT4 下仅 Smooth-K 不够，Q 的 channel outlier 同样要处理，但 Smooth-Q 没有免费午餐：把 $(Q-\bar{Q})(K-\bar{K})^T$ 展开后 ， $\bar{Q}K^T$ ，这一项是逐列不同的，必须作为补偿项预先算好（复杂度 $O(ND)$ ，相对 $O(N^2 D)$ 可忽略）、在 kernel 内加回。消融结论：Smooth-K/Smooth-Q 远比 Hadamard 旋转类技巧有效，两者正交可叠加，做到几乎无损。

量化粒度上提出 per-thread 量化：按 PTX MMA 指令分析每个 GPU 线程实际负责哪些 token，把同线程的 token 编为一个量化组——反量化仍可在一个时钟周期内 fuse 掉（per-block 的速度），粒度却比 per-block 细 32–64 倍（字幕口径），精度与 per-token 相当甚至更好（原因讲者称尚不明确）。

最硬核的一段是 FP8 累加器精度问题：讲者团队探测发现，NVIDIA FP8 矩阵乘的累加器文档标称 FP32，实际只有 FP22（8 位指数 + 13 位尾数），误差会随序列长度累积。他们在 SageAttention2 技术报告首版（2024-11-07）中首次报告并给出分析，解法是 two-level accumulation：外加 FP32 buffer，把每块 MMA 的低精度结果逐段累加到 FP32，把误差限制在单个分块内。讲者特别指出 DeepSeek V3 技术报告（2025-05）中的相关陈述与此几乎一致但晚于他们且未引用——此为讲者一方的说法，纪要按字幕口径记录，不作裁断。

结果：kernel 约 3 倍于 FlashAttention2，在 Hopper 上比 FlashAttention3（FP16）快约 2 倍，与 FlashAttention3 FP8 版速度相当但精度高很多（长序列大海捞针、混元视频等例子上对方生成崩坏而 SageAttention2 无损；字幕口径），且全显卡可用（FlashAttention3 仅限 Hopper）。CogVideoX-5B 端到端约 2 倍。后续 SageAttention2++ 把 PV 的 FP8 乘法改用 FP16 累加器，再提速 25%（需做缩放防溢出）。

### SageAttention 3：FP4 前向与低比特训练

第三代把 Q/K/P/V 全推到 FP4（NVFP4）：5090 上约 1000 TOPS（字幕口径）——讲者对比称 H100 上 FlashAttention3 也只有约 660 TOPS，而视频生成部署实际都跑在 4090/5090 这类卡上（这类任务是 compute bound，不需要高带宽 HBM），便宜卡反超有现实意义。推理加速上限约 5 倍。同时提出首个低比特注意力反向过程用于训练加速：从 base model 微调到 instruct model 可做到无损（字幕口径），预训练仍有一点差距、讲者称近期方法已基本解决。Q&A 补充：训练版用 INT8/FP8，MFU 约 50%–60%（未专门优化）；训练精度与推理精度互不绑定，训好的模型可直接换 FP4 算子推理，不存在「训推精度对齐」问题，因为 attention 本身无参数。

### 稀疏注意力之一：SpargeAttention

稀疏路线的两个难点：怎么定 router（哪些块可以跳）、以及注意力其实没那么稀疏。SpargeAttention 定位是首个算子级稀疏注意力：只吃 Q/K/V 三个 tensor 就能执行，不依赖具体模型。方法三件套：

1. **选择性 token 压缩预测**：对相邻 token 做 mean pooling 压缩后算一张粗 attention map，大值块正常算、小值块跳过；但只压缩相似度足够高的 token，相似度不达标的行/列强制必算——否则粗 map 的概率不可信。
2. **Sparse online softmax**：利用 FlashAttention 本就要维护的 local max 与 global max，只加一行判断——当某块 local max 比 global max 小很多（差值大于 10 时 softmax 后近似为 0），直接跳过该块的 P/V 运算。零额外开销，单这一招就能提速约 30%–40%。
3. **建在 SageAttention kernel 上**：即使稀疏度为零也白拿量化加速。

结果为 4–7 倍无损加速（字幕口径；与 SageAttention 叠加、50% 稀疏度时约 10 倍），预测开销可忽略。Q&A 中讲者坦承动态稀疏的短板：存在预测误差，某些模型上不如针对该模型调好的静态稀疏（如滑动窗口 pattern）；但泛用性是动态路线的优势，他判断后续主流是动态稀疏（DSA、MoBA 都是动态的）。

### 稀疏注意力之二：SLA（稀疏 + 线性混合）

讲者先给稀疏上限的实测：Wan2.1-1.3B（30K 序列）oracle 稀疏度只有约 45% 无损，强行拉到 92% 时 attention 输出误差超过 30%；语言模型 30K 约 70%–80%、视频 DiT 约 50%（字幕口径）。序列更长（如混元约 120K）可到 60%–70%，但 99% 仍有损。

突破口是一个分解观察：把 attention map 按值拆开——最大的约 8% 的值构成一个稀疏度 90% 以上、但高秩的矩阵（不可低秩近似，适合稀疏计算）；其余约 92% 的概率质量构成一个稠密、但秩只有个位数的矩阵（适合低秩近似，而线性注意力本质就是低秩近似）。于是 SLA 把 $P$ 拆两路：大值走 sparse attention，小值走 linear attention，加总即结果；前向、反向都实现了高效 kernel。代价是改变了 attention 的计算逻辑，需要微调适配：替换 SDPA 一行、少量数据微调即可（top-k 取 0.05–0.1 时约 1000–2000 步、一台机器一天内；top-k 0.2 时约 500 步；全量参数训练已在商业模型验证，只训 SLA 投影矩阵的初步实验效果略逊但可行）。

结果：30K 序列上 95% 稀疏度无损视频生成，attention 约 14 倍加速、端到端约 2.2 倍（字幕口径）；与 SageAttention2 叠加的商业版本达约 70 倍、已在上线视频模型中验证无质量损失；H100 上（top-k = 0.2 的保守设置）相对 FlashAttention2 约 15 倍、相对 FlashAttention3 约 10 倍（字幕口径）。

### 影响力与生态（讲授口径）

SageAttention GitHub 主仓约 2.8K star、系列合计约 3.4K；讲者称 100 多家知名企业在真实产品中使用，已集成进 ComfyUI、Diffusers、NVIDIA TensorRT、阿里 Tora（A100 上端到端加速 52%）、百度飞桨（37.4%）、昆仑万维（50% 以上）等，并拿到生数、智谱等的应用证明。用法就是 pip 安装后把模型里的 SDPA 调用换成 SageAttention 一行。讲者强调五篇工作都在商业上线模型里真实用过。

### Q&A 要点（讲者最坦白的边界）

- **只优化 attention 值不值**：收益 = 注意力时间占比，长 prefill 短 decode 的 LLM 与视频生成最划算；decode 阶段序列长度为 1、是 memory bound，低比特 tensor core 加速意义不大，目前 Sage 主要帮 LLM 的 prefill（讲者正在推进 vLLM 的 prefill 支持；MLA 版本有 release 计划）。
- **稀疏度由什么决定**：第一是序列长度，其次是具体哪个模型，与输入关系很小；稀疏模式主要由模型决定。静态稀疏对单模型准、动态稀疏通用，后续主流是动态。
- **与 MoBA/NSA/DSA 的关系**：量化与稀疏正交，SageAttention 的 kernel 可直接加速这些稀疏注意力的 prefill 计算；Sparge 的 router 思路也可迁移。
- **FP4 选型**：NVFP4 比 MXFP4 准得多（量化粒度 1/16 对 1/32、scale 用 E4M3 对 E8M0），速度相同；但 LLM 上 FP4 仍可能掉点，语言模型推荐 SageAttention2（++），P/V 在任何场景都优先 FP8 而非 INT8 或 FP4。
- **量化/反量化开销**：复杂度 $O(N)$ 相对注意力的 $O(N^2 D)$ ，1K 序列约占 5%、4K 约 1%、视频长序列 1% 以内；mean pooling 预测开销在 10K 以上不到 1%（只算 mean pooling 在 0.5% 以下）。
- **block size**：推荐 64 或 128×64，不要小于 64——再小 FlashAttention 本身的速度就垮了；block 越小稀疏越好做，这是固有 tradeoff。
- **评测可信度**：承认 VBench 等指标对折损不完全敏感；他们的做法是 VBench、VLM reward 等约 9 个指标全跑、掉点都极小才算无损，商业场景最终靠人眼；注意微调过的模型不能用像素相似度判损，training-free 场景才可以。
- **实验条件建议**：做这类 kernel 工作，显卡数量可以不多，但每种卡最好有一张（4090/5090 各两张也能起步）。

## 关键数字总表

| 指标 / 说法 | 数值 | 来源 |
|---|---|---|
| 视频生成中注意力时间占比 | 超过 80%，例子中 250 秒 vs 其余 50 秒 | 字幕 |
| 量化后注意力时间（同上例子） | 250 秒 → 5 秒，质量损失很小 | 字幕 |
| 5090 tensor core 吞吐阶梯 | FP16 约 209（讲者口径单位），FP8 约 2 倍、INT8 约 4 倍、FP4 约 8 倍 | 字幕 |
| SageAttention kernel 加速 | 约 2 倍 vs FlashAttention，无精度损失 | 字幕 |
| SageAttention 短序列端到端 | CogVideo（约 1.7 万序列）约 35% | 字幕 |
| SageAttention2 kernel 加速 | 约 3 倍 vs FlashAttention2；约 2 倍 vs FlashAttention3 FP16（Hopper） | 字幕 |
| SageAttention2 端到端 | CogVideoX-5B 约 2 倍、注意力约 3 倍 | 字幕 |
| SageAttention2++ 额外提速 | 约 25%（FP8 乘法 + FP16 累加器） | 字幕 |
| FP8 累加器实际精度 | 标称 FP32、实测 FP22（8 指数 + 13 尾数），SageAttention2 于 2024-11-07 首报 | 字幕 |
| per-thread 量化粒度 | 比 per-block 细 32–64 倍，反量化零额外延迟 | 字幕 |
| SageAttention3 FP4 吞吐 | 5090 约 1000 TOPS；H100 上 FlashAttention3 约 660 TOPS | 字幕 |
| SageAttention3 训练版 MFU | 约 50%–60%（未专门优化） | 字幕 |
| SpargeAttention 加速 | 4–7 倍无损；单 sparse online softmax 一招约 30%–40% | 字幕 |
| Sparge + Sage 叠加（50% 稀疏） | 约 10 倍 | 字幕 |
| oracle 无损稀疏度 | Wan2.1-1.3B（30K）约 45%；强拉 92% 输出误差超 30% | 字幕 |
| SLA 稀疏度与加速 | 95% 稀疏无损（30K），注意力约 14 倍、端到端约 2.2 倍 | 字幕 |
| SLA 商业版（叠 SageAttention2） | 注意力约 70 倍，已上线验证 | 字幕 |
| SLA 在 H100（top-k = 0.2） | 约 15 倍 vs FlashAttention2、约 10 倍 vs FlashAttention3 | 字幕 |
| SLA 微调成本 | top-k 0.05–0.1 约 1000–2000 步；top-k 0.2 约 500 步 | 字幕 |
| 生态采用 | 100+ 企业；阿里 Tora A100 端到端 52%、飞桨 37.4%、昆仑万维 50%+ | 字幕 |

## 可迁移

- 对 post-training / RL infra 的直接启发：rollout 与训练里真正吃时间的往往就是长序列 attention（长上下文 RL、视频/多模态生成尤甚）；评估任何「高效 attention」先问它在目标负载中的时间占比，再问它是量化路线（上限被 tensor core 位宽封顶）还是稀疏路线（上限被序列长度与模型的可稀疏性封顶）——这两条线正交，先量化保底、再稀疏冲量，是讲者用五篇工作验证过的组合顺序。
- Smooth-K 式的「数学等价变换」值得优先于「近似技巧」：在动任何近似（旋转、裁剪）之前，先看能不能用不改变函数值的变换（减均值、重参数化）把分布整形到量化友好的形状；这类变换无损、可叠加、且往往只需在算子入口加一行。
- kernel 工作的可信度评估清单（本期现成的）：是否给出真实 kernel 而非模拟、是否跨显卡验证、精度是否在长序列大海捞针类任务上验过、加速是否含预测/反量化 overhead、评测是否多指标 + 人眼双轨。讲者对 FlashAttention3 FP8 的批评（只在 Hopper、精度崩坏）就是反例。
- 训推精度解耦的观念可迁移到 RL 训练栈：attention 无参数，训练用什么精度与推理部署用什么精度互不绑定；同理，RL 里 rollout 引擎与训练引擎的数值精度可以分开选型，不必强行对齐。

## 疑问 / 下一步

- FP8 累加器 FP22 的探测细节与 DeepSeek V3 技术报告的先后关系，讲者给了一方说法，值得回 SageAttention2 技术报告（2024-11-07 版）与 V3 报告原文对读核实。
- SLA 中线性注意力那路的低秩近似误差如何随 head、层、任务变化，字幕只有结论没有分布图；且 SLA 需要微调、对 RL 训练中途切换算子的稳定性未提及，想深挖需读论文的微调协议与消融。
- SageAttention3 的反向训练目前 MFU 约 50%–60% 且预训练仍有差距，低比特 attention 能否进预训练主循环（而不仅是微调）仍是开放问题；讲者称近期已有方法基本解决，待其论文确认。
- 动态稀疏 router 的预测误差在什么分布下会系统性犯错（讲者只给了「有时不如静态」的定性说法），对把 Sparge/SLA 用到 RL rollout 的长尾输入时是个实际风险点。

## 原文金句

> 「AI Infra 的本质其实就是把所有的运算都变成 compute bound。」（约 21 分钟处，讲 FlashAttention 的意义时）

> 「我们只要在 attention 算子的最前面加一行 $K = K - \bar{K}$ ，就可以解决这个问题了。」（约 35 分钟处，讲 Smooth-K；字幕口径的公式化转述）
