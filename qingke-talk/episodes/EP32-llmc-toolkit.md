# EP32 — LLMC：大语言模型压缩工具的开发实践

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP32-llmc-toolkit.html

> 「终极问题：你在跑算法之前，先问问自己能不能加速——往往不能加速。」——第二位讲者在 Q&A 收尾时反复给出的判词，也是全场落地经验的总纲

## 元信息

- 期号：32（官网期号）
- 标题：LLMC：大语言模型压缩工具的开发实践
- BV：BV1BeNZeeEQy
- 时长：01:51:01（讲授约 89 分钟 + Q&A 约 22 分钟；两位讲者接力，前半讲背景与算法、后半讲使用与扩展）
- 提炼日期：2026-10-02
- 分享嘉宾：两位 LLMC 开发者（字幕中互称「石桥」与「雍阳」，另一处作「云阳」；两人未在字幕中自述全名与单位。LLMC 论文作者页含 Shiqiao Gu、Yang Yong，与字幕读音相近，全名以论文作者页为准）
- 相关论文：Ruihao Gong, Yang Yong, Shiqiao Gu, et al., *LLMC: Benchmarking Large Language Model Quantization with a Versatile Compression Toolkit*，EMNLP 2024 Industry Track，https://aclanthology.org/2024.emnlp-industry.12/
- 相关代码：https://github.com/ModelTC/LightCompress （讲授时仓库名为 LLMC，现已更名 LightCompress；讲授中给出的找法是「Google 或 Bing 搜 LLMC，第一条就是」）
- B站链接：https://www.bilibili.com/video/BV1BeNZeeEQy/
- 字幕原文存档：本地 `transcripts/EP32.txt`（2635 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP32，已存档）清洗提炼，信息来源为 AI 字幕原文（经人工清洗；个别专名可能有识别误差，如 QuaRot 在字幕中作「Corrot/Court/QUERT」、lm-eval-harness 作「lm evil」、TensorRT-LLM 作「特斯拉 TM」、Pile 类校准集作「拍 EVO」，均以论文与官方文档为准）。凡讲授未给出精确数字处，以下只作定性转述并标注（字幕口径），不补数字。

> 关联：EP25《LLMC：大语言模型的量化基准》讲的是 LLMC 的 benchmark 论文面（算法横评结论），本期是同一工具的开发实践面：架构设计、算法实现里做的三处改进、现场演示与两位开发者的一线落地经验。两期合起来是「结论 + 怎么用」的一对。

## 一句话总结

LLMC 把 LLM 量化做成了一个不绑定单一算法、单一后端的通用压缩工具包：用「subset 抽象 + 逐 block 上卡」的流水线让 405B 级模型也能在单张 80GB 卡上量化，用与官方实现对齐到小数点后两三位的复现质量托底，再在 AWQ、OmniQuant、QuaRot 三个算法上各做一处针对性改进（clip 对称性对齐、AWQ 初始化可学习参数、QuaRot+GPTQ 组合）；但全场真正的信息量在后半段两位开发者的落地经验：量化方案的选择顺序是「后端 → 方案细节 → 硬件 → 业务形态 → 精度容忍度」，算法只是最后一步，且任何量化动手之前先问「究竟快不快」。

## 核心

### 引入：为什么要压缩，以及 LLM 量化难在哪

模型参数量随年份近线性增长（讲者举 Llama 405B 为当时最大），部署代价直接劝退普通用户：175B 的 GPT-3 约需 5 张 80GB 或 8 张 40GB GPU（字幕口径）。压缩的收益是推理加速、降功耗、降显存与存储/传输成本，但代价是精度风险——LLMC 的立项动机就是「为了保持压缩模型的精度而提出的工具」。

LLM 量化区别于小模型量化的核心挑战是**激活离群值（outlier）**：某些 token 的个别 channel 绝对值远大于其他 channel，per-token 量化时这些 channel 主导量化区间，其余 channel 反量化后恢复不出来。讲者用 LLM.int8() 一系的经典曲线说明：模型规模跨过约 2.7B 后，8 比特量化的精度出现大幅下跌，正是 outlier 涌现的位置；混合精度（outlier channel 保持 FP16/BF16、其余 8 比特）能把曲线拉回，这也反证了 outlier channel 是精度损失的主因。后续所有算法（AWQ 的激活感知缩放、QuaRot 的旋转、GPTQ 的误差回注）本质上都是在处理这同一件事。

### LLMC 的定位与总体设计

讲者给出的规模口径：支持 **18 种压缩算法**（另一处作「近 20 种」），覆盖权重量化、激活量化、KV cache 量化、混合精度，以及结构化/非结构化稀疏（本次主要讲量化）；模型侧覆盖基座/对话/代码模型（Llama、Qwen、书生等）、MoE（DeepSeek、Mixtral）、多模态（LLaVA、Qwen2-VL、Llama 3.2）；导出 **5 个主流推理后端**（字幕列举 LightLLM、TensorRT-LLM、SGLang、vLLM、MLC-LLM）；评测对接 PPL、OpenCompass、lm-eval-harness。与业界同类框架的对比结论是：LLMC 在算法数、模型广度、评测方式、后端数四维都最全，设计理念是「更通用」，不聚焦某一个后端或某一个算法。

两个关键设计决定了它的工程形态：

1. **subset 抽象**。量化实际作用于每个 block 内部的线性层；LLMC 把同一 block 内「共享同一输入」的线性层划成一个 subset（以 Llama 为例共 7 个线性层、4 个 subset：QKV / O / gate+up / down）。新模型接入只需按其拓扑定义 subset，不必像很多框架那样重写一遍 HuggingFace 的 forward 并在其中穿插量化逻辑；与 Llama 同构的模型直接复用定义。这一抽象后面还被复用为「只对 down 层单独做 AWQ」这类精细操作的抓手。
2. **逐 block 流水线**。校准数据先过 embedding 得到第一个 block 的输入，然后逐 block 循环：block 从 CPU 搬上 GPU → 执行算法 → 搬回 CPU。任意时刻 GPU 上只有一个 block，因此显存占用极低，405B 级模型单张 80GB 卡即可完成量化（字幕一处作「4.5B」，结合后文「400 多 B 单卡量化完」的表述，应为 405B 级的识别误差）。例外是 GPTQ：其 Hessian 计算与分解显存需求大，单卡会 OOM，LLMC 为其做了多卡实现。

实现质量准则是**与官方实现/论文结果做精度对齐**：讲者展示的对比表中，LLMC 各算法实现与官方实现的 PPL 基本对齐到小数点后两位至三位。

### 三个算法的实现与改进（全场技术密度最高的部分）

**AWQ：clip 的对称性必须与量化方式对齐。** AWQ 的思想是激活感知：激活值大的 channel 对应权重更重要，用缩放系数体现重要性，再对权重做 clip 并搜索 clip 因子。讲者指出其缺陷：原始 AWQ 搜索的是对称 clip 因子，但在非对称量化 setting 下用对称截断，在低比特（尤其 2 比特）会带来严重精度下降。LLMC 的改进是让 clip 与量化「对齐」：对称量化用对称 clip，非对称量化就搜索非对称的 clip 因子。实验（字幕口径，未给精确值）：W3 下两者结果差不多，W2 下原始 AWQ 的 PPL 显著变大、精度基本崩掉，改进后恢复正常。

**OmniQuant：用 AWQ 的搜索结果初始化可学习参数。** OmniQuant 学习两组参数：weight clipping 因子与等价变换的 scale/shift。其缺陷是训练非常不稳定，超参与初始化选不好就学不好、学得久（其原始初始化分别借用 SmoothQuant 与 OS+ 的方法，clip 从 1 开始学）。讲者展示：2 比特下原始设置需约 40 个 epoch 才学到 PPL 9.62；LLMC 先用 AWQ 的搜索过程给可学习参数一个更好的初始化，只用 5 个 epoch 即达到更低的 PPL（字幕给出 8.66 与 12.30 两个数，未逐一说明对应设置）。

**QuaRot + GPTQ：一个「峰度更低反而更差」的悖论，以及天然互补的组合。** QuaRot 用随机 Hadamard 旋转矩阵作用于激活与权重以削平 outlier，缺陷有二：矩阵是随机的，结果有波动；旋转时不考虑量化输出误差。讲者用两个指标把这件事讲透了：旋转后张量的峰度（字幕口径：元素减均值、除以标准差后取四次方的统计量）确实比 AWQ 更低（outlier 更少），但各线性层输出的余弦相似度明显低于 AWQ、结果反而更差——因为旋转把数值推向 0.5 附近（字幕 Q&A 口径），舍入不确定性增大。结论：**缓解 outlier 不保证效果**。GPTQ 则是逐列量化、把已量化列的误差回注到未量化列，缺陷是误差会一路累积到最后的列上。两者组合恰好互补：QuaRot 先削平 outlier 使每块量化误差变小，GPTQ 再微调权重、避免误差累积；可视化上 AWQ+GPTQ 的后续 block 误差明显加深（误差累积），QuaRot+GPTQ 则没有这一现象，PPL 与下游准确率都远好于单独使用或 AWQ+GPTQ（字幕口径）。这一组合成为讲者后文推荐的标准流水线。

后端导出环节只举了一个例子但很说明问题：vLLM 加载 BF16 原始模型权重占约 29G，LLMC 量化成 FP8 后加载约 15G，几乎减半，同题生成的文本肉眼几乎一致。

### 使用方式：config 解剖与三段评测

第二位讲者接棒做现场演示，要点如下：

- **环境**：LLMC 免安装（指定路径即可跑），依赖 `pip install -r requirements`，或直接用官方 Docker（CUDA 12.1 / 12.4 两个 tag，另有阿里云镜像）；镜像里预置了校准/评测数据集，省去从 HuggingFace 下载的麻烦。
- **config 结构**：随机种子；model（类型、路径、tokenizer 快/慢、torch_dtype）；calibration（校准集配置）；eval（评测配置）；quantization（method、权重/激活各自的比特数、对称与否、per-channel/per-group 与 group size、per-token 动态等，外加每个算法的 special 专属设置）；save。
- **save 的两种浮点格式最易误解**：`save_trans` 存的是经变换调整后的权重（仍是 BF16），`save_fake` 是再经量化-反量化（QDQ）后的浮点模型——存下来的都不是量化模型、体积不会变小，但已经是「更适合量化的浮点模型」，直接拿它做朴素量化也比原始模型精度高。
- **三段评测**：config 里可同时评 pretrain（原始浮点）、transform（变换后）、fake-quant（QDQ 后）三段的 PPL，演示中 Qwen2.5-0.5B 做 AWQ W4 非对称、group 128：浮点 PPL 14.25 → transform 后仍是 14 点几 → fake-quant 后变 16（字幕口径）。
- **四种评测方式**：custom generation（给定自定义问答，直接看量化前后输出，抓 bad case）、custom PPL（在业务数据集上算 PPL）、token accuracy（量化后下一 token 与浮点模型的一致率）、WikiText PPL（学术口径）。校准集与评测集都可以换成业务数据，且要拼好 system prompt 格式；多卡时用 torchrun 按 DP 把数据分到多卡。

### 落地经验（上）：先选方案，再谈算法

讲者明确把顺序拆成两段：先定**量化方案**（几比特、对称与否、粒度），再选**算法**。方案选择的六步：

1. **先选推理后端**，确保模型能跑起来；
2. **只能在后端支持的方案里选**——后端不支持 W4A8，就做不了 W4A8；
3. **摸清方案细节**：权重/激活各几比特、对称或非对称、均匀或非均匀、per-channel/per-group、per-tensor/per-token、动态或静态；
4. **摸清硬件细节**：FP8 需要 H100、L40S、4090 等新一代卡，A100 上只能 INT8/INT4（FP8 的量化过程可在 A100 上模拟跑，但推理跑不了）；还要确认 KV cache 能否量化、group size 多少；
5. **按业务形态选**：长输入（prefill 计算密集）优先 W8A8（加速计算），因为 W4A16 在长 prefill 上可能比 FP16 还慢；长输出（decode 访存密集）优先 W4A16（省搬运）；短输出则 decode 总时间本来就少，优先 W8A8；
6. **按精度容忍度选**：容忍度排序是 KV cache 最大（4 比特通常能稳住，且能把显存省给 KV cache 本身），其次激活（8 比特），权重最敏感（4 比特是常见下限）；动态量化的 W8A8 精度通常高于 W4A16。

另有五条细粒度经验（讲者称 “more things”）：其一，输入里有关键长设定（如长 system prompt）时，激活量化容易把关键信息量化丢，倾向只做 weight-only；其二，部署有 prefix cache 时长 system prompt 已被缓存，实际计算量没想象中大，未必需要 W8A8；其三，**prefix cache 预存**：把固定 system prompt 的浮点 KV 预先存好，量化时这一段永远走浮点 cache，可做到这一段无损（讲者推荐 IntactKV 一类 attention sink 论文，具体见 Q&A）；其四，**先问究竟快不快**：速度取决于算子在具体 shape 上的实现，FP6 等方案在很多 shape 上是减速的，非 NVIDIA 硬件（华为等）特性又不同，先测速再决定做不做；其五，**「FP8 is all you need」**（限有 FP8 卡且实测能加速时）：FP8 本质是非均匀量化，与大模型激活的不均匀分布天然契合，精度高又快，静态 FP8 精度也高，复杂算法未必需要——但到了 Blackwell 的 FP4，精度就不会这么乐观，算法仍有用武之地。

### 落地经验（下）：算法流水线与两个判断

方案固定后，讲者给出的算法流水线：

1. **用好校准数据**：工业落地用业务场景数据，拼好 prompt/system prompt 格式；多模态要对齐图片/音频预处理（尤其图片预处理链）；
2. **推荐组合：先 QuaRot 旋转、再 GPTQ**（config 里有现成的 combination 两段配置）；
3. **关掉在线旋转**：QuaRot 在 down/O 层引入的在线旋转精度收益高，但多数推理引擎未适配、不可部署，关掉换可部署性；
4. **AWQ 补 down 层**：down 层 outlier 最难处理，关掉在线旋转后 down 没被旋转，就用 subset 抽象把其他 subset 的 `do_trans` 关掉、单独对 down 做一遍 AWQ，最后可再叠一轮 GPTQ；
5. **PPL 的价值是折中判断**：WikiText PPL 掉 1–2 点（乃至 10 点），业务大概率也掉；只差 0.1–0.3 则不能断言谁更好。工业落地应更看重自定义数据集上的评测（custom PPL + custom generation 人工看 bad case），而不是学术 PPL 的小数点之争；
6. **training-based 方法（OmniQuant、QAT 类）要打问号**：实际用下来容易过拟合校准集——校准集覆盖不了所有场景；这类论文常见在 Wiki 上训、Wiki 上测，换成 C4 训、Wiki 测，效果往往没有论文里那么好。当然不绝对，需要自己试。

### 扩展性：加模型、加算法、多模态

- **加新模型**：找到模型推理逻辑（transformers 库或与权重同目录的代码），按拓扑定义 subset（每个 subset 及其前驱 op），再重写获取 blocks/embedding 等成员函数；Llama 同构模型直接复用。多模态更复杂：不只要定义语言部分，还可能要量化视觉 tower（如 InternVL 的 vision subset），且 QuaRot 的旋转等价 merge 还要考虑 vision projector 这条并联分支（语音同理）。
- **加新算法**：多数算法是 blockwise 优化，重写 LLMC 的 `subset_transform` 一个函数即可（AWQ 的核心逻辑就在此类里）。
- **多模态对齐的判断法**：LLMC 里直接打印喂给 `model.generate` 的 input，与推理代码里同一张图片的 input 逐项比对，一致才算预处理对齐。
- **定位声明**：LLMC 是纯算法调精度工具，不含推理算子，内部推理用原生 transformers；团队新开了一个量化算子测速 benchmark QuantHorizon（Q&A 中字幕作「量化地平线 / QUANTORION」，取竞速游戏《地平线》之意），支持 INT6/FP8、W3A6、W4A6 及稀疏+量化算子（字幕口径），用来回答「究竟快不快」。
- **剪枝与蒸馏现状**：非结构化稀疏不能加速；结构化剪枝与 NVIDIA 2:4 稀疏精度很糟、难部署，整 block 删除也试过、精度糟糕。蒸馏路线（举 NVIDIA 的 pipeline 为例）用 32 个节点、每节点 8 张 H100，资源消耗远大于量化——讲者的结论是量化更经济实用。

### Q&A 要点

- **AWQ 的缩放系数 s 越大精度越好吗？** 不是。s 是 per-channel 的，目的是让 channel 间分布均衡，不是统一放大；应看量化前后的分布，而不是 s 的绝对大小。
- **AWQ 推理为何比 GPTQ 快？** 与算法无关，取决于推理算子的写法；速度问题用 QuantHorizon 测，别把算法和算子混为一谈。
- **SmoothQuant 与 AWQ 的区别**：SmoothQuant 没有搜索，直接按公式确定迁移参数（Q&A 口径）。
- **W4A16 为什么能加速？** 它并不减少计算量（还多一步反量化），快在权重体积小、搬运少，decode 是访存受限阶段所以净收益为正。
- **QuaRot 缓解了 outlier 但 PPL 更高，是否说明 outlier 不重要？** 讲者认可这个问题问得好：QuaRot 官方仓库 issues 里也有同样的质疑；他们的解释是旋转把数值推到 0.5 附近、舍入不确定性增大，所以缓解 outlier 不等于效果好，建议 QuaRot 必须配 GPTQ 一起用。
- **量化能否精度无损？** W8A16 几乎无损；W8A8、W4A16 一般都会掉一些，FP8 好一些。
- **同等算力下小模型 vs 大模型量化？** 讲者倾向大模型量化更好，但不确定，更取决于基座模型本身训练得鲁不鲁棒。
- **激活为什么比 KV cache 更敏感？** 激活不止 K/V，还有 Q、O、up、down，其中 down 层最敏感（其前有乘法操作会放大激活）；KV cache 只有 K、V 两类张量。这一条与「AWQ 补 down 层」的流水线互为印证。
- **混合精度会不会负担很高？** 可以克制地混：只混 down 层、W4A16 与 W8A16 混、只混部分层；层内混合也行，但老规矩——先问快不快，往往不快。
- **W4A4 为什么不推荐？** H 卡上没有对应算子，往往不如 W8A8 快、精度还更低；学术可以，部署不值。W4 以下同理，先看硬件与算子（如 llama.cpp 在 CPU 上支持 W2–W6 才另当别论）。
- **torchrun 多卡在 30/60 层后 NCCL timeout**：团队没遇到过，建议用官方 Docker、调大超时阈值，并可把 config 发给团队复现。
- **Qwen 模型跑 QuaRot 报错**：某些 size 的 Hadamard 矩阵不支持时，可退而用 LLMC 内置的随机矩阵替代。
- **Trans V1/V2 之别**：区别不大，源于 AWQ 论文改版后 trans 公式随之变版的一个小 trick。
- **CV 模型为何常用 training-based PTQ**：CV 模型小、任务专注，过拟合的影响小；LLM 通用性强，校准集过拟合的代价更明显。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| GPT-3 175B 部署需求 | FP16 原始模型 | 约 5 张 80GB 或 8 张 40GB GPU | 字幕（引入） |
| outlier 涌现的规模阈值 | 8 比特量化精度曲线 | 约 2.7B 参数后精度大跌 | 字幕（量化挑战） |
| 工具规模 | — | 18 种压缩算法（另一处作近 20 种）、5 个推理后端 | 字幕（工具介绍） |
| 单卡量化上限 | 逐 block 流水线 | 405B 级模型单张 80GB 卡 | 字幕（一处作「4.5B」，按上下文取 405B 级） |
| AWQ 对称 clip 缺陷 | 非对称量化、W2 | 原始 AWQ 的 PPL 显著变大（近崩），clip 对齐改进后恢复正常；W3 下差异不大 | 字幕（定性，未给精确值） |
| OmniQuant 收敛 | 2 比特，原始初始化 | 约 40 epoch → PPL 9.62；AWQ 初始化 → 5 epoch 达到 8.66 / 12.30（两数对应设置字幕未逐一说明） | 字幕（算法部分） |
| QuaRot 的悖论 | 与 AWQ 对比 | 峰度更低（outlier 更少），但层输出余弦相似度更低、效果更差 | 字幕（算法部分） |
| 后端导出体积 | vLLM 加载 BF16 权重约 29G | FP8 量化后约 15G | 字幕（后端部分） |
| 现场演示 | Qwen2.5-0.5B，AWQ W4 非对称 group 128 | PPL：浮点 14.25 → transform 后 14 点几 → fake-quant 后 16 | 字幕（演示） |
| 精度容忍度下限（经验） | 极端可接受比特 | KV cache 4 比特、激活 8 比特、权重 4 比特 | 字幕（落地经验） |
| 蒸馏路线资源对照 | NVIDIA 蒸馏 pipeline | 32 节点 × 每节点 8 张 H100 | 字幕（扩展部分） |

## 可迁移

- **「subset 抽象」是一切精细量化操作的前提**：先把 block 内的线性层按共享输入划成子集，后面「关在线旋转、单独给 down 层补 AWQ」这类外科手术式操作才有抓手。做自己的压缩/评测工具时，先投资这层结构定义，比先堆算法更划算。
- **量化选型顺序可直接抄**：后端能跑 → 后端支持的方案集合 → 方案细节（对称/粒度/动态静态） → 硬件代际（FP8 卡有无） → 业务形态（长输入选 W8A8、长输出选 W4A16） → 精度容忍度（KV 先行）。算法（AWQ/GPTQ/QuaRot）是最后一步，且有现成推荐流水线：QuaRot → GPTQ，关在线旋转，down 层单独 AWQ。
- **评测纪律**：WikiText PPL 只用来抓大掉点（掉 1 点以上要警惕），小数点级差异（0.1–0.3）不作数；上线决策看业务数据上的 custom PPL + custom generation 的 bad case。这条同样适用于 RL 模型的压缩评测。
- **「先测速再量化」**：任何量化方案落地前先用算子级 benchmark 确认目标 shape 上真的加速；W4A4、FP6、层内混合精度在讲者经验里「往往不快」。

## 疑问 / 下一步

- OmniQuant 的「5 epoch 达到 8.66 与 12.30」两个 PPL 分别对应什么模型/设置，字幕未逐一说明；若要把这条写进自己的实验记录，需回 LLMC 仓库的 OmniQuant 配置与文档核对。
- 讲者推荐的 prefix cache 预存与 IntactKV 的具体做法只给了思路（固定 system prompt 的浮点 KV 预存、量化时强制命中），与主流推理引擎（vLLM/SGLang 的 prefix cache 实现）如何拼接，未展开。
- QuantHorizon 在讲授时刚起步（「一两周」），其算子覆盖与测速口径值得后续跟踪，看它是否成为量化选型的标准前置步骤。
- 两位讲者的全名与单位字幕未自述；若需正式引用，建议以 LLMC 论文（EMNLP 2024 Industry Track）作者页为准。

## 原文金句（1-2句）

> 「终极问题：你在跑算法之前，先问问自己能不能加速——往往不能加速。」——第二位讲者 Q&A（字幕 104:36）

> 「FP8 is all you need……但是到了（Blackwell 的）FP4，那个时候精度就不会那么乐观了，你可能还是需要上一些算法。」——第二位讲者谈 FP8 的边界（字幕 66:24–67:40）

> 「在你做量化算法之前，先问问它能不能跑、能不能快、能不能支持。」——第二位讲者谈新后端/新模型的接入顺序（字幕 100:19）
