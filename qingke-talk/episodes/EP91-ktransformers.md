# EP91 — KTransformers，在大模型微调与推理中的系统化实践

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP91-ktransformers.html

> 「（MoE）总参数量很大、每次计算只激活一小部分，但用 GPU 跑时全部参数仍要加载到显存里——实际上它对显卡的要求还是很高的。」——讲者对项目出发点的概括（字幕 03:29–03:46，清洗口径）

## 元信息

- 期号：91
- 标题：KTransformers，在大模型微调与推理中的系统化实践
- BV：BV18wUPBEE4r
- 时长：00:55:02（演示与讲授约 36 分钟 + 技术原理与问答约 19 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：李佩林（KTransformers 项目核心参与者，主讲）、张明星（清华大学计算机系，技术原理与问答环节补充；两人姓名字幕中亦作「李培林」，头衔与分工以官网介绍为准）
- 相关论文：KTransformers 项目论文（讲者提及推理侧的详细测试可参考项目此前论文，字幕中未给出标题与链接）
- 相关代码：KTransformers（GitHub 开源项目；含 KT-Kernel 推理内核与 KT-SFT 微调部分）；前端生态为 LLaMA-Factory（微调）与 SGLang（推理）
- B站链接：https://www.bilibili.com/video/BV18wUPBEE4r/
- 字幕原文存档：本地 `transcripts/EP91.txt`（1305 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP91，已存档）清洗提炼，信息来源为 AI 字幕原文（经人工清洗；个别专名有识别误差，如 KTransformers 在字幕中作「k transformers/kitchen former」、LLaMA-Factory 作「lama factory」、SGLang 作「SG浪/SJL」、LoRA 作「LAURA/LARA」、AMX 作「MX」、llama.cpp 作「拉玛点CPP」、DeepSeek 作「deep sick」，均已按项目与公开资料校正）。凡讲授口径与项目文档可能不同，以下均按字幕口径记录。

## 一句话总结

KTransformers 的核心判断是：MoE 模型总参数大、每次只激活一小部分，把全部参数塞进显存是对昂贵 GPU 显存的浪费；而 CPU 内存便宜得多、只是带宽不足，恰好适合承接低激活的 routed experts。于是它的方案是 GPU+CPU 异构协同——attention 与 shared experts 放 GPU、routed experts 卸载到 CPU，配上 AMX 高性能内核与 Expert Defer 等调度优化，并以「加速组件 + 成熟前端」的定位接进 LLaMA-Factory（微调）与 SGLang（推理）生态，让 4090 级别的机器也能微调和推理 671B 甚至上千 B 的模型。

## 核心

### 立论：MoE 的参数分布决定了异构放置

讲者用两个例子锚定问题：Qwen 的 235B-A22B 意味着总参 235B、每次只激活 22B；DeepSeek 671B 里 routed experts 占绝大多数参数，但每步只激活其中 6–8 个专家（视模型配置）。纯 GPU 方案必须把全部参数装进显存，显存需求与实际计算量严重脱钩。CPU 的内存价格比显存低得多，短板只是带宽——而低激活专家恰好是「参数多、计算少」的工作负载，正适合放 CPU。这是整个项目的出发点：把参数量占比最大的 routed experts 卸载到 CPU，attention 与 shared experts 留在 GPU。

### 产品形态：只做加速组件，借前端生态落地

KTransformers 的定位不是全栈框架，而是后端加速组件，需要配前端使用：

- **微调**：以 LLaMA-Factory 为前端，KT 作为后端替换原生 HuggingFace 路径，`USE_KT=1` 一行接入；微调以 LoRA 为主（全参对超大模型显存不现实）。数据预处理、checkpoint 保存等都沿用 LLaMA-Factory 的既有能力。
- **推理**：以 SGLang 为前端，KT-Kernel 作为 MoE 高性能后端合入；GPU 侧的并行等能力主要由 SGLang 承担。也有 LLaMA-Factory 的 chat 接口可用，但讲者推荐推理统一走 SGLang 路线（版本更新更快）。

实测口径：4090 上可跑 671B 甚至 1000B 量级模型；14B 模型用 KT 后显存从全 GPU 的约 40G 降到约 5.x G，代价是需要 50 多 G 内存；DeepSeek V3 量级的 LoRA 微调约需 20–40 小时（约 1 万条、512 token 量级的估算）；Kimi K2（约 1000B）微调约需 2.1 TB 内存，团队测试时加了 200G swap 完成。

### 微调实操要点（含几个容易踩的坑）

- **必须传 bf16 模型**：DeepSeek V3、Kimi K2 发布时是 FP8/int4 格式，需先解量化成 bf16；仓库里提供了在 CPU 上做解量化的脚本，照顾 GPU 不强的用户。粗略判断方法：参数量与文件体积对不上 2 倍关系就是低精度格式。
- **安装补丁**：CUDA runtime 补丁不只针对 CUDA 11.8，12.4、12.8 同样要打；flash-attention 要按 ABI true/false 选对安装命令，否则会报 undefined symbol。
- **YAML 配置**：LoRA target 设为 all 时 KT 默认覆盖 attention 与 shared expert 两部分；`cutoff_len` 与 chunk size 有联动坑——cutoff 大于 chunk size 且数据长度超过 chunk 时会出问题，chunk 不要设得太小；KT-Optimizer 规则负责把专家层替换成 KT 自己的专家并行算子，backend 目前支持 AMX INT8 与 BF16。
- **多卡策略**：用 model parallel 而非 HuggingFace Trainer/Accelerate 默认的 data parallel，因为后者会打破 KT 的算子放置策略；中间结果留在各设备上不回传，只有 loss 回到 GPU 0 计算，显存也能在多卡间摊匀。

### 推理路径：KT-Kernel + SGLang

使用流程是先把 MoE 权重用项目脚本量化成 KT 需要的格式（INT8 或 INT4），再启动 SGLang server；常用模型的预量化权重可以直接从 ModelScope 下载。讲者现场演示了从 README 安装（install.sh 会自动检测本地是否有 AMX 等指令集再编译）、准备权重、起 server 到 HTTP 请求调用的完整链路，并给了 CUDA OOM 时的调参指引（调 SGLang 的放置与 chunk 相关参数，速度略降但能跑起来）。微调一侧的 whl 包已发布，推理一侧有 Docker；后续计划把微调与推理环境封装进同一个 Docker。

### 内核与调度：prefill 拼算力，decode 拼重叠

技术部分（讲者整理自团队此前分享）按阶段拆解瓶颈：

- **Prefill 是计算密集型**，CPU 的短板在算力：用 AMX 指令集替代 AVX512，并针对 AMX 的特殊内存布局做 tiling 布局的 GEMM 内核，配合 kernel 融合与 work-stealing。微调基准上，单个 CPU 优化后的 BF16 GEMM 可超过 20 TFLOPS，INT8/INT4 更高。
- **Decode 的瓶颈在协调**：一次 forward 有成千上万个极短的 kernel launch，且 CPU 专家计算与 GPU attention 交替执行、互相等待。解法有三件——CUDA Graph 把 kernel 序列录制成图一次性重放；NUMA TP 等并行优化（相关特性在 0.4.1、0.4.2 版本中陆续合入）；**Expert Defer**：一层内按路由得分从高到低排序专家，CPU 先算高分专家、模型立即进入下一层，GPU 算 attention 时 CPU 再补算被延迟的低分专家，靠残差连接保证被延迟的专家仍能有效贡献后续层，机制类似流水线并行。讲者给的口径是 decode 端到端比 llama.cpp 快约 2 倍，Expert Defer 在几乎不影响模型行为的前提下再额外提供约 1.45 倍提升；延迟多少专家合适是速度与精度的权衡，量化曲线在项目论文里。

### Q&A 要点（边界与路线图）

- **平台支持**：华为 910B 等国产卡推理可做、微调暂未支持；ARM 当前一代缺矩阵计算指令集，要等下一代；NVIDIA 卡目前都支持，消费级 4090 甚至 4080 多卡是微调主场景。
- **单卡 4090 能跑什么**：内存足够大时推理可跑 Kimi K2（开源里更大的也少见）；微调因 backward 显存更高，主要适配较好的是 Qwen 系，24G 显存下 DeepSeek 的 KV cache 约可支持 120K 以上序列（讲者印象口径）。
- **RL 方向**：社区同学已实现 DPO 并提了 PR，尚未 review 合入；讲者判断单机跑 RL 不现实，多机互联或许能勉强跑；预训练则明确说这点算力可能性不大。
- **场景定位**：边缘场景（没有大 GPU）与高校实验室（CPU 多、GPU 少）是主要用户。
- **Prefill 走 PCIe 搬数据的备选方案**：在 roadmap 上，但讲者算过账——会卡在 PCIe 带宽上，是否最优取决于模型大小、带宽与算子计算强度，两种做法都会支持。
- **离理论最优还有多远**：张明星的答复是，模型结构不变时只剩算子与数据搬运的优化、不会有质变；几倍到十倍级的提升需要模型架构与底层系统共同设计（Expert Defer 就是为推理改了一点模型结构的例子）。团队 Q4 roadmap 的重点是易用性、兼容性（AMD、老 CPU）与性能。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| MoE 激活比例示例 | Qwen 235B 总参 | 每次激活 22B；每步激活 6–8 个 routed experts | 字幕（背景部分） |
| 显存占用（14B 模型微调） | 全 GPU 加载约 40G | KT 约 5.x G 显存 + 50 多 G 内存 | 字幕（演示部分） |
| 可运行模型量级 | 4090 级别机器 | 671B 甚至约 1000B 模型可微调/推理 | 字幕（演示部分） |
| DeepSeek V3 量级微调时长 | 约 1 万条、512 token 量级估算 | 约 20–40 小时 | 字幕（演示部分） |
| Kimi K2（约 1000B）微调内存 | — | 约 2.1 TB 内存，测试加 200G swap | 字幕（技术部分） |
| CPU GEMM 性能 | 单个 CPU，BF16 | AMX 优化后超过 20 TFLOPS，INT8/INT4 更高 | 字幕（技术部分） |
| Decode 端到端速度 | llama.cpp 对比 | 约 2 倍 | 字幕（技术部分） |
| Expert Defer 额外加速 | 几乎不影响模型行为 | 约 1.45 倍 | 字幕（技术部分） |
| 单卡长序列（印象口径） | DeepSeek，24G 显存 | KV cache 约可支持 120K 以上序列 | 字幕（Q&A） |
| 版本节奏 | 0.4.1 覆盖前两个 decode 优化特性 | 0.4.2 补上第三个（字幕口径） | 字幕（技术部分） |

## 可迁移

- **按「参数量/激活量」比值分层放置**：高参数、低激活的组件放便宜的大内存，低延迟敏感的组件放 GPU——这条原则可直接类比 RL infra 里 rollout 权重、参考模型、KV cache 的分级放置，先算每类张量的参数量与每步实际访问量再谈放置。
- **Expert Defer 的调度思想**：按路由得分给计算排优先级、低优先级活延迟到其他硬件的忙碌间隙里补算，本质是优先级 + 填充调度；与 agentic RL 中长尾 rollout 的 defer/填充、训推重叠是同一类手法。借残差连接兜底「延迟计算仍有效」这一点，则提示延迟/近似计算要预先留好数学上的容错结构。
- **组件化采用路径**：KT 只做加速后端、借 LLaMA-Factory 与 SGLang 的前端与生态入口而非自建全栈，是 infra 组件冷启动的现实打法——先解决「装得上、一行接入」，再谈功能完整。

## 疑问 / 下一步

- 微调吞吐的系统化对比（不同模型、不同硬件）字幕只给了量级估算，完整数字要查项目文档与仓库 benchmark。
- Expert Defer 延迟专家数与精度损失的量化权衡曲线在项目论文中，talk 未展开；这是把该机制迁移到其他异构调度前最需要先看的一组数。
- DPO 的社区 PR 合入时间、多机 RL 的可行性都还是开放项，讲者未给承诺。

## 原文金句（1-2句）

> 「（MoE）总参数量很大、每次计算只激活一小部分，但用 GPU 跑时全部参数仍要加载到显存里——实际上它对显卡的要求还是很高的。」——讲者立论（字幕 03:29–03:46，清洗口径）

> 「真的能有几倍甚至十倍以上的性能提升，就需要模型的架构和底层的系统共同去配合。」——张明星答「距理论最佳值还有多大提升空间」（字幕 51:40–51:47，清洗口径）

> 「你的局部资源不够，那就尽可能把各种各样的异构资源都协同起来使用。」——答「这个架构可以做什么」（字幕 45:30–45:37，清洗口径）
