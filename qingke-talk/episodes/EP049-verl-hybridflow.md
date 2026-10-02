# EP049 — verl 源码解读与 HybridFlow 编程范式讲解

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP049-verl-hybridflow.html

> 「我们最终的一个愿景，就是说我们能够让用户只用关心如何去指定这样一个 data flow graph，然后在数学上定义他的行为，然后把所有的，包括更底层的 kernel 的一些优化，全都交给我们这个框架底层来实现。」——讲者在背景部分给出的 verl 愿景（字幕约 05:28–05:48）

## 元信息

- 期号：49
- 标题：verl 源码解读 与 HybridFlow 编程范式讲解
- BV：BV1Cs7WzNEDX
- 时长：01:24:11（讲授约 47 分钟 + Q&A 约 37 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：童雨轩（清华大学计算机系本科生，verl core contributor；曾在 THUKEG、HKUST-NLP、CMU-LTI、字节跳动 Seed 实习。主持人口播姓名的字幕识别为「童宇轩/彭宇轩」，不稳定，以官网预告文为准）
- 相关论文：HybridFlow: A Flexible and Efficient RLHF Framework（Guangming Sheng、Chi Zhang、Ziling Ye、Xibin Wu、Wang Zhang、Ru Zhang、Yanghua Peng、Haibin Lin、Chuan Wu；香港大学 + 字节跳动），https://arxiv.org/abs/2409.19256（EuroSys 2025）
- 相关代码：https://github.com/verl-project/verl（原 volcengine/verl）
- B站链接：https://www.bilibili.com/video/BV1Cs7WzNEDX/
- 官网期号：49（预告链接未确认，仅标期号）
- 字幕原文存档：本地 `transcripts/EP49.txt`（1776 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（已存档）清洗提炼（经人工清洗；个别专名可能有识别误差，verl 在字幕中作「VO/VR/VL」、Ray 作「raid/re」、SPMD 作「SPMG/FTMD」，均以官方仓库与论文为准）。本版由原论文还原版升级而来：凡 talk 现场给出的内容标注（字幕口径），论文独有的实验数字单独标注（论文口径）。

## 一句话总结

这期是 verl 的"源码导读课"：嘉宾以 debugger 视角从入口文件出发，按执行顺序走完一条 PPO/GRPO 数据流，让听众理解 verl 为什么长这样——答案在 HybridFlow 的混合编程范式里：**顶层用单控制器写算法数据流，节点内用多控制器（SPMD）跑分布式训练/生成**，数据只在进程间搬运小体量的 prompt/response，重的参数与激活留在 worker 内部；灵活（换算法只写顶层几十行）和高效（相对基线 1.53×–20.57× 吞吐，论文口径）是同一个设计的一体两面。

## 核心

### 背景/问题：RLHF 框架要把数据流图映射到 GPU 集群

讲者把 RL 建模成一个 data flow graph：多个模型（actor、critic、reference、reward）、多个阶段（生成、准备 experience、训练）、多种 workload（同一模型在不同阶段的最优实现还不一样），本质是一个"有着复杂的 component 与时间空间依赖的调度问题"。框架要做的事：把这张图实现成一个具体 execution pattern，要最大化总吞吐。

但他强调目前这仍是一个理想的目标，verl 是用 HybridFlow 这样的技术（single controller 与 multi controller 相结合）去争取"灵活性和效率的权衡"。执行时有三类约束：数据有依赖就必须先后执行；同阶段、无依赖、又不在同一套设备上的模型可以并行；被分配到同一套设备的模型只能串行。

### 源码走读：入口 → 初始化 → fit 函数（单控制器层）

- **入口**是 `main_ppo.py` 的 task runner：先分配资源、最后进入 `fit`。资源一侧定义 resource pool（资源池）、把每个角色（workload）映射到资源池、实例化 resource pool manager 连同 config 一起传给 trainer。默认实现出于简单起见只用一个**全局资源池**，所有 workload 用全部 GPU、串行执行——讲者解释这是在大部分情况下较优的解、也避免用户过早考虑分布式细节。
- **初始化**最重要的事是初始化 worker 并分组为 worker group（一个 worker group 对应一个资源池）。这里讲者回答了一个初看奇怪的设计：为什么多个 workload 要挤在同一个进程里、每个 GPU 只维护一个进程？根源是 PyTorch 的显存管理——一个进程会 reserve 大于实际使用的显存，进程之间不能共享这部分 reserve；不同 workload 开不同进程就会互相浪费显存。verl 的解法是 **process colocation**：把不同 workload 融合到同一个进程里，在不同时间跑不同 workload；代码上把若干 worker class 合成一个新的 worker group class、拥有所有 worker 的方法。
- **fit 函数**是 PPO 例子的核心：真正的核心逻辑只有十几行——逐 epoch、从 dataloader 取 batch、转成 `DataProto`（贯穿全程的数据结构）后，逐阶段把数据经 Ray RPC 发给对应 worker group、等结果返回。讲者专门论证了为什么频繁的进程间通信不会慢：通信压力主要集中在参数、hidden states、优化器状态这些大数据上，而 RL 控制逻辑传递的只是 prompt 与 response，数据量相对很小；拿这个代价换来的是用 Ray 定义逻辑的灵活性。
- 约束小结（讲者口径）：默认 fit 是同步逻辑，因为串行执行不需要暴露异步接口。

### SPMD 层：worker 内部怎么调度（多控制器）

worker 内部用的是 SPMD（single program multiple data）范式——torchrun 就是最典型的 SPMD：多进程跑同一份代码、按 rank 等环境变量处理不同数据；DDP、ZeRO、FSDP，Megatron 的 tensor/pipeline parallel 背后都是这个模式。verl 基于 Ray 实现，需要自己管理 SPMD 的环境变量与调度，它给出的抽象是 talk 的重点：

- **register decorator** 给被修饰的函数附加四类属性：dispatch mode（数据如何从单控制器分发到各 worker）、execute mode（各 worker 如何执行）、blocking（是否阻塞等待）、materialize futures（参数里的 future 是否要先物化再传入）。decorator 本身只把配置暂存到函数的 magic attribute 里，不改行为。
- **dispatch function / collect function**：每种 dispatch mode 映射到一对函数。DP dispatch 把参数按 world size 切分（split 工具函数均分成若干块），collect 则把各 worker 结果 concat 后返回单控制器。要新行为就定义新的 proto 与函数。
- **execute function**：execute mode 映射到函数名（如 execute_all 是 execute_all_async 的别名），其逻辑是遍历 worker、取各自那份参数、经 Ray remote 调用、收集结果。
- **function generator** 把 method name、dispatch function、collect function、execute function 与 blocking 旗标拼成真正执行的函数：先 dispatch 切分，再 execute 远程调用，blocking 决定是否等待 future 物化，最后 collect 汇总返回。讲者总结："可以说这一段代码是 verl 它 multi controller 逻辑核心的核心。"
- 数据怎么读：单控制器读取全部数据，dispatch function 切分给每个进程，SPMD 进程不各自起 dataloader——"所有的数据流动都是由 single controller 管理的，multi controller 只负责用 SPMD 模式高效完成中间计算"。

### Programming Guide：五级定制深度

讲者按从简单到难排列用户定制点：

1. **只改数据**：符合约定字段（prompt、data_source、reward_model、extra info 等；verl 也接受 images/videos）即可直接训练；示例可看仓库的 `examples/data_preprocess`。
2. **自定义 dataset class**：通过 importlib 指定源文件与类名注入框架；class 只需能读数据文件、拿 tokenizer 把数据转成模型输入 id、接受 config 与 processor，内部转换完全自定义。
3. **自定义 reward**：同样可给源文件 + 函数名；naive 接口的函数接受 data_source、solution_str、ground_truth、extra_info，想改评分只需改 `compute_score`；想做跨样本批量逻辑（样本间对比、并行打分）则自定义 reward manager，可参考 PRIME reward manager 与 DAPO reward manager 这样的实现。
4. **自定义 loss**：没有专门接口，最简单的方法是在代码库里搜 `backward` 调用、找到 loss 定义处改或加新项——例子是在 actor 的 policy 更新里先 `compute_policy_loss` 算 policy gradient loss、再叠加其他 loss 项（如 SFT loss）最后 backward。
5. **大改训练逻辑**：算法逻辑都集中在 trainer 的 `fit` 函数里，override 它即可，如 DAPO trainer 实现 dynamic sampling：生成后先决定过滤/保留哪些 trajectory、拼回 batch、凑不够所需 prompt 数就回到采样阶段动态调整、凑够才进入训练。

### 实验/实战：论文口径的吞吐数字

注意：talk 本体是源码走读，没有现场跑实验。以下数字为 HybridFlow 论文（论文口径），talk 中未逐项展开：

- 论文在多种 RLHF 算法与模型规模上对比 DeepSpeed-Chat、OpenRLHF、NeMo-Aligner，总体吞吐提升 **1.53×–20.57×**；
- 70B 模型平均加速约 9.64×；
- 模型间切换（transition）开销最多降低约七成到九成；生成阶段在 NeMo-Aligner 中可占单轮迭代 81.2% 的时间，是基线的主要瓶颈。

机制侧的支撑（讲者口径）：3D-HybridEngine 在训练与生成两种并行布局之间直接重分片权重、零显存冗余，避免了基线把权重绕道主机内存/磁盘的税。

### Q&A 要点（讲者边界很清楚）

- **verl 现在最大的缺点**：同步逻辑在长文本训练时长尾分布的问题——generation 阶段少数 GPU 分到长 sequence 一直生成、其他 GPU 空转，资源浪费大。解法如 Kimi k1.5 式的 partial rollout，代价是带来 off-policy 问题、算法侧要弥补。
- **OOM 如何 debug**：先定位在哪一步 OOM。在 inference engine 重新 wake（load 到 GPU）时 OOM，就调小它的 `gpu_memory_utilization`；某模型 inference 时 OOM，多半 micro batch 太大、调小 micro batch size。verl 提供了 offload 工具把 params/grads/optimizer states offload 到 CPU，代价是 offload/reload 的 overhead；团队有 RFC 想写一份"怎么配参数才不 OOM、报了 OOM 如何 debug"的文档。
- **三类 batch size**：generation batch size（每次生成采样多少 prompt）、mini batch size（一次参数更新用多少 prompt；一个 batch 分若干 mini batch 做多次更新）、micro batch size（只跟计算实践有关，跟算法无关：余显存不够一次算完整个 mini batch 的梯度时切成 micro batch、梯度累加）. 命名上一个 verl 的特别处：所有 batch size 都以 prompt 为单位，与 SFT 惯用的单样本单位差一个"每个 prompt 采 $n$ 个 response"的系数。
- **auto mapping 为什么没有**：把 workload 映射到什么资源本质是自动并行问题，搜索空间巨大、目前还是 open question，verl 未支持。
- **SPMD 数据流**：single controller 读全部数据、切分后一次性分发给每个 worker（数据量小，没必要在分发上做过细优化）。
- **内存泄漏**：旧版本 VLM 有过内存泄漏、0.7.3 之后已修；async generation 是很新的 feature，可能还有 leak，欢迎提 issue 并附可复现脚本。
- **7B 模型 GRPO 训完不如 base model**：在一个已 instruction tuning 过的模型上继续 RL，本来就不保证提升；这类问题是算法问题、不是框架问题，框架角度没法给答案，需要具体问题具体分析（比如先查自定义 reward model）。
- **reference model 的作用**：源于 InstructGPT——reward model 不是 ground truth、可能被 hack，reference 是限制更新范围、避免为了最大化有问题的 reward 而崩溃的约束。
- **为什么 vLLM 默认 `gpu_memory_utilization` 在 verl 里设 0.5**：历史遗留——早期版本 inference engine 的 KV cache offload 默认没开，为了给训练引擎腾显存没把 utilization 设大；现在默认开 offload 后，推理引擎自己跑时可以开到 0.8–0.9。
- **提高生成 GPU 利用率**：整合 async engine 后可以做请求调度——先只发能打满各 GPU 的请求、剩余请求保留，监控哪个 GPU 利用率低就把新请求发给它。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| RLHF 吞吐（vs DeepSpeed-Chat/OpenRLHF/NeMo-Aligner） | 1× | 1.53×–20.57× | 论文 |
| 70B 模型平均吞吐加速 | 1× | 约 9.64× | 论文 |
| 模型切换（训练↔生成）开销降幅 | 基线绕道主机/磁盘 | 最多降低约 71%–89% | 论文 |
| 生成阶段占 NeMo-Aligner 单轮迭代时间 | — | 最高 81.2% | 论文 |
| fit 核心逻辑规模 | PPO 例子 | 十几行代码定义整条数据流 | 字幕 |
| vLLM `gpu_memory_utilization` verl 默认值 | vLLM 自身默认约 0.9 | 0.5（历史遗留），现 offload 后可自设 0.8–0.9 | 字幕 |
| 内存泄漏问题修复版本 | 0.7.3 及之前旧版本 VLM | 0.7.3 之后已修 | 字幕 |

## 可迁移

- 对 coding data / RL infra 工作的直接可试的点：
  1. 读任何 RL 框架先画数据流图再读源码：节点（模型角色）、边（数据依赖）、每节点的并行布局——verl 的 `main_ppo.py` 与 fit 函数就是按这个顺序组织的，照 talk 的 debugger 路线走最省力。定制时先选深度：能只改数据/reward 解决就不要动 fit；要动 fit 时参考 DAPO trainer 的 override 模式。
  2. 选框架时把"训练↔生成切换成本"列为一等指标：权重重分片是否零冗余、是否绕主机内存，直接决定 rollout 占比高时的总吞吐；而 batch size 语义各家不同（verl 以 prompt 为单位），迁移配置前先对齐一遍三类 batch size 的定义。
- Infra 视角：单/多控制器的分界线，本质是"哪些数据值得在进程间搬"——prompt/response 量级可以走 RPC 换灵活性，params/activations 必须留在 worker 内；放置策略（colocate vs split）没有普适最优，随集群规模反转，需要按规模实测。

## 疑问 / 下一步

- 3D-HybridEngine 的零冗余重分片在 MoE（专家并行布局与训练布局差异更大）上如何成立？talk 未展开，verl 后续版本对 MoE 的 resharding 路径值得单独追。
- 讲者点名的最大缺点是同步逻辑在长文本 generation 阶段的长尾问题；partial rollout 路线会把问题转嫁给算法侧的 off-policy 修正，效果与代价的实测对照值得后续跟进。

## 原文金句（1-2句）

> 「我们最终的一个愿景，就是说我们能够让用户只用关心如何去指定这样一个 data flow graph，然后在数学上定义他的行为，然后把所有的，包括更底层的 kernel 的一些优化，全都交给我们这个框架底层来实现。」——讲者对 verl 愿景的自述（字幕约 05:28–05:48）

> 「可以说这一段代码（function generator）是 verl 它 multi controller 逻辑核心的核心。」——讲者总结 function generator 在整个 SPMD 抽象中的位置（字幕约 36:50–37:01）

> 「它们总体来说所有的数据流动都是由 single controller 管理的，multi controller 就只负责用 SPMD 的模式去高效地完成它中间的这个计算。」——Q&A 中讲者对两层分工的一句概括（字幕约 67:04–67:15）
