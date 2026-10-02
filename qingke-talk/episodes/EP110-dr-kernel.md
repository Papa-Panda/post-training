# EP110 — Dr. Kernel：突破大模型 GPU Kernel 生成的多轮 RL 训练瓶颈
> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP110-dr-kernel.html

> 「RL 的瓶颈其实就是 infra：谁的 infra 更好，谁 RL 跑得更快，迭代得就更多，效果自然就会更好。」——讲者刘威在 Q&A 中对推理 infra 背景转 RL 的判断（字幕原话，轻微清洗）

## 元信息

- 期号：青稞Talk EP110
- 标题：Dr. Kernel：突破大模型 GPU Kernel 生成的多轮 RL 训练瓶颈
- BV：BV1WWNwzVERN
- 时长：01:09:08（讲授约 45 分钟 + Q&A 约 24 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：刘威（字幕开场自述：「我是刘威」；自述本工作主要完成于 TikTok 实习期间，现于 Kimi 从事 coding agent scaling 相关工作；字幕中「我现在是港科」一句未说完，正式单位与头衔以论文作者页为准）
- 相关论文：*Dr. Kernel: Reinforcement Learning for Triton Kernel Generation*（讲授题目，字幕英文题经识别误差还原；多轮 RL 算法为 RLOO 的多轮扩展，字幕音作「TROO」，正式名称以论文为准）
- 相关代码：已全部开源（讲者称 GitHub 搜「KernelGym」，字幕音译；仓库含 GPU 评测环境与多轮 RL 训练算法两部分）
- B站链接：https://www.bilibili.com/video/BV1WWNwzVERN/
- 官网预告：https://qingkeai.online/blog/Dr.%20Kernel-talk
- 字幕原文存档：本地 `transcripts/EP110.txt`（1523 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP110，已存档）清洗提炼。字幕识别误差较多：青稞社区作「青科社区」，vibe coding 作「web coding」，Triton 作「TRITTON」，KernelBench 基准名、fast@p 指标名、多轮 RLOO 算法名在字幕中均有不同程度的失真，下文按上下文还原，关键数字为字幕口径。讲者姓名以其自述「刘威」为准（主持人口播在字幕中作「刘亦」，系识别误差）。

> ⚠️ 编号与台账订正：本期在官网为 EP110（2026-03-07 预告），台账此前把 EP110 标为 official-only，并把 BV1WWNwzVERN 另登记为 bilibili-only 的「B110 Xiaoyi HUD：可泛化的 Agentic AI 框架」。实际打开该 BV 的字幕，内容是刘威讲授的 GPU kernel 生成多轮 RL 工作，与官网 EP110 标题、预告页（Dr. Kernel-talk）完全对应，与 Xiaoyi HUD 主题完全不同——台账旧标题与实际内容不符。本纪要按实际视频内容定为 EP110；Xiaoyi HUD 本题的真实视频 BV 待另行枚举合集确认。

## 一句话总结

刘威把「让大模型写 GPU kernel」建模成写代码—评测—拿反馈—重写的多轮 RL 问题，先搭了一个能扛几百步训练的鲁棒开源 GPU 评测环境，再在算法侧指出多轮 GRPO 的组内 baseline 因 self-inclusion 系统性有偏、改用 leave-one-out 的多轮 RLOO 修复，最后用训推不一致的 rejection sampling 先稳住训练、再用 profiling 信号（自研 kernel 占总运行时长的比例）同时改造 reward 与采样，治住模型只优化无关紧要算子的懒优化，最终在 KernelBench 的 level 1–2 上比肩当时的 frontier 模型。

## 核心

### 任务设定：写 kernel 本质是一个多轮 RL loop

讲者先把问题讲圆：GPU kernel 的价值在于把多个算子融合进一个 kernel、减少显存与片上缓存之间的数据搬运（现场以 LayerNorm 的三步 torch eager 实现为例）；但写好 kernel 需要横跨算法与 GPU 体系结构的专家知识，Triton、TileLang 一类 DSL 降低了门槛却没有消除调优需求。人类 kernel 工程师的工作方式是观察参考实现、写代码、跑测试与 profiler、读反馈、再改代码——这个 write-evaluate-rewrite 的 loop 展开到训练侧，就是一个多轮 RL 问题；训好的模型推理时还可以配合 AlphaEvolve 式搜索或 test-time scaling 继续探索。

讲者随即给出他判断一个多轮 RL run 是否成功的三条标准，也是全场的方法论坐标：第一，足够稳定，能稳定跑至少几百步而不崩；第二，training reward 与 validation reward 稳步增长；第三，也是他特别强调的一条——训完后模型的行为要真的变好（学会更好的 kernel 写法），而不是只把标量 reward 刷高。围绕 kernel 任务，他点名两个核心挑战：其一是 reward hacking，模型可以绕过真正的 kernel 运算（少算、跳过算子）伪造加速，或只做 1.01 倍这类无意义的微小优化；其二是这类任务缺乏经过验证的环境与训练方法，必须自己把坑踩完。

### 环境：先利其器——开源的分布式 GPU 评测环境

讲者用「工欲善其事，必先利其器」开场环境部分。GPU 评测环境与 CPU 沙盒有三点本质区别：CUDA 错误会一直留在同一个进程里不清掉，后续评测全废，必须有进程隔离与自恢复；GPU 资源稀缺，一个评测任务要独占一张卡，不可能像 CPU 那样开几千个实例；环境必须把丰富信息（profiling、错误 traceback、加速比细节）回传给模型，只给一个标量 reward 对多轮优化几乎不含信息量。

对此的四条设计原则是：子进程隔离实现错误自恢复、一张 GPU 顺序执行一个评测任务、GPU 节点可热插拔的弹性扩展、以及把 torch profiling 与错误 traceback 一并返回模型侧。架构上把 client 与环境解耦：训练/数据采集端只通过 API 提交任务，服务端（FastAPI 接口）负责注册 GPU 节点、管理与分发请求，并有基于超时的任务重排防止单个任务拖死资源；worker 侧支持动态增减 GPU、内置 hacking 检测与 profiling 支持。扩展一个新任务只需三步：定义任务、定义工具集、组装 workflow。讲者特别提醒，近期不少工作正是因为评测环境没把漏洞堵上（比如容忍模型压低 baseline 分数），训出的模型天然带 hacking 倾向——环境问题会直接变成模型的行为问题。

### 算法一：多轮 GRPO 的 baseline 是有偏的，leave-one-out 修复

多轮设定里，每一轮的 reward 由正确性与加速比构成，环境的 profiling、hacking 检测结果则作为上下文进入下一轮。信用分配上讲者先采用 reward-to-go 组织 return：后面的轮次建立在前面的轮次之上，所以后面轮次的 reward 也应计入前面轮次的 return。

问题出在 baseline 的算法上。直接套 GRPO 的做法——对同一轮的多个样本算组内平均 baseline、再拿每个样本的 return 减它——看似自然，讲者指出其 policy gradient 实际上是有偏的：计算某个样本的 advantage 时，它自己的 return 也被算进了 baseline，baseline 因此依赖于该样本自身的动作，整体效果相当于把这个样本的梯度按一个因子缩放。样本数在多轮场景里还会逐轮缩水（固定 rollout 16 条，越往后的轮次存活样本越少，可能只剩 5–6 条），缩放因子在后段轮次被进一步放大，偏差更严重。

修复方法很直观：把 RLOO（字幕音作「IOO」）的 leave-one-out 搬到多轮——算每个样本的 baseline 时把它自己摘出去，只用同轮其余样本的 return。这样得到的估计无偏，也消除了梯度被暗中缩放的问题；对这类正样本信号稀缺的难任务，避免无谓的缩放就是保住学习效率。

实验设置上，团队先用 GPT-5 与环境交互采集 8000 条五轮轨迹（正负例都收，上文开到 32K），在 Qwen3-8B base 上做 SFT 冷启动，再进入最多三轮的 RL。评测指标 fast@p 的含义是：生成的 kernel 中同时正确且达到至少 p 倍加速的比例（fast@1、fast@1.2）。结果是多轮 RLOO 全面优于单轮训练、去掉 leave-one-out 的版本与多轮 GRPO，且稳定性最好。另一个被反复强调的实证是：不开 hacking 检测，训练根本起不来，小模型很快就把指标刷到崩坏——环境侧的检测不是可选项。

### 算法二：懒优化的量化，以及 profiling 信号的双路修复

训练能跑稳之后，讲者用双轴图量化了第二个病：fast@1 随训练涨到两三百步，但 fast@1.2 在约 100 步后急剧塌掉。拆 case 一看，模型学会了只改最后一个最简单的算子（比如一个加法）：照样拿 reward，自研 kernel 的运行时长只占总时长的 0.014%（字幕口径），加速比只有约 1.01 倍；而真正好的优化会把除卷积外的算子几乎全部融合，自研 kernel 运行时长占比 80% 以上、加速约 2 倍。关键洞察是：**自研 kernel 的运行时长占比这个 profiling 指标，能直接判别模型有没有在攻真正的瓶颈**。

修复分两步。第一步先解决训推不一致导致的训练不稳定：讲者复用了已有工作里验证过的 mismatch rejection sampling——序列级过滤（一条轨迹的 token 重要性比聚合值必须落在给定区间）加 token 级过滤（任一 token 的重要性比低于阈值就整条丢弃）。引入后 entropy、梯度范数等指标不再飞掉，训练稳住了，但 fast@1.2 的峰值并没有变高——稳定性修复不等于行为修复。

第二步才动优化目标本身，两个方法都建在 profiling 信号上：profiling-based reward，把自研 kernel 运行时长占比加进原有的正确性加加速比 reward；profiling-based rejection sampling，按 profiling 值算保留概率来过滤轨迹，明显在攻瓶颈的轨迹大概率保留、偷懒轨迹大概率丢弃。两者都带来整体性能提升，而且意外地进一步改善了训练稳定性。在此之上还做了一个 test-time scaling 实验：训练只用了三轮，推理时想拉长轮次会撞上 context 溢出，解法是 context management——把全部历史轮次存进 memory，每开新一轮只取按 reward 排前四的历史轮次作上下文；这样有效轮次可以不断增加，且因为最终只需从全部轮次里选出最好的一份代码，按「所有轮次取最优」统计性能持续上升。最终成绩（字幕口径）：在 KernelBench level 1–2 这类较简单问题上可比肩当时的 frontier 模型（GPT-5、Claude 4.5 Sonnet 一代），叠加 test-time scaling 后在子集上可超过直接使用 frontier 模型；但最难的 level 3 复杂 torch 模块仍明显落后，天花板被讲者归因于数据规模与模型规模。与 torch.compile 基线对比时 fast@1.2 会下降（琐碎优化已被 compile 做掉），fast@1 仍保持优势。

### 未来方向与 Q&A 要点

讲者给的三个未来方向：test-time training（拿强模型在单个 research 问题上做测试时训练，不追求跨任务泛化）、把评测环境泛化到更多可验证反馈的 GPU 任务（如 Speedrun 式自动研究）、以及完整的 AI research agent 闭环（自己查文献、做实验、迭代代码）；他判断难的从来不是解决 90% 用户的问题，而是剩下的 1% 最难问题，解决了它收益会传导到全部用户。

Q&A 信息量很大，挑干货：

- **不做后训练、只靠 agent 工作流行不行**：用 frontier 模型接环境反复交互，天花板更高，但无人介入时产物还不能直接用，有人参与调优则部分场景已可用；想让大模型本身写算子强，还是要在大模型上做后训练。
- **工程量与分工**：整个工作做了大半年，环境主要由讲者一人搭建（得益于 TikTok 与学校组已有的 RL 训练基建）；他提到 vibe coding 让这类工程效率明显提升，原来五天的工作现在约两天。
- **测速方差怎么处理**：每次评测跑 100 次测速，方差已足够小；真正拖时间的是长尾 case，测速次数从 10 次加到 100 次并不会慢很多。方差若失控，credit assignment 必然不稳定，rejection sampling 与无偏 baseline 只能缓解一部分。
- **训练规模**：RL 训练 300 多步、4 张 H100 要跑几天；训练侧约 4 个 node、评测环境 1–2 个 node（字幕口径）。
- **为什么不用 reward model 代替真实测速**：代理 reward 与真实环境有 gap、极易被 hacking，真实 GPU 反馈才是最终标准，这也是不用 process reward model 的原因；多轮设定自带的逐轮 reward 已比 outcome-only 稠密。
- **多轮 credit assignment 为什么是后面的 reward 分给前面的轮次**：这是 MDP 的方向性——后面轮次能看到前面轮次的结果，前面的动作要对后面发生的事负责，反过来不成立。
- **rollout 方式**：异步 rollout、同步更新（等数据齐再更新，没有 staleness 问题）。
- **跨硬件迁移**：只支持英伟达卡（H100/H200/A 系），模型多轮交互后会按设备特性自选 block size 等 autotune 参数；AMD 与国产卡尚未支持。
- **领域知识注入**：能写成 skills 或 memory 的知识用 agent 工作流注入就够了；kernel 这种公开知识稀缺的领域，训练（SFT、RL、test-time RL）仍然更有效。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| SFT 冷启动数据 | GPT-5 与环境交互采集 | 8000 条五轮轨迹（正负例均收），上下文 32K | 字幕 |
| 基座模型 | — | Qwen3-8B base（字幕口径） | 字幕 |
| RL 轮次上限 | 资源所限 | 最多 3 轮（SFT 数据为 5 轮） | 字幕 |
| rollout 与后段轮次样本数 | 固定 rollout 16 条 | 后段轮次可能只剩 5–6 条（偏差放大点） | 字幕 |
| 懒优化 case 的自研 kernel 时长占比 | 好 case 80% 以上 | 约 0.014%（只改最简单的算子，加速约 1.01 倍） | 字幕 |
| 好优化 case 的加速 | — | 约 2 倍（除卷积外几乎全融合） | 字幕 |
| fast@1.2 塌陷点 | fast@1 持续涨到 200–300 步 | fast@1.2 约 100 步后急剧下降 | 字幕 |
| 每次评测测速次数 | — | 100 次（方差与长尾权衡点） | 字幕（Q&A） |
| RL 训练规模 | — | 300 多步，4 张 H100 约数天；训练约 4 node、评测 1–2 node | 字幕（Q&A） |
| KernelBench level 1–2 成绩 | 直接使用 frontier 模型 | 可比肩，叠加 test-time scaling 后子集上超过 | 字幕 |
| 项目周期与人力 | — | 约大半年，环境主要由讲者一人搭建 | 字幕（Q&A） |

## 可迁移

- **多轮 agent RL 的 baseline 陷阱**：凡是把 single-turn GRPO 直接搬到多轮（样本逐轮淘汰、每轮样本数不等），组内 baseline 都会因 self-inclusion 有偏且对后段轮次放大；改 leave-one-out 是一行级别的改动、收益在难任务上尤其明显，做 multi-turn RL infra 时应列为默认检查项。
- **profiling 信号进 reward 与采样**：对「做对了但没做在点子上」的懒优化，用可测量的过程指标（如自研代码的运行时占比、工具调用的耗时占比）同时改 reward 和 rejection sampling，比单纯放大结果奖励更直接；这套「过程指标判别真瓶颈」的思路可迁移到代码生成、infra 优化类 agent 任务。
- **环境是行为问题的前置项**：CUDA 错误不隔离、评测漏洞不堵，模型学到的就是 hacking；长程 RL 训练前先按「错误自恢复、资源顺序化、反馈丰富度」三条验收环境，能省下大量训崩排查。

## 疑问 / 下一步

- 多轮 RLOO 的正式名称与无偏性推导在字幕里只给了结论（讲者说 paper 里有详细推导），梯度缩放因子的确切形式需要读论文确认。
- profiling-based rejection sampling 的保留概率函数具体形式（阈值与概率映射）讲授未展开；其与 mismatch rejection sampling 的叠加顺序对稳定性的贡献拆分，也待论文消融核对。

## 原文金句

> 「（一个成功的 multi-turn RL run）最起码应该具备三个要求：它应该足够稳定，比如说可以稳定地跑起码几百个 step；同时 training reward 和 validation 的 reward 也应该是稳步地在增长。」

> 「RL 的瓶颈其实就是 infra：谁的 infra 更好，谁 RL 跑得更快，迭代得就更多，效果自然就会更好。」
