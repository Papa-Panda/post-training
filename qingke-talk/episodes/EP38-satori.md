# EP38 — Satori：通过训练 LLM 做自回归搜索来增强推理能力

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP38-satori.html

> 「R1 和 O1 是把 search 的能力给 internalized 到一个 model 里去，而不是需要一个 generator guided by 一个 verifier。」——讲者对本期方法定位的判词

## 元信息

- 期号：EP38
- 标题：Satori：通过训练 LLM 做自回归搜索来增强推理能力
- BV：BV1WiHezGEQN
- 时长：01:09:00（字幕末条 4135 秒 ≈ 01:08:55，含开场寒暄与 Q&A）
- 提炼日期：2026-10-02
- 分享嘉宾：MIT EECS 四年级 PhD 学生（字幕自述；英文名与本科院校字幕识别不清，不硬写）
- 提炼方式：B站 AI 字幕原文（已存档 `transcripts/EP38.txt`，1228 条，带时间戳）清洗提炼；个别专名有识别误差（如 Satori 字幕作「SATI / story」、GRPO 作「GR PO」、OpenRLHF 作「open i l h f f」），均以语境校正
- 相关论文：Satori（与 DeepSeek-R1 同期工作，讲者自述 2024 年 10 月底开始探索同类路线）

> 提炼方式说明：本纪要基于 AI 字幕原文（经清洗，专名可能有识别误差），信息来源以字幕为准；讲授中未逐项口播的幻灯片数字（如各 benchmark 的逐项分数）不臆测、不补写，只记讲者明确说出的数字与定性结论。

## 一句话总结

Satori 把「搜索」内化进单个 LLM：先用多智能体（generator + critic）合成的 1 万条 CoAT（Chain-of-Action-Thought）格式数据做 format tuning，让模型学会用特殊 token 显式触发 continue / reflect / explore 三种动作；再用 PPO + restart-and-explore（从自生成的中间状态重启 rollout）+ 三项奖励（规则奖励、ORM 偏好加成、反思加成）做 RL，并借传统 RL 的 kick-starting 做第二轮迭代。讲者的核心判断是：具体 RL 算法（GRPO 还是 PPO）无关紧要，关键在奖励设计、防 reward hacking 与全面 scale up。

## 核心

### 1. 背景：三条提升推理的路线与各自的账

讲者先把现有路线分成三类，并给出各自的成本结构：

- **蒸馏（distillation）**：门槛低、只需 SFT，但收集与过滤高质量 CoT 数据昂贵，且有「没有强模型就无从蒸馏」的先有鸡还是先有蛋问题。S1、LIMO 这类 less-is-more 工作本质是用少量数据激活 base model 已有的知识，对弱 base 的增益与跨域泛化存疑；讲者认同「SFT 主要为了 memorize knowledge，只有 RL 才能让 model 具备通用的泛化能力」这一判断。
- **外部搜索（search with external guidance）**：majority vote、reward model 打分、PRM 逐步打分做 beam search、MCTS——都不需要训练、test-time scaling 效果确凿，但推理时要同时部署 generator 与 reward model 两个模型，且要采样 64–128 次，大量 suboptimal 的 reasoning step 被直接丢弃，inference 成本浪费严重。
- **自我提升（self-improvement）**：STaR 式迭代 SFT，或 R1 / O1 式纯 RL。门槛最高（需要 scale 模型规模与训练算力），但讲者认为潜力最大。

讲者对 R1 的剖析占了相当篇幅，几个明确判断值得记：R1-Zero 让模型自由产生 thinking 步骤「对于 model 的行为是非常不可控的」，更像 proof of concept；反思能力「emerge 出来」的说法比较玄学，很可能 base model 本来就含有带反思的预训练数据。更合理的做法仍是先 SFT 控制行为、再 RL 激发能力——这正是 R1 本身的策略（cold-start SFT → RL → 用 RL 后模型重生成 600K CoT + 200K 非推理数据合成 800K 数据集 → 从 base 重新 SFT → 再 RL）。R1 需要两轮迭代的两个原因：一是先专注学 reasoning task 再混入非推理数据；二是第一轮 RL 的 policy 可能陷入 local optimum，从 base 重开 SFT 可以解决这个问题——这个 insight 与 Satori 自己的 kick-starting 设计直接呼应。

### 2. 方法：CoAT 三动作 + 两阶段训练

**问题形式化**：把 LLM 当作 agent policy，input question 是 initial state，每生成一个 reasoning step 是一个 action，state 是问题加上已生成的步骤，直到输出 final answer；最简单的 reward 是答案对错（对得 1、错得 0），目标是最大化 expected reward。但未经训练的 base model 根本不知道什么时候该反思、什么时候该探索新路径，也执行不了不同种类的动作。

**CoAT（Chain-of-Action-Thought）**：用 special token 把每一步的 action space 显式切开——

- `continue` token：沿当前路径继续推理，采样下一个 reasoning step；
- `reflect` token：回溯，触发自我验证与自我反思；
- `explore` token：放弃当前路径，提出新的解法。

与 R1 把反思隐含在 SFT 数据里、靠 "wait" 这类 transition phrase 触发不同，Satori 的做法是显式的 meta action：直接定义几个动作，用不同 special token 触发不同行为。

**第一阶段 format tuning**：难点是没有现成的 CoAT 格式 demonstration。讲者明确否决了人工标注与 MCTS + PRM 引导合成（Q&A 中承认最初试过 tree 式数据构建，「对 compute 的要求很高」，质量也难保证），最终采用 multi-agent data synthesis：generator 先生成多条初始轨迹，critic model 指出错误路径错在哪一步（比如第十步里第六步有问题）；对本身正确的轨迹，critic 随机挑一步做 verify。generator 根据 feedback 做 self-refinement，refine 后仍正确的路径收作 demonstration，仍错的返还 critic 进入下一轮。类比就是同桌解题：同桌只给 hint、指出哪一步错，不代做。实际只用了 **1 万条（10K）** 数据做 SFT，base model 就能完全 follow 带反思与探索的推理模式；讲者的判断是格式学习本身不难，难的是第二阶段真正用这种模式解题。

**第二阶段 RL（restart-and-explore）**：推理是 long-horizon 问题——可能要十几步甚至几十步才到正确答案，而 reward 只在最后出现，从 initial state 重新 rollout 对 RL 不友好。借鉴传统 RL 的 go-explore，Satori 不只从问题起点 rollout，还把模型自己生成的中间 state（partial trajectory）作为起点再出发。若中间 state 是错误的部分轨迹，就鼓励模型从错误 state 触发 reflect 动作、学会 self-correct——类比让 robot agent 从之前撞到的障碍物处重新出发探索、绕开障碍到达 target。

**奖励三项加权**（无 PRM，只用 ORM，与 R1 的取向一致）：

1. rule-based reward：最终答案对错的 0/1，不用任何 process reward；
2. preference bonus：ORM 给出的 soft score。动机是纯规则奖励太稀疏——难题一开始全做错、reward 全零，而 ORM 能分辨「都做不对」的轨迹之间细微的优劣，给出 positive 的软信号；
3. reflection bonus：从错误 state 出发成功解题给奖励；从本来就正确的 partial state 出发反而没解出来，给 penalty。直接塑造反思与纠错行为。

**Kick-starting 迭代**：第一轮 RL 后 policy 可能 converge 到 local optimum。受传统 RL kick-starting 启发，把当前 policy 当 teacher，蒸馏回 base 得到 student，再从 student 开第二轮 RL，得到 round 2。实验里 round 2 相对 round 1 确有提升，尤其在 AMC、AIME 这类难题上。讲者也顺带给了弱模型提点的通用配方：先用 format tuning + RL 训出一个强模型，再把强模型的能力蒸馏到弱模型——与 R1 把 R1-Zero 能力蒸馏的做法异曲同工。

算法细节：RL 用传统 PPO（非 GRPO），context window 4096（讲者称数学域大部分题目够用），总共训练约 1000 个 step（对比工业界动辄近万步），代码基于 OpenRLHF。讲者对 RL 算法的态度很明确：「具体的 RL 的算法其实不是最重要的，包括 DeepSeek 用的 GRPO、我们用的 PPO，其实 doesn't matter」，更重要的是 reward design 如何避免 reward hacking、奖励稀疏、训练稳定性，以及把一切 scale up（数据量、算力、时间、模型规模）。

### 3. 实验与分析（讲者口播口径）

- **主结果**：基座为 Qwen2.5-Math-7B base。Satori 超过同基座的 Qwen2.5-Math-7B-Instruct——而后者 SFT 阶段用了 2.5M 条数据，Satori 的 format tuning 只用 10K 条、RL 约 300K 条。注意讲者随即自我限定：数学域涨分不意外，因为训练数据本质都是数学数据。
- **跨域泛化（讲者最看重的结果）**：只在数学上训练，模型在其他 reasoning domain 上同样明显超过 Qwen2.5-Math-7B-Instruct，且与别家 instruct 模型 comparable 甚至更好。讲者认为这说明 RL 训出的 reasoning 能力能泛化到训练数据未覆盖的 domain。
- **CoAT vs 传统 CoT 对照**：同数据量、同参数、同 setup，传统 CoT SFT + 同规模 RL 训出的模型在较难 benchmark 上远远不如 Satori——难题需要模型利用反思与 self-exploration 才能解。
- **纠错能力细粒度分析**：专门挑「已给出 final answer 后又触发 reflection 并给出不同新答案」的样本，统计改对与改错的比例。RL 训练后，三个 benchmark（两个数学、一个域外）上把错改对的概率都远远高于把对改错——RL 真正激发了纠错能力，而不只是格式模仿。
- **反思的动态学**：response length 呈先降后升——训练前期模型反思能力弱、甚至「一本正经地胡说八道」，RL 的 reward 会让模型主动放弃 reflection 去走 shortcut（只要答案对、不反思也有 reward）；到约 200 step 后模型把反思学好了，又重新偏好用反思拿更高 reward，token length 与反思频率随之大涨。
- **难度自适应**：相对 format tuning checkpoint，题目越难（difficulty level 越高），提升越大、用的 token 也越多，二者成比例——讲者称之为 RL 带来的类 test-time scaling behavior。
- **RL vs 同规模 SFT**：用 RL 阶段同规模的 300K 数据做 SFT，效果远不如 RL 得到的 checkpoint。
- **demo**：AIME 2024 题目上模型先做错、触发反思后换全新方法做对；一道 MATH 题答案前后相同，但反思起到了 verify 作用；MMLU-Pro economics 题目出现多次反思、多次 explore，最后逐项检查选项给出正确答案。

### 4. 讲者的立场与边界（Q&A）

- 对 O1 的判断斩钉截铁：O1 就是单个模型自回归地完成推理，没有另一个 reward model 做 test-time guidance（讲者称有可靠消息）；O1 更可能用 MCTS 或 PRM 去合成 SFT 数据做 format tuning，但 inference 时大概率是单模型。O3 怎么做则明确说不知道。
- 给同行的建议很直白：不要再做数学 domain 了（「做数学这个 domain 是没有前途的」，AIME 都刷到四五十分、没有意义），去找 code、agent 等别的 domain；传统那套训 reward model + PRM 做 test-time search 的路线「建议不要再花时间去探索了」，那不是 R1 / O1 的技术路线。MCTS 可以用来合成 SFT 数据，但纯用 MCTS 训 reward model 做 test-time scaling 已很难有 novel contribution。
- 资源口径：SFT 数据合成对 GPU 要求不高；RL 阶段一个 node 8 张 GPU 是底线（越多越快）；训练这种模型对学术界不友好，主要卡在 RL infra 与 GPU 资源。这也是他们只试了 7B 的原因——讲者反复强调 RL 肯定模型越大越好：大模型 capacity 大、更容易 sample 到好轨迹、得到更好的 policy，小模型很难 sample 到更好的 trajectory。
- round 3 / round 4 没试过，只做到 round 2；没有明确答案的任务 reward 怎么设计是 open question，可用的信号是 preference reward。
- 不存在中英文混杂问题：讲者认为只要 SFT 让模型学会推理模式，就不会出现 language mixing——这反过来是他批评 R1-Zero 不可控的依据。

## 关键数字总表

| 指标 | 数值 | 来源 |
|---|---|---|
| format tuning 数据量 | 1 万条（10K）CoAT demonstration | 字幕（方法与 Q&A） |
| RL 阶段数据规模 | 约 300K 条 | 字幕（实验与分析） |
| 对照基线 Qwen2.5-Math-7B-Instruct 的 SFT 数据量 | 2.5M 条 | 字幕（实验） |
| 基座模型 | Qwen2.5-Math-7B base | 字幕（实验） |
| RL 训练步数 | 约 1000 step（工业界口径近 1 万 step） | 字幕（Q&A） |
| context window | 4096 | 字幕（Q&A） |
| R1 第二轮再训练数据构成（讲者转述） | 600K CoT + 200K 非推理 = 800K | 字幕（背景剖析） |
| 外部搜索典型采样次数（讲者口径） | 64–128 次 | 字幕（背景剖析） |
| 反思行为的转折点 | 约 200 step 后 response length 与反思频率回升 | 字幕（分析实验） |

## 可迁移

- **给 post-training 的可试清单**：反思/纠错能力不必靠 PRM 逐步打分——把「从错误中间状态重启 rollout + 纠错成功给 bonus、把对的改错给 penalty」做成奖励项，是比训 PRM 轻得多的塑造手段；且中间 state 由模型自生成，不需要外部构造错误样本。
- **format 先行、能力后至的配比**：本期 10K 条格式数据就足以让 7B base 完全学会三动作推理格式，RL 才是能力来源。做 coding data / agent 数据时可先小量验证「格式是否已学会」，把预算留给 RL 阶段的采样规模。
- **Infra 视角**：讲者把瓶颈说得很直白——RL 阶段的 infra 与算力（restart-and-explore 意味着 rollout 要支持从任意中间状态续跑、且需要 ORM 在线打分）才是学术界与工业界的真实差距；评测侧可直接抄「改对率 vs 改错率」这一对指标，它比 pass@k 更能区分模型是真会纠错还是只会换答案。

## 疑问 / 下一步

- ORM preference bonus 与 reflection bonus 的权重、以及 ORM 本身带来的 reward hacking 风险，讲者没有展开（只说最终 reward 是三项加权）；ORM 用什么模型、如何防被 policy 钻空子，待查论文。
- restart-and-explore 中间 state 的采样分布（挑错误轨迹的哪一步、错误与正确 state 的比例）直接影响反思信号的密度，字幕未给细节；这是复现时最需要回论文确认的一项。
- context 只有 4096、在数学域够用，但迁移到 code / agent 的长轨迹场景时 restart 机制如何与长上下文共存，讲者只给了方向（去做 code、agent），没有实测。

## 原文金句

> 「具体的 RL 的算法其实不是最重要的，包括 DeepSeek 用的 GRPO、我们用的 PPO，其实我觉得 doesn't matter。更重要的呢是 reward，reward 的 design 如何避免 reward hacking 的问题。」

> 「不要做数学这个 domain 了，做数学这个 domain 是没有前途的，最好就是找一些别的 domain，像 code、像一些 agent，这些领域。」

> 「大道至简，就是用最简单的方法。」（谈为何最终放弃 tree 式数据构建、只用 generator + critic 合成反思数据）
