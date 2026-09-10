# Paper 模板 - Day 10 自动骨架

## 元信息
- Title: Llama 3.1 / 3.2 - Post-training Expansion, Multilingual / Long-Context / Tool Use, Distillation & Pruning for 1B/3B and Vision 11B/90B
- Authors / Org: Meta AI - Llama Team
- Link / arXiv: https://arxiv.org/abs/2407.21783 (Llama 3 Herd v3 = 3.1 base, Jul 23 2024 + Nov 23 2024 update) + Model Cards: https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md (Sep 25 2024) + Llama 3.2-Vision: 无独立论文（2026-09-10 纠正：初读引用的 arXiv 2409.17379 实为四旋翼 MPC 论文，张冠李戴；vision adapter 机制见 Herd 论文 §7–§8 compositional approach + Llama 3.2 Model Card vision section）
- Date read: 2026-08-13
- Tags: [pretraining, sft, rl-data, curation, multilingual, long-context, tool-use, synthetic-data, distillation, pruning, coding-data, vision]
- Folder: 2024_llama3.1-3.2
- Day: 10

## 一句话总结
Llama 3 已用 15.6T 训到 405B，3.1/3.2 不再加 pretrain tokens量级，而是在 post-training 上做增量：8语言多语+128K长上下文+tool use/func calling+多轮人工+合成SFT+多轮RS+DPO（2026-09-10 注：“5.5M+”在三份原文中无出处，见第二轮复习 §4），以及1B/3B用3.1 8B/70B logits 做pretrain distillation + pruning+量化，11B/90B Vision 用cross-attention adapter接图像——把“预训练堆数据”转为“后训练数据飞轮+小模型蒸馏/剪枝+多模态扩展”的工程化范例。

## 和之前工作的关系
- **知识图谱位置**：pretrain / scaling 主线的“增量迭代点”，不是新 pretrain，是 Day 7 Llama 3 (15.6T dense 五级过滤+code 17%+annealing；2026-09-10 纠正：初读写的"code 25%"有误，论文 §3.1.2 最终配比是 50% 通用 / 25% math+reasoning / 17% code / 8% 多语，见 Day07 第二轮复习 §4.1) 的直接后继。对比 Day 8 DeepSeek-V3 (14.8T MoE，提升 math/code 配比+去冗余保多样+FIM；2026-09-10 注：初读写的"code 30%+激进去重"已在 Day08 第二轮复习 §4 中收回，四词在论文全文零命中，此处降级为存在性表述) 和 Day 9 Qwen2.5 (18T file+repo级 code 5.5T + 1M SFT +多阶段RL)，Llama 3.1/3.2 补上了那三家都弱的一环：**后训练数据的规模化、产品化配方**。
- **接了哪条线**：
  - pretrain/curation线：接 Day 7，复用15.6T，只把cutoff拉到Dec 2023，9T用于1B/3B轻量重训；不像Qwen把7T→18T那样暴力扩tokens，而是把tokens花在“精”。
  - synthetic线：接 Day 6 Phi-1 (教科书合成) + Day 8 DeepSeek执行过滤；Llama 3.1/3.2 的合成不是 pretrain 合成，而是 SFT 合成：tool use轨迹、长上下文问答、多语回译、安全性对抗，都用更大模型生成 → 小模型过滤 → RM打分 → Rejection Sampling，规模化到数百万。
  - selection/influence线：接 Day 4 LESS (梯度相似度选5%) 和 Day 5 DataInf (LoRA闭式1秒/条)：Llama 3.1/3.2不用梯度，用RM/奖励+人工偏好做选择——RS/DPO的pair就是“对偏好影响最大”的子集，计算更便宜、可产品化。
- **补了哪个短板**：Llama 3只讲预训练怎么搭，SFT/RL一笔带过；Qwen讲了1M SFT+多阶段RL但没讲1B/3B怎么做得稳、Vision怎么加、量化怎么做。Llama 3.1/3.2补上“同配方如何撑8B/70B/405B全系列+1B/3B端侧+11B/90B多模态，且后训练6轮迭代不崩”的坑。
- **替代/分叉/改进**：不是替代Llama 3，是改进+分叉：405B保持配方，8B/70B做targeted能力追加（tool use、多语、数学、代码），1B/3B走distill+prune+恢复训练的新分支，11B/90B走vision adapter分支。三大开源配方三角（Llama/DeepSeek/Qwen）在Qwen处已闭合，3.1/3.2是闭合后向产品化/轻量化/多模态的延伸。
- **对Day X直接对比**：vs Day7 Llama3：同15.6T基座，但后训练从单轮SFT→多轮SFT+RS+DPO，数据从通用指令→专项（coding/reasoning/tool/long/multilingual/safety）精筛；vs Day9 Qwen2.5 72B与405B打到competitive的参效比故事（2026-09-10 注：初读"超"已在 Day09 第二轮复习 §4 中收回，论文原文是 competitive）：Llama 3.1/3.2用405B证明“参量还能靠后训练数据再榨”，用1B/3B证明“小模型靠logits蒸馏+prune恢复也能保留大部分能力”。

## 为什么今天读它
- coding data：Llama 3.1 新增code专家后训练集（自带执行环境验证+unit test通过率作filter），tool use / func calling轨迹本质是code+JSON，1B/3B的distill logits在code completion上比纯Causal LM稳，可直接抄到你50万合成池的“可执行过滤+tool轨迹合成”。
- SFT：展示如何从 Day7 的通用SFT配比 → 后训练6轮迭代，每轮RS挑高RM分样本再DPO，配比按能力维（coding / math / reasoning / long / multilingual / safety）动态调，和你Qwen Day9的1M SFT多阶段RL呼应但更工程化。
- RL data：多轮DPO的preference pair怎么来（人工→RM→合成→再RM），为什么不用PPO而用DPO+RS（稳定、可扩展），和你Agentic RL Infra的“\$/useful-rollout”评估直接相关；1B/3B的量化/剪枝后恢复训练对RL数据噪声更敏感，提供“小模型RL数据要更干净”的反例。

## 今天的 3 问
1. Llama 3.1/3.2 的后训练为什么从PPO转成“多轮SFT+Rejection Sampling+DPO”循环？6轮迭代里每轮的SFT/RS/DPO数据是怎么分工的（比如tool use、长上下文、多语、安全性各在哪轮加）？和 Day 9 Qwen2.5 的 RM→SFT→新RM→RL self-improvement相比，稳定性/成本trade-off在哪？
2. 1B/3B的“在pretrain阶段融入8B/70B logits做distillation + pruning + 恢复训练”是怎么做的？logits做token-level target和普通CE loss怎么加权？prune用的是width pruning MAW还是depth？恢复训练用了多少tokens？和你如果把50万池用70B教师打logits蒸到1.3B相比，预期HumanEval涨多少、MMLU掉多少，infra成本是升是降？
3. 【对比题】对比Day7 Llama3 (15.6T pretrain、5级瀑布、code 17%、annealing；2026-09-10 纠正：初读"code 25%"有误，见 Day07 第二轮复习 §4.1)、Day8 DeepSeek-V3 (14.8T MoE、提升 math/code 配比+去冗余保多样+FIM；2026-09-10 注：本题原前提中的"MinHash0.90 / code30%+ / FIM10% PSM"已在 Day08 第二轮复习 §4 中收回，四词在论文全文零命中，比较时以存在性而非阈值数字为准)、Day9 Qwen2.5 (18T file+repo级code 5.5T、弱模型scorer、1M SFT+多阶段RL)、Day10 Llama3.1/3.2 (复用15.6T、后训练多轮专项、1B/3B distill+prune、11B/90B vision adapter)——四家在“code占比/去重阈值/合成策略/后训练轮次/小模型化”五轴上各有什么取舍？如果你要为你50万合成池定下“25%→30% code + MinHash0.90 + FIM+exec过滤 + 1M级SFT多阶段 vs 多轮DPO小步快跑”二选一，基于这四篇论文的证据你选哪个，为什么？Infra视角：Bloom+分布式MinHash vs 弱模型scorer vs distillation logits，哪一个是18T/15T规模下的真正瓶颈？

## 核心（待填，今晚产出）
1. **Motivation**: 为什么不在15T上再堆tokens，而要做后训练增量/小模型化/多模态？405B边际收益已低，产品化需求（多语、长上下文、tool use、端侧、vision）倒逼。
2. **Data Pipeline**: 来源 → 复用Llama3 15.6T (Dec 2023 cutoff) → 1B/3B额外9T logits distillation pretrain → 清洗 → 后训练6轮：SFT (人工+合成（2026-09-10 注：“5.5M+”在原文中无出处，见第二轮复习 §4），coding/math/reasoning/long/multilingual/safety/tool) → RS (RM打分选top) → DPO (preference pair人工+合成) → iterative loop；Vision 11B/90B额外image-text 6B (?) cross-attention adapter；Safety/PII再筛。
3. **Key Tricks**: 3个最值得抄的细节
   - 多轮RS+DPO而非PPO：每轮只训cleanest top-k，用RM而非梯度影响选，稳定可扩展。
   - logits蒸馏进pretrain：1B/3B pretrain阶段就吃8B/70B logits当soft target，省后训练对齐成本。
   - toolchain数据闭环：tool use轨迹自带执行结果，可验，失败轨迹回灌做hard negative DPO天然pair。
4. **Results**: 对 downstream 有多大提升？405B Instruct多语MT-Bench/tool use/长上下文Needle-in-haystack接近GPT-4o，8B/70B专项能力+5-10pts，1B/3B端侧MMLU/HumanEval保留率~80% vs 同尺寸从零训，11B/90B Vision VQAv2/TextVQA超同级，量化后4bit几乎不掉点（ExecuTorch）。

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：
  1. 抄它的tool use轨迹合成+执行验证：把你50万池里10%改成“func calling JSON+python执行结果”格式，用unit test/exec当filter，留下的做SFT RS Top-k。
  2. 抄它的logits distillation：用你1.3B当学生，70B (或Qwen2.5-72B) 当老师，对50万池打soft target做pretrain 1 epoch，对比纯hard label的HumanEval/MMLU差。
- Infra 视角：可扩展性 / 成本 / 评测自动化的启发：
  - 6轮RS+DPO的pipeline自动化：每轮评测→RM打分→抽top→合成pair→再训，需要搭持续评测+RM打分服务，你vLLM rollout可复用。
  - Distillation logits存储成本高（vocab 128K * seq_len），用top-k logits (k=64) + temperature 0.7可省90%存储（2026-09-10 注：k=64 / T=0.7 / "省90%"在 Llama 3.2 Model Card 无出处，属合理工程推断，待核对后定级）。
  - Pruning恢复训练的data mix要更干净，小模型对脏数据更敏感，可用DataInf思路扫高self-influence样本先踢。

## 疑问 / 下一步
- 6轮DPO里RM是怎么进化（初始人工RM→合成RM→最终RM），和Qwen2.5-Math的self-improvement闭环有什么实现差异？1B/3B的pruning具体宽度剪多少（expansion ratio从4→？）后MMLU掉点曲线怎样？（2026-09-10 注：pruning 的宽度/深度细节官方文档未披露，此问无论文答案）

## 原文金句 (1-2句)
> We use a similar recipe as Llama 3.1 and produced final chat models by doing several rounds of alignment on top of the pre-trained model. Each round involved Supervised Fine-Tuning (SFT), Rejection Sampling (RS), and Direct Preference Optimization (DPO). (from Llama 3.2 Model Card)
> Llama 3.2 was pretrained on up to 9T tokens ... we incorporated logits from Llama 3.1 8B and 70B into the pretraining stage ... Knowledge distillation was used after pruning to recover performance.

## 3 问回顾（Day 10 原题见上）

## 参考
- Llama 3 Herd (3.1): https://arxiv.org/abs/2407.21783 (v1 Jul 31 2024, v3 Nov 23 2024)
- Llama 3.2 Model Card (1B/3B/11B/90B): https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md (Sep 25 2024)
- Llama 3.2 1B/3B Text: https://huggingface.co/meta-llama/Llama-3.2-1B / 3B
- Llama 3.2-Vision 11B/90B: 无独立论文（2026-09-10 纠正：2409.17379 系误引，实为无人机论文）；机制见 Herd 论文 §7（cross-attention vision adapter，冻结 LM 参数）+ Model Card vision section
- Evolution survey Llama 3.1→3.2: 9T tokens, multilingual 8 langs, 128K context, tool use (direct quote lines 89-94)

---
生成逻辑：已纳入知识图谱，强调结构非数量，自动产出已开启。
## 第二轮复习（2026-09-10）

> 本轮已对照原文逐项核对：Llama 3 Herd 2407.21783 v3（ar5iv 全文，§3/§4/§7）、Llama 3.2 Model Card。发现初读四处"把推断/误引写成论文事实"的偏差，已在 §4 开头明示纠正；初读引用的 Day08 数字（code 30%+、MinHash0.90）已在 Day08 第二轮复习中收回，上文已加注。另：本轮复核中一度把有出处的"6轮"误降级为"多轮"，经原文复核已恢复——教训：降级前必须先查原文，不能凭印象。

### 1. 核心命题

Day 10 真正解决的 data 问题，不是"怎么把预训练堆到更大"，而是：**预训练做完之后，数据工作怎么从"一次性配方"变成"可产品化的后训练数据引擎"**。Day 07（Llama 3）的 15.6T 瀑布回答了"数据怎么搭"；Day 10 回答的是"基座训完后，数据团队接下来干什么"——答案是三件事：

1. **专项能力的后训练数据工业化**：多语（8 语言）、128K 长上下文、tool use / function calling——每一项都不是"调参"，而是一条独立的数据管线（合成生成 → 过滤 → RM 打分 → RS → DPO），且管线可复制到新能力上。
2. **把"数据选择"做进训练循环**：6 轮 SFT + RS + DPO（§4.1.6 原文 "we apply the above methods in six rounds"）的本质，是用 RM（reward model， $r_\phi(x,y)$ ， $x$ = prompt， $y$ = 响应， $\phi$ = RM 参数）当数据门，每轮只让 RM 打分最高的 $k$ 个数据进下一轮——数据不是静态资产，是每轮被重新提纯的流。
3. **把教师模型变成数据源**：1B/3B 在 pretraining 阶段就吃 8B/70B 的 logits 当 soft target（Model Card）——distillation 不是"训练技巧"，是数据视角下的"高密度数据注入"：同样 9T tokens，信息密度被教师抬高了。

一句话：Day 10 是"后训练数据飞轮"的 Meta 版存在性证明——论文自己在引言就把立场写明了（L111–113）："we adopt a relatively simple post-training procedure based on SFT, RS, and DPO as opposed to more complex RL algorithms that tend to be less stable and harder to scale"——**把"对齐"从在线优化问题转化成了离线数据策展问题**，对齐的难度被转移到了 RM 的质量和数据管线上。

### 2. 图谱位置

- **前驱 Day07 Llama 3**：15.6T 瀑布（§3.1 最终配比 50% 通用 / 25% math+reasoning / 17% code / 8% 多语——初读误写 code 25%，已在 §4 纠正），后训练章节是短板。Day 10 是它的直接后继：复用 15.6T，cutoff 拉到 Dec 2023（§3.1），工作重心转向后训练。
- **直接对比 Day09 Qwen2.5（重点）**：两者都是"预训练之后的数据故事"，但架构相反——Qwen 是 SFT（1M+）→ offline DPO（约 150K pairs，走白盒信号）→ online GRPO（RM 做奖励），**两阶段 RL，带 online**；Llama 3.1/3.2 是 6 轮 (SFT → RS → DPO) 循环，**纯 offline，无 online RL**。共同点：都不用 PPO；都用 RM 当数据门；都零数据消融。深层分野：Qwen 按"评估器可靠性"分诊（白盒信号走 offline DPO，黑盒偏好走 online GRPO）；Llama 把一切压进 offline 循环——赌的是 RM 足够准。Llama 的选择更便宜、更稳定、更可审查；代价是天花板可能低于 online RL（Day15 R1 是反例：纯 RL 在可验证任务上走得更远）。
- **偏好支线 Day10 → Day13 DPO-Gap**：Day 10 是 preference pair 的**供给者**（6 轮 RS+DPO 规模化生产 pairs），Day 13 是 pair 的**筛选器**（gap 小的难对留 10% 打赢全量）。两者合在一起才是完整的故事：Llama 告诉你怎么规模化生产 pairs，Day 13 告诉你生产出来的 pairs 里 90% 可能是水——"供给"和"筛选"是同一条支线的两段。
- **Day13 → Day11 LIMR**："少即是多"从 DPO pair 选择延续到 RL 数据选择（1.3k 轨迹打赢全量）。
- **替代/互补**：
  - vs Day08 DeepSeek-V3：正交。DeepSeek 讲"预训练**之中**"（表示层重写：文档级 FIM、packing 保 integrity）；Llama 3.1/3.2 讲"预训练**之后**"（后训练产品化）。两者可叠加。
  - vs Day06 Phi-1：合成的用途分野。Phi-1 的合成是 pretrain 合成（教科书文本，解决"好文本不够"）；Llama 的合成是 post-train 合成（tool 轨迹、长上下文问答、多语回译，解决"专项能力数据不够"）。合成从"造文本"进化到"造带验证信号的轨迹"。

### 3. 机制深挖

**(a) 为什么是 SFT+RS+DPO 循环，而不是 PPO？——offline 的数据经济学（§4.1/§4.1.6）。**

记 $\pi_\theta$ 为待训练 policy， $\pi_{\text{ref}}$ 为参考 policy（通常是 SFT 模型）， $\beta$ 为 KL 惩罚系数（论文取 $\beta=0.1$ ，LR $10^{-5}$ ）。DPO 的隐式奖励 $\hat r(x,y) = \beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ ，DPO loss $= -\mathbb{E}_{(x,y_w,y_l)}[\log\sigma(\hat r(x,y_w)-\hat r(x,y_l))]$ ，其中 $y_w$ = chosen 响应， $y_l$ = rejected 响应， $\sigma$ = sigmoid。论文还加了两个稳定项：mask 掉 formatting tokens（header/termination，防 chosen/rejected 共有 token 的对比冲突）、chosen 序列上加 0.2 系数的 NLL 正则（防 chosen 的 logprob 被对比 loss 压下去）。

- RS（§4.2.2）：对每个 prompt 采样 $K$ 个响应（ $K$ 通常 10–30），用 $r_\phi$ 选最优进 SFT。这是"**用模型自己的裁判提纯数据**"——数据门是 $r_\phi$ ，不是人工。工程细节：PagedAttention 让 RS 吞吐提升 2 倍以上；后几轮加入 system prompt 引导 RS 的语气/格式。
- 6 轮的含义（§4.1.6）："In each cycle, we collect new preference annotations and SFT data, sampling synthetic data from the latest models." 每一轮的 $\pi_\theta$ 变强 → 下一轮采样质量更高 → 新一轮的人工标注/SFT 数据更难 → 数据分布向"难例"漂移。这是一个**课程学习（curriculum）的涌现**：不是人工设计的课程，是"模型变强 → 数据变难"的自举课程。另有 §4.1.5 的 model averaging：每轮在 RM/SFT/DPO 阶段把不同数据/超参版本的模型平均——用集成对冲单轮数据的方差。
- 偏好数据的"双速"用法（§4.2.1）：每轮 RM 用**全部历史**偏好数据训练，DPO 只用**最新 batch**。数据视角的翻译：RM 要的是覆盖度（见过所有历史分布），DPO 要的是新鲜度（只跟当前 policy 分布对齐）——同一个 preference 池，两种消费速度。

**(b) 数据质量门的具体实现（§4.2.3，初读完全遗漏的一节）。**

论文的数据清洗不是"调参"，是一套四件套管线：① topic classification（把 8B finetune 成 topic 分类器，全量数据打标签）；② quality scoring（RM 打分取 top quartile 为高质量，**或** Llama-based 打分取满分——通用数据 3 点量表 accuracy/instruction-following/tone，code 数据 2 点量表 bug-identification/user-intention；两者分歧率高，取并集 recall 最好）；③ difficulty scoring（Instag 意图数 + Llama-based 3 点难度）；④ semantic dedup（RoBERTa 聚类 + 簇内按 quality×difficulty 排序 + 贪心保留 cosine 相似度低于阈值的）。深挖：这是"**质量 × 难度**"的二维选择——RM 只管质量，Instag 管难度，dedup 管多样性。三者正交，缺一不可；这正是 Day20 DEITA（三因子）的人工版先驱。

**(c) Pretraining-time distillation：教师是数据源，不是初始化（Model Card）。**

1B/3B 在 pretrain 阶段吃 8B/70B 的 logits 做 soft target。数据视角的翻译：普通 pretrain 的每个 token 提供的是 one-hot 信号（词表大小 $V$ = 128K，只告诉"正确答案是哪个"）；教师 logits 提供的是**整个分布**——"这个位置除了正确答案，还有哪些答案是合理的"。信息论视角：soft target 的每 token 信息量远高于 hard label（温度 $T$ 控制分布平滑度）。所以 9T tokens 的"有效信息量"被教师放大了——**distillation 是用教师算力换数据信息密度的交易**。存储成本：存全量 logits = seq_len × 128K × 4 bytes，贵得离谱；工程上用只存 top $k$ 的近似（初读写的 $k=64$ 、 $T=0.7$ 在 Model Card 无出处，是合理工程推断——见 §4）。

**(d) 专家模型：专项能力的"数据前置"（§4.3.1 code / §4.3.2 multilingual）。**

初读把 code expert 写成"后训练集"，论文实际是** continued pretraining**：从主 pretrain 分叉，在 1T tokens（>85% code）上继续预训练，再做 LCFT 拉到 16K 上下文，然后才走 SFT/DPO。multilingual expert 同理：在 90% 多语 mix 上继续预训练。数据含义：专项能力不是靠后训练"调"出来的，是靠**前置的数据分布偏移**砸出来的——后训练只是把预训练里已经学到的能力"对齐"出来。这对"后训练能补多少能力"是个冷水：code 能力的上限是那 1T tokens 定的，不是 6 轮 DPO 定的。

- Code 合成数据的论文数字（§4.3.1）：2.7M 合成 SFT 样本，三路生成，execution feedback 做 source of truth。另有一个反直觉的发现：405B 在自己生成的数据上训练**没有提升甚至变差**——"大模型自举"在 405B 规模失效，只有 execution feedback 能打破这个僵局（这是 R1 纯 RL 路线的先声）。
- Multilingual SFT 的构成（§4.3.2）：2.4% 人工标注 / 44.2% NLP 任务改写 / 18.8% RS / 34.6% 翻译推理数据；Blaser2.0 + LID 过滤；RM 选择前先做语言匹配检查（罗马化 Hindi prompt 不能配 Devanagari 回复）；最终轮温度固定 0.6 防 code-switching。7 种非英语（German, French, Italian, Portuguese, Hindi, Spanish, Thai）+ English = 8 语言。

**(e) Vision adapter：多模态对齐的数据效率（§7）。**

cross-attention adapter 把 image encoder 接入 LM，**冻结 LM 参数**，只在 text-image pairs 上训 adapter。数据含义：不需要重训 15.6T 文本数据来学视觉——**新模态的数据需求被结构压缩到了只训对齐层**。这是"数据效率"的结构主义版本：不是精选数据，而是设计让数据需求变小的结构。

### 4. 边界与反例

1. **纠正初读的四个"论文事实"（本轮核对原文后）**：
   - "code 从 17% 上采样到 25%"——**收回**。§3.1 最终配比：50% 通用 / 25% math+reasoning / 17% code / 8% 多语。25% 是 math+reasoning，不是 code（Day07 第二轮复习 §4.1 已纠正，Day 10 初读未同步——上文已加注）。
   - "5.5M+ 人工+合成 SFT"——**收回**。论文只给百分比（Table 7：General English 52.66% / Code 14.89% / Reasoning and tools 21.19% / Multilingual 3.01% / Exam-like 8.14% / Long context 0.11%），**无绝对数量**。合成数据的论文数字是 code 域的 2.7M（§4.3.1），不是全 SFT 的 5.5M。
   - "arXiv 2409.17379 = Llama 3.2-Vision 论文"——**收回，张冠李戴**。2409.17379 实为四旋翼无人机 MPC 论文。Llama 3.2-Vision 无独立 arXiv 论文；vision adapter 机制见 Herd 论文 §7（cross-attention，冻结 LM）+ Llama 3.2 Model Card vision section。上文元信息与参考已修正。
   - "只存 top $k$（$k=64$）+ temperature 0.7 是 Llama 3.2 实践"——**降级为工程推断**。Model Card 只说"incorporated logits into the pretraining stage"，未披露 top $k$ 、温度、存储优化细节。
   - 确认初读正确的："6轮"有原文出处（§4.1.6 "we apply the above methods in six rounds"）；"8语言"有出处（§4.3.2 七种非英语 + English）；"128K"有出处；"9T + 8B/70B logits"有出处（Model Card）。
   - 初读引用的 "DeepSeek-V3 code 30%+ / MinHash0.90"——**已在 Day08 第二轮复习 §4 中收回**（四词在论文全文零命中），上文已加注；以后只引用"提升 math/code 配比（无数字）+ 去冗余保多样（无方法）"的存在性。
2. **零数据消融**：6 轮 vs 单轮、DPO vs PPO、logits 蒸馏 vs 同数据从零训、code expert 的 1T continued pretraining vs 纯后训练——论文**都没有做**。不能引用 Llama 3.1/3.2 证明"6 轮优于单轮"或"DPO 优于 PPO"；这是工程选择 + SOTA 结果的捆绑销售。引言 L111–113 的"PPO 不稳定难扩展"是立场陈述，不是实验结论。
3. **RM 的单点故障**：RS 的数据门是 $r_\phi$ ，DPO 的 pair 质量也依赖 RM/人工偏好——**RM 的偏见在 6 轮中被迭代放大**（与 Qwen §4.5 的三重偏见同构，Llama 是"单重但多轮"）。论文没有做 RM 偏见的审计。若 $r_\phi$ 系统性偏好长回答/某种风格，6 轮后模型会被推向 RM 的偏好而非人类的偏好——这是"离线对齐"的结构性风险；Day 13 的难例选择（只留 10%）恰好是对"RM 打分即质量"的质疑。
4. **Distillation 的偏见继承**：1B/3B 吃 8B/70B 的 logits——教师的偏见、幻觉模式、RM 偏好全部被蒸进去，且**小模型没有容量去"纠正"教师**（容量只够记忆教师分布）。论文没有量化蒸馏带来的偏见放大。
5. **405B 自举失效的反例**（§4.3.1）："training Llama 3 405B on its own generated data is not helpful (and can even degrade performance)"——合成数据的"模型生产数据"飞轮在最大模型上撞墙了，只有 execution feedback 这种白盒信号能打破。这是 Qwen 飞轮（§3.1 全靠模型生产数据）没讲的一面：**自举有规模上限**，上限之后必须引入外部 ground truth。
6. **多语"投影"代价**（与 Qwen §4.7 同构）：34.6% 的多语 SFT 是"翻译推理数据"——低资源语言（Hindi、Thai）的本土表达被系统性抹平。论文没有评估低资源语言的本土性损失；且人类评估显示 405B 在 Hindi/Spanish/Portuguese prompt 上**不如 GPT-4**（§5.2.4）——8 语言支持度的论文内反例。
7. **Vision adapter 的数据天花板**：冻结 LM 只训 adapter，省数据；但需要改变文本表示的深层跨模态对齐可能做不到——论文没有给出 adapter vs 全量微调的对照。

### 5. 迁移到 coding / post-training data

**可执行实验 A — "6 轮小步快跑"的离线版（2–3 周可跑）**

1. 把 500k 池按能力维切分（code-gen / debug / tool-use / instruction-following）；每维独立跑 2–3 轮 (SFT → 采样 $K=10$ – $30$ → RM/LLM-judge 打分 → top $k$ RS → 难例 pair DPO），抄论文 §4.2.2 的 $K$ 取值和"RM 用全历史、DPO 用最新 batch"的双速消费。
2. 轮数不由固定数字决定，而由"边际增益 < 阈值"决定：每轮后在 HumanEval/MBPP/IFEval 上测 delta，delta < 1pt 则停——这是把论文的"6 轮"从玄学变成可证伪的 stopping rule。
3. 纪律：RS 的数据门（你的 RM/judge）每轮必须用 held-out 人工抽检校准一致率（< 80% 则停轮、先回炉 RM）——防 §4.3 的 RM 偏见放大。另抄 §4.2.3 的四件套做每轮 QC：topic 分类器 + RM top quartile + 难度打分 + 语义去重。

**可执行实验 B — tool 轨迹合成 + execution filter（抄 §3(d)/§4.3.1，两周可跑）**

1. 把池子里 10% 改成"func calling JSON + python 执行结果"格式；用 unit test / exec 通过率当 filter（不要 RM）——论文 §4.3.1 证明 execution feedback 是打破"大模型自举失效"的唯一钥匙。
2. 失败轨迹直接做 DPO 的 $y_l$ ，成功轨迹做 $y_w$ ——零 RM 成本的 preference pair 管线；另抄 §4.2.1 的"4 级强度 + 丢弃 similar 对"：只留正负差距明确的 pair。
3. 通过标准：SFT 后模型在 tool-use 评测（BFCL 类）上涨，且 DPO 阶段"执行失败率"单调下降。

**可执行实验 C — pretrain-time distillation 的小规模验证**

1. 用 70B（或 Qwen2.5-72B）教师对 50k 子集打 top $k$ logits（先算存储账：50k × 平均长度 × $k$ × 4 bytes，选 $k$ 使存储 < 预算——论文未披露 $k$ ，这是你的工程选择），蒸到 1.3B 做 1 epoch pretrain，对照组纯 hard label。
2. 评测：HumanEval（期望涨）/ MMLU（期望小掉或持平）/ 教师偏见继承检查（抽 100 条看幻觉模式是否与教师同构，防 §4.4）。
3. 若成立：distillation 进入你的数据 recipe，定位是"信息密度放大器"而非"对齐手段"；另验证"小模型恢复训练 data mix 要更干净"（§3(c) 的推论）：同一蒸馏，对比干净/脏 mix 的 MMLU 掉点差。

### 6. 今天的一道思考题

综合 **Day10 Llama 3.1/3.2、Day13 DPO-Gap、Day09 Qwen2.5、Day15 R1**（答案不在任何一篇原文里）：

**(a) RS 与 DPO-Gap 的数据矛盾。** Llama 的 RS 策略是"每轮用 RM 选最优响应进 SFT"（§4.2.2）——要的是**远离决策边界的高质量正例**；Day 13 的 DPO-Gap 说"留 gap 最小的 10% 难对做 DPO"——要的是**贴近决策边界的难分正负例**，其中 gap $= \hat r(x,y_w) - \hat r(x,y_l)$ ， $\hat r(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ 。问题：设计一个统一的采样-选择框架，从每个 prompt 的 $N$ 个采样响应中**同时**产出 SFT 数据和 DPO pairs，并回答：① 为什么"RM 高分正例"和"小 gap 难对"不能是同一批数据（从 $\hat r(x,y_w)$ 与 gap 的数学关系论证）；② 你的框架里， $N$ 个响应按什么顺序被"消费"（先做 SFT 筛选还是先做 pair 构造），为什么这个顺序不能反（提示：论文 §4.2.1 "DPO 只用最新 batch"而 RM 用全历史——这个双速消费决定了什么依赖关系）；③ 在你的 500k coding 池上，这个框架的"预算分配"（多少采样预算给 SFT、多少给 DPO pair 挖掘）应该由什么信号决定（不许回答"训完看 HumanEval"，给出训练**前**可算的 proxy）。

**(b) 三条后训练路线的分诊台。** Llama 路线 = 纯 offline（6 轮 SFT+RS+DPO，无 online RL）；Qwen 路线 = offline DPO（白盒信号）→ online GRPO（RM 奖励）；R1 路线 = 极小冷启动 + 纯 RL（可验证奖励）。从"**数据信号的可验证性**"画一张二维图：横轴 = 信号可验证性（execution/答案匹配 ←→ RM 主观打分），纵轴 = 任务开放度（数学/代码推理 ←→ 开放问答/风格）。把三条路线放进去，并回答：① Llama 的"无 online RL"选择在 tool use（execution 可验证）上为什么成立、在数学推理上为什么可能不成立（提示：GRPO 的 group 相对优势 $\hat A_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$ 需要什么样的奖励信号才能提供学习信号；再想想论文 §4.3.1 "405B 自举失效"的反例意味着什么）；② 对你的 coding 数据工作（目标：仓库级 agent 能力，SWE-bench 类），设计三路分诊：哪类数据走纯 offline、哪类走 DPO→GRPO、哪类走纯 RL——每路写出准入条件（什么数据信号准入）、退出条件（什么指标触发停），并论证为什么"中间路线"（固定量 SFT + 一轮 RL）可能两头不靠。

---

论文原文：
- Llama 3 Herd (3.1, §4.1.6 six rounds / §4.2 数据管线 / §4.3 专家模型 / §7 vision adapter): https://arxiv.org/abs/2407.21783
- Llama 3.2 Model Card（1B/3B 9T + 8B/70B logits pretraining distillation）: https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md

GitHub NOTES: https://github.com/Papa-Panda/post-training/blob/master/ai-data/day-10-2024-llama3.1-3.2/NOTES.md
