# Paper 模板 - Day 10 自动骨架

## 元信息
- Title: Llama 3.1 / 3.2 - Post-training Expansion, Multilingual / Long-Context / Tool Use, Distillation & Pruning for 1B/3B and Vision 11B/90B
- Authors / Org: Meta AI - Llama Team
- Link / arXiv: https://arxiv.org/abs/2407.21783 (Llama 3 Herd v3 = 3.1 base, Jul 23 2024 + Nov 23 2024 update) + Model Cards: https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md (Sep 25 2024) + Llama 3.2-Vision: https://arxiv.org/abs/2409.17379 (vision adapter draft)
- Date read: 2026-08-13
- Tags: [pretraining, sft, rl-data, curation, multilingual, long-context, tool-use, synthetic-data, distillation, pruning, coding-data, vision]
- Folder: 2024_llama3.1-3.2
- Day: 10

## 一句话总结
Llama 3 已用 15.6T 训到 405B，3.1/3.2 不再加 pretrain tokens量级，而是在 post-training 上做增量：8语言多语+128K长上下文+tool use/func calling+多轮5.5M+人工+合成SFT+多轮RS+DPO，以及1B/3B用3.1 8B/70B logits 做pretrain distillation + pruning+量化，11B/90B Vision 用cross-attention adapter接图像——把“预训练堆数据”转为“后训练数据飞轮+小模型蒸馏/剪枝+多模态扩展”的工程化范例。

## 和之前工作的关系
- **知识图谱位置**：pretrain / scaling 主线的“增量迭代点”，不是新 pretrain，是 Day 7 Llama 3 (15.6T dense 五级过滤+code 25%+annealing) 的直接后继。对比 Day 8 DeepSeek-V3 (14.8T MoE code 30%+激进去重+FIM) 和 Day 9 Qwen2.5 (18T file+repo级 code 5.5T + 1M SFT +多阶段RL)，Llama 3.1/3.2 补上了那三家都弱的一环：**后训练数据的规模化、产品化配方**。
- **接了哪条线**：
  - pretrain/curation线：接 Day 7，复用15.6T，只把cutoff拉到Dec 2023，9T用于1B/3B轻量重训；不像Qwen把7T→18T那样暴力扩tokens，而是把tokens花在“精”。
  - synthetic线：接 Day 6 Phi-1 (教科书合成) + Day 8 DeepSeek执行过滤；Llama 3.1/3.2 的合成不是 pretrain 合成，而是 SFT 合成：tool use轨迹、长上下文问答、多语回译、安全性对抗，都用更大模型生成 → 小模型过滤 → RM打分 → Rejection Sampling，规模化到数百万。
  - selection/influence线：接 Day 4 LESS (梯度相似度选5%) 和 Day 5 DataInf (LoRA闭式1秒/条)：Llama 3.1/3.2不用梯度，用RM/奖励+人工偏好做选择——RS/DPO的pair就是“对偏好影响最大”的子集，计算更便宜、可产品化。
- **补了哪个短板**：Llama 3只讲预训练怎么搭，SFT/RL一笔带过；Qwen讲了1M SFT+多阶段RL但没讲1B/3B怎么做得稳、Vision怎么加、量化怎么做。Llama 3.1/3.2补上“同配方如何撑8B/70B/405B全系列+1B/3B端侧+11B/90B多模态，且后训练6轮迭代不崩”的坑。
- **替代/分叉/改进**：不是替代Llama 3，是改进+分叉：405B保持配方，8B/70B做targeted能力追加（tool use、多语、数学、代码），1B/3B走distill+prune+恢复训练的新分支，11B/90B走vision adapter分支。三大开源配方三角（Llama/DeepSeek/Qwen）在Qwen处已闭合，3.1/3.2是闭合后向产品化/轻量化/多模态的延伸。
- **对Day X直接对比**：vs Day7 Llama3：同15.6T基座，但后训练从单轮SFT→6轮SFT+RS+DPO，数据从通用指令→专项（coding/reasoning/tool/long/multilingual/safety）5.5M+精筛；vs Day9 Qwen2.5 72B>405B的参效比故事：Llama 3.1/3.2用405B证明“参量还能靠后训练数据再榨”，用1B/3B证明“小模型靠logits蒸馏+prune恢复也能保留80%能力”。

## 为什么今天读它
- coding data：Llama 3.1 新增code专家后训练集（自带执行环境验证+unit test通过率作filter），tool use / func calling轨迹本质是code+JSON，1B/3B的distill logits在code completion上比纯Causal LM稳，可直接抄到你50万合成池的“可执行过滤+tool轨迹合成”。
- SFT：展示如何从 Day7 的通用SFT配比 → 后训练6轮迭代，每轮RS挑高RM分样本再DPO，配比按能力维（coding / math / reasoning / long / multilingual / safety）动态调，和你Qwen Day9的1M SFT多阶段RL呼应但更工程化。
- RL data：多轮DPO的preference pair怎么来（人工→RM→合成→再RM），为什么不用PPO而用DPO+RS（稳定、可扩展），和你Agentic RL Infra的“$/useful-rollout”评估直接相关；1B/3B的量化/剪枝后恢复训练对RL数据噪声更敏感，提供“小模型RL数据要更干净”的反例。

## 今天的 3 问
1. Llama 3.1/3.2 的后训练为什么从PPO转成“多轮SFT+Rejection Sampling+DPO”循环？6轮迭代里每轮的SFT/RS/DPO数据是怎么分工的（比如tool use、长上下文、多语、安全性各在哪轮加）？和 Day 9 Qwen2.5 的 RM→SFT→新RM→RL self-improvement相比，稳定性/成本trade-off在哪？
2. 1B/3B的“在pretrain阶段融入8B/70B logits做distillation + pruning + 恢复训练”是怎么做的？logits做token-level target和普通CE loss怎么加权？prune用的是width pruning MAW还是depth？恢复训练用了多少tokens？和你如果把50万池用70B教师打logits蒸到1.3B相比，预期HumanEval涨多少、MMLU掉多少，infra成本是升是降？
3. 【对比题】对比Day7 Llama3 (15.6T pretrain、5级瀑布、code 25%、annealing 5%高质)、Day8 DeepSeek-V3 (14.8T MoE激进去重MinHash0.90、code30%+、FIM10% PSM执行过滤)、Day9 Qwen2.5 (18T file+repo级code 5.5T、弱模型scorer、1M SFT+多阶段RL)、Day10 Llama3.1/3.2 (复用15.6T、后训练6轮专项、1B/3B distill+prune、11B/90B vision adapter)——四家在“code占比/去重阈值/合成策略/后训练轮次/小模型化”五轴上各有什么取舍？如果你要为你50万合成池定下“25%→30% code + MinHash0.90 + FIM+exec过滤 + 1M级SFT多阶段 vs 6轮DPO小步快跑”二选一，基于这四篇论文的证据你选哪个，为什么？Infra视角：Bloom+分布式MinHash vs 弱模型scorer vs distillation logits，哪一个是18T/15T规模下的真正瓶颈？

## 核心（待填，今晚产出）
1. **Motivation**: 为什么不在15T上再堆tokens，而要做后训练增量/小模型化/多模态？405B边际收益已低，产品化需求（多语、长上下文、tool use、端侧、vision）倒逼。
2. **Data Pipeline**: 来源 → 复用Llama3 15.6T (Dec 2023 cutoff) → 1B/3B额外9T logits distillation pretrain → 清洗 → 后训练6轮：SFT (5.5M+ 人工+合成，coding/math/reasoning/long/multilingual/safety/tool) → RS (RM打分选top) → DPO (preference pair人工+合成) → iterative loop；Vision 11B/90B额外image-text 6B (?) cross-attention adapter；Safety/PII再筛。
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
  - Distillation logits存储成本高（vocab 128K * seq_len），用top-k logits (k=64) + temperature 0.7可省90%存储，Llama 3.2 1B/3B实践。
  - Pruning恢复训练的data mix要更干净，小模型对脏数据更敏感，可用DataInf思路扫高self-influence样本先踢。

## 疑问 / 下一步
- 6轮DPO里RM是怎么进化（初始人工RM→合成RM→最终RM），和Qwen2.5-Math的self-improvement闭环有什么实现差异？1B/3B的pruning具体宽度剪多少（expansion ratio从4→？）后MMLU掉点曲线怎样？

## 原文金句 (1-2句)
> We use a similar recipe as Llama 3.1 and produced final chat models by doing several rounds of alignment on top of the pre-trained model. Each round involved Supervised Fine-Tuning (SFT), Rejection Sampling (RS), and Direct Preference Optimization (DPO). (from Llama 3.2 Model Card)
> Llama 3.2 was pretrained on up to 9T tokens ... we incorporated logits from Llama 3.1 8B and 70B into the pretraining stage ... Knowledge distillation was used after pruning to recover performance.

## 3 问回顾（Day 10 原题见上）

## 参考
- Llama 3 Herd (3.1): https://arxiv.org/abs/2407.21783 (v1 Jul 31 2024, v3 Nov 23 2024)
- Llama 3.2 Model Card (1B/3B/11B/90B): https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md (Sep 25 2024)
- Llama 3.2 1B/3B Text: https://huggingface.co/meta-llama/Llama-3.2-1B / 3B
- Llama 3.2-Vision 11B/90B: Paper draft https://arxiv.org/abs/2409.17379 + Model Card vision section
- Evolution survey Llama 3.1→3.2: 9T tokens, multilingual 8 langs, 128K context, tool use (direct quote lines 89-94)

---
生成逻辑：已纳入知识图谱，强调结构非数量，自动产出已开启。
