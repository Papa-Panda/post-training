# Paper 模板

## 元信息
- Title: The Llama 3 Herd of Models
- Authors / Org: Meta AI (Llama Team, Aaron Grattafiori et al. ~500 authors)
- Link / arXiv: https://arxiv.org/abs/2407.21783
- Date read: 2026-08-10
- Tags: [pretraining, coding-data, curation, quality, scaling, flywheel, synthetic-data]

## 一句话总结
Meta 用 15.6T tokens 从零搭的工业级预训练管线，堆了 5级质量过滤+多轮去重+code上采样25%+多轮合成/回译，把 405B 训到对标 GPT-4，证明了长尾质量飞轮比单纯堆数据量更管用。

## 核心
1. **Motivation**: 之前 Llama 2 2T 就见顶，再堆 token 收益衰减。问题不在量，在 web 臭、重复多、code配比低。需要一套可控 15T 的配方，让 405B 既懂 web 又会 code 还能长推理。
2. **Data Pipeline**: 来源 → Web (50%+)、Code (25%+)、Multilingual、多模态；清洗 → heuristic (行长/符号密度/黑名单) → exact dedup (Bloom) + fuzzy MinHash + URL去重 → safety/PII；质量 → fastText 粗筛 → Roberta/BERT质量分类器 → Llama 2 70B 当质量打分器(educational value) → 最终人工抽检；合成 → Code回译、Math推理链、拒绝采样蒸馏；配比 → code从17%→25%，初期重通用，后期 annealing 加高质合成。
3. **Key Tricks**:
   - Code upsample 实测：17%→25% HumanEval +8-12 pts，MBPP +5，不伤 MMLU，超过30%反而长文本掉。
   - 级联过滤是关键：heuristic砍30%垃圾，MinHash再砍15%近重复，fastText砍20%，Llama2打分器再砍10%低质，最后15.6T是15T里挑的不是堆出来的。Phi-1是单级textbook classifier，Llama3是5级瀑布，成本/召回更好控。
   - Annealing + 高质合成：最后 5% tokens 用超高质 code + math chain-of-thought 做 annealing，405B 推理直接 +10%。
4. **Results**: 15.6T 训完 405B，HumanEval 89.0% (vs 86% Llama2-70B-code基线), MBPP 81%，MMLU 87.3%，长上下文 128k NIAH接近满分。用同样配方训 8B/70B 都超 Llama2同尺寸。

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：
  1. 把你 50万 合成池按 Llama3 五级走一遍：先 exact dedup、再 MinHash 0.85、fastText教科书分>0.7、再用你 1.3B 当打分器筛一遍，应能砍掉30%水数据且 HumanEval 不掉。
  2. Code配比直接抄 25% ceiling，前 80% steps 用 25%，后 20% annealing 提到 35%纯高质 code+exec验证过的数据，做 RL 前的 SFT 冲刺。
- Infra 视角：15T级别必须流式+分片，Bloom dedup + 分布式MinHash是瓶颈，已有 Llama3 infra开源脚本可复用，评测自动化上用持续 HumanEval 每 5k steps 跑一次做配比信号。

## 疑问 / 下一步
- Llama3 herd 里 code从17%→25%的收益是否在 70B 也线性，还是405B才明显？
- 它的 Llama2打分器如果换成你现在 RL reward model 做二次筛，会不会比通用质量分更准？

## 原文金句 (1-2句)
> We pretrain on 15.6T tokens with careful curation to balance knowledge, code, and reasoning — quality trumps raw size.

> Code upsampling to 25% significantly improves reasoning while preserving general capabilities if done before annealing.

## 参考
- GitHub 官方配方讨论: https://github.com/meta-llama/llama3 (含数据配方摘要)
- 同步创建: papers/2024_llama3-herd/README.md 见 herd 版
