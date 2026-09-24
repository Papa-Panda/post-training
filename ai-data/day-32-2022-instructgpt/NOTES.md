# Day32 InstructGPT — NOTES

## 元信息
- Title: Training language models to follow instructions with human feedback
- Authors / Org: Ouyang et al. / OpenAI
- Link / arXiv: https://arxiv.org/abs/2203.02155
- Date read: 2026-09-24
- Tags: [rl-data, preference-data, rlhf, human-feedback, curation, quality]

## 一句话总结
RLHF 数据管线的正典：40 名筛选过的 contractor 产出三层数据——12,725 条 SFT demonstration、33,207 条 prompt 的 K 选排序（RM 用）、31,144 条无标注 API prompt（PPO 环境用）；175B InstructGPT 被偏好率 85±3% 碾压同尺寸 GPT-3，1.3B 版本甚至打赢 175B GPT-3。

## 核心
1. **Motivation**: 预训练目标（next-token on web）与"按用户意图办事"错位；public NLP 数据集（FLAN/T0）捕捉不到真实 API prompt 分布（生成/头脑风暴占 ~57%，分类/QA 只占 ~18%），需要直接从人类偏好造数据。
2. **Data Pipeline**: prompt 两个来源——API Playground 提交（去 PII）+ labeler 手写（Plain / Few-shot / User-based 三类）。三层数据分工：
   - SFT：labeler 写 demonstration，train 12,725（labeler 11,295 + customer 1,430；labeler 占主导是因为同一条 instruction 采样不同 few-shot 组合可合成多个数据点）；
   - RM：每条 prompt 给 labeler 看 $K = 4$ ~ $K = 9$ 个模型输出做全排序，train 33,207 条 prompt（labeler 6,623 + customer 26,584）；
   - PPO：31,144 条纯 API prompt，零人工标注，只当 RL 环境输入。
3. **Key Tricks**:
   - RM 训练把每条 prompt 的 $\binom{K}{2}$ 个组合对当**单个 batch 元素**：每个 completion 只需一次 forward pass，且不再过拟合（朴素打散 shuffle 一遍就过拟合，因为每个 completion 被用 $K-1$ 次梯度更新）；
   - RM 只用 6B（175B RM 训练不稳定，不适合当 value function），并归一化使 demonstration 均值 reward 为 0；
   - RM loss（Eq 1）：
   
   $$\operatorname{loss}(\theta) = -\frac{1}{\binom{K}{2}} \mathbb{E}_{(x,y_w,y_l)\sim D}\left[\log\sigma\left(r_\theta(x,y_w) - r_\theta(x,y_l)\right)\right]$$
   
   - PPO 加 per-token KL 惩罚锚定 SFT；PPO-ptx 再混入预训练梯度（Eq 2，$\gamma \geq 20$ 在 1.3B 上修复 DROP/SQuAD 回退），用 8 倍于 RL episode 数的预训练样本；
   
   $$\operatorname{objective}(\phi) = \mathbb{E}_{(x,y)\sim D_{\pi_\phi^{RL}}}\left[r_\theta(x,y) - \beta\log\left(\pi_\phi^{RL}(y|x)/\pi^{SFT}(y|x)\right)\right] + \gamma\,\mathbb{E}_{x\sim D_{pretrain}}\left[\log \pi_\phi^{RL}(x)\right]$$
   
   - labeler 协议本身就是数据资产：40 名 contractor 经 screening（敏感言论 flag 一致性、排序一致性 75% cutoff、demonstration 6/7 分）+ onboarding + 详细指令 + 共享答疑群；标注者间一致性 72.6±1.5%；
   - 标注指令写死 trade-off 规则：通常 harmless + truthful > helpful，除非 (a) helpful 明显更强、(b) truthful/harmless 只略差、(c) 非高风险域；终极标尺是"customer assistant 你更想收到哪个输出"。
4. **Results**: 175B InstructGPT vs 175B GPT-3 偏好率 85±3%，vs few-shot GPT-3 71±4%；TruthfulQA 真实有用回答 ~2 倍；闭域幻觉率 21% vs 41%；被要求礼貌时毒性输出 -25%；bias（Winogender/CrowS-Pairs）无改善；PPO 有 alignment tax（SQuAD/DROP/HellaSwag/翻译回退），PPO-ptx 大幅缓解——单纯加大 KL 系数修不好；FLAN/T0 微调版在 API 分布上不如 SFT 基线（InstructGPT 对其 head-to-head 78±4% / 79±4%）。

## 可迁移
- 对 coding data 工作：三层数据分工可直接套用——人工写"理想代码回答"（SFT demonstration）/ 对同一需求的多个模型输出做排序（RM 数据）/ 无标注的真实用户 coding prompt 池（RL 环境）。labeler 指令的 helpful/truthful/harmless 三角可改写成代码场景的 correct/safe/clear，外加 trade-off 例外条款。
- Infra 视角：RM 只需 6B（偏好建模的容量需求远小于生成），K 选排序的单 batch 元素技巧省 forward pass；PPO-ptx 的 8 倍预训练混合说明 RL 数据不能脱离 pretrain 分布——rollout 数据管线里要常驻一条 pretrain 回流。

## 疑问 / 下一步
- 论文 5.2 自认对齐的是"40 个 labeler + 研究员这个特定人群"的偏好，而 held-out labeler 来自同一 vendor 且未经 screening——泛化实验的外部效度有多强？
- Table 8：PPO valid 集每 customer 平均 31.55 条 prompt，valid 客户高度重复——validation reward 的代表性是否被高估？
- 训练时让 labeler 优先 helpfulness，终评时却优先 truthfulness/harmlessness——标注目标与评测目标不一致，这是刻意设计还是妥协？对 RM 训练的信号意味着什么？

## 原文金句 (1-2句)
> This procedure aligns the behavior of GPT-3 to the stated preferences of a specific group of people (mostly our labelers and researchers), rather than any broader notion of "human values".
