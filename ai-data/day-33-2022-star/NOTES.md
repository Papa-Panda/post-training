# Day33 STaR — NOTES

## 元信息
- Title: "STaR: Bootstrapping Reasoning With Reasoning"
- Authors / Org: Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah D. Goodman / Stanford（Goodman 另属 Google Research）
- Link / arXiv: https://arxiv.org/abs/2203.14465
- Date read: 2026-09-25
- Tags: [rl-data, reasoning-data, synthetic-data, bootstrapping, rationalization, quality]

## 一句话总结
用 rationalization 自举推理数据：以 $P = 10$ 条带 rationale 的 few-shot 例子为种子，让 GPT-J 自己对全数据集生成 rationale——答对的保留、答错的把正确答案塞进 prompt 反向生成 rationale 再过滤，多轮外循环把"自产"的推理数据越滚越大；CQA 上 6B 的 GPT-J 做到 72.5%，逼平 30 倍参数的 GPT-3（73.0%），是 SFT 数据与 RL 数据之间"模型自己造数据"的开山之作。

## 核心
1. **Motivation**: CoT rationale 能涨点，但已有的两条路都有硬伤——人逐条写 rationale 太贵、每个新任务都要重来；纯 few-shot in-context 精度远不如在 (x, y) 数据上直接做答案 SFT。而现成数据集只有问题和答案、没有中间推理过程。STaR 的思路是把"模型已有的推理能力"本身变成数据生产引擎，用少量种子把无 rationale 的大数据集"翻译"成 rationale 数据集。
2. **Data Pipeline**: 外循环（expert iteration 式），每轮三步：
   - **生成**：当前模型在 few-shot prompt 下对数据集每条 $x$ 生成 $(\hat{r}, \hat{y})$ ，只保留 $\hat{y} = y$ 的（核心假设：能得出正确答案的 rationale 质量更高）；
   - **rationalization**：答错的题把正确答案作为 hint 塞进 prompt，反向生成 rationale，再检查它是否能推出正确答案，能则保留——采样分布从 $p(r\mid x)$ 换成 $p(r\mid x, y)$ ，已知答案的反推搜索空间小得多；
   - **重训**：每轮都从**原始预训练模型 $M$ 从头训练**（不接续上一轮，防过拟合），训练集 = 本轮保留的自生成 rationale + rationalization 样本；重复直到 plateau。
   - 数据规模：算术任务从 50,000 条均匀采样（按位数），每轮抽 10,000 条，每位数用 10 条随机 few-shot rationale 例；CQA 用固定 10 例 prompt；GSM8K 用 7,473 train / 1,319 test。
3. **Key Tricks**:
   - **低温 greedy 解码**，不用高温多采样扩数据：高温会让"答案对、推理错"的污染样本混进训练集，学坏且泛化差——在算术任务上高温采样的 scratchpad 会发散成无意义文本、训练停滞；rationalization 是更便宜高效的扩数据手段（多采 10 个样本 ≈ 10 倍生成时间）；
   - **训练时保留 few-shot prompt**：CQA 上 60.9%→68.8%（无 rationalization）、69.9%→72.5%（有），且能显著抑制多轮之后 rationale 风格漂移（drift）；
   - rationalization 样本的梯度信号更强：未 rationalization 的保留样本都是模型低温度下最自信的输出，梯度反而弱；
   - 过滤信号只有"答案对错"：CQA 是 5 选 1，随机就有 20%、简单启发式 ~30%，蒙对的 rationale 也会混进训练集——这是该方法的数据质量上限来源。
4. **Results**:
   - CQA（dev）：STaR + rationalization **72.5%**（用了 86.7% 训练数据：78.2% 自生成 + 8.5% rationalization），vs 直接答案微调 60.0%（100% 数据）、few-shot CoT 36.6%、175B GPT-3 直接微调 73.0%——6B 模型追平 30 倍大模型；
   - 算术（n 位加法）：16 轮后 **89.5%**，vs 无 rationalization 的 10,000 例 baseline 76.3%；第 1 轮后 2 位数加法从 <1% 跳到 32%；带 rationalization 能同时学多位数，不带则是逐级 stage-wise（先学好 n-1 位才会 n 位）；
   - GSM8K：**10.1%→10.7%**（只用了 25.0%/28.7% 数据，其中 0.5% 来自 rationalization），vs 直接答案微调 5.8%、few-shot CoT 3.1%；收敛用了 36 轮 + 额外 10 轮（30 轮后还 cap 了步数防训练过长）；
   - 人评：20 名众包工作者 30% 更偏好 STaR 生成的 rationale（vs few-shot， $p = .039$ ），74% 更偏好 vs 人写 rationale（ $p < .001$ ）——作者自认不代表人类水平，只说明高质量 rationale 难 elicited。

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：coding 场景天然有"答案"（单测通过/CI 绿）——对 pass 的题保留模型自生成的 CoT，对 fail 的题把正确解法或单测报错塞进 prompt 做 rationalization，就是 STaR 的 coding 版；采样保持低温 + 答案过滤做数据阀，别用高温多样性采样凑量（错误推理污染训练集是论文实证过的坑）。
- Infra 视角：外循环 = 生成→过滤→重训 的数据飞轮，每轮从 base 重新训练防过拟合，代价是训练 compute 随轮数线性涨；GSM8K 36+10 轮才收敛说明飞轮迭代次数是主要成本——先小模型/小数据验证"首轮 few-shot 高于 chance"这个启动条件（论文里 GPT-2 量级连算术都启动不了），再放大。

## 疑问 / 下一步
- 论文坦言过滤器只看答案对错：coding 里"单测通过但推理过程胡扯"的样本怎么筛？step-level 信号（Day35 PRM 的方向）是不是必要的下一块拼图？
- CQA 5 选 1 有 20% 蒙对率，蒙对 rationale 混入训练集的真实比例有多大？对多轮后的风格漂移贡献多少？
- rationalization 的 hint 工程（怎么把正确答案塞进 prompt）在开放域任务上是否通用——还是只在选择题/有标准答案的任务上成立？和 Day32 InstructGPT 的 labeler 写 demonstration 相比，hint 法的数据成本边界在哪？

## 原文金句 (1-2句)
> Thus, STaR lets a model improve itself by learning from its own generated reasoning.

> We propose what is, to our knowledge, the first technique to allow a pre-trained large language model to iteratively use its language modeling capacity to improve itself.

## 思考题
1. coding 里"单测通过但推理过程胡扯"的样本怎么筛？STaR 的答案级过滤够吗，还是需要 Day35 的 step-level（PRM）信号？
2. rationalization 的 hint 法（把正确答案塞进 prompt）在开放域/无标准答案任务上还成立吗——和 Day32 labeler 手写 demonstration 的成本边界在哪？

相关讨论（Gemini 网页版，2026-09-25）：https://gemini.google.com/app/94ddf894506e0a29
