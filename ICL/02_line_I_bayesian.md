# 02 线 I：隐式 Bayesian 推断

[← 01 定义](01_definition_timeline.md) · [下一章：GD 视角 →](03_line_II_gd.md)

## 1. 一个透明的有限概念模型

设潜在任务 $c\in\mathcal C$，先验为 $p(c)$。为便于推导，先采用条件独立近似：给定 $c$，演示 $z_i=(x_i,y_i)$ 独立来自 $P_c$。于是：

$$p(c\mid D_k)=\frac{p(c)\prod_{i=1}^{k}P_c(z_i)}{\sum_{c'\in\mathcal C}p(c')\prod_{i=1}^{k}P_{c'}(z_i)}$$

查询的 posterior predictive 是混合，而不是默认取 MAP：

$$p(y_q\mid x_q,D_k)=\sum_{c\in\mathcal C}p(y_q\mid x_q,c)p(c\mid D_k)$$

对真实概念 $c_*$ 与竞争概念 $c$，后验赔率满足精确恒等式：

$$\log\frac{p(c_*\mid D_k)}{p(c\mid D_k)}=\log\frac{p(c_*)}{p(c)}+\sum_{i=1}^{k}\log\frac{P_{c_*}(z_i)}{P_c(z_i)}$$

若 $z_i$ 确实独立来自 $P_{c_*}$，单个证据项的期望为：

$$\mathbb E_{z\sim P_{c_*}}\left[\log\frac{P_{c_*}(z)}{P_c(z)}\right]=D_{\mathrm{KL}}(P_{c_*}\Vert P_c)$$

这给出正确直觉：低先验或难区分概念需要更多证据。但**仅凭 KL 下界不能直接写出一个普适的高概率 sample-complexity 等式**；还需控制似然比尾部、依赖性、模型错设与 prompt/pretraining 分布偏移。

## 2. 与 Xie et al. 的边界

Xie et al. 的正式设置不是上面的 IID 分类器，而是具有文档级潜概念的 mixture of HMMs。[T-bound] 论文证明在一组可恢复性、支持集与分布偏移条件下，next-token predictor 可在 prompt 中推断共享潜概念；实验 GINC 也出现顺序敏感和 zero-shot 优于 few-shot 的情形。

因此可以说：

- 长程连贯的预训练目标可能训练出“先判断文档/任务，再预测 token”的机制；
- prompt 与预训练文本格式不完全相同不必然阻止 ICL；
- 不能由此推出任意真实语言模型的 posterior、收敛速率或 shot 阈值。

## 3. 标签噪声：模型取决于假设

若任务 $c$ 规定确定映射 $f_c$，可显式加入噪声通道：

$$p(y_i\mid x_i,c)=(1-\epsilon){\bf 1}[y_i=f_c(x_i)]+\epsilon\rho(y_i\mid x_i)$$

- 当模型相信 $\epsilon$ 很小，冲突标签强烈反驳当前概念。
- 当模型相信 $\epsilon$ 大，错误标签影响被削弱。
- 当标签词本身带强语义先验，有限概念模型也会错过真实行为。

Min et al. 的分类/多选结果表明，随机标签在其设置中“barely hurts”；这支持 label space、输入分布和格式也贡献很大，但不能外推为生成式代码任务对错误输出不敏感。[01](01_definition_timeline.md) 的四组对照应逐任务重做。

## 4. coding-data 映射

对代码任务，不应把 $c$ 粗暴写成一个标签。更实用的是因子化：

$$c=(\text{language},\text{repository conventions},\text{API contract},\text{bug class},\text{output protocol})$$

这带来三条可检验假设：

1. **覆盖**：示例应覆盖查询需要的因子，而非只最大化表面相似度。
2. **区分**：能排除竞争假设的示例比重复同类示例更有价值。
3. **校准**：冲突示例应降低置信度；若模型仍极度自信，说明它可能只跟随强先验或格式。

不要预设“Python 纠错固定需要 5-shot”或“模板调用 2-shot 足够”；shot 数是模型、任务、tokenizer、顺序和上下文预算共同决定的实验量。

## 5. 可运行验证

```bash
python3 ICL/icl_mechanisms.py
```

[`posterior`](icl_mechanisms.py) 在 log space 中累积有限概念证据；测试验证 posterior odds 的乘法更新与长序列数值稳定性。它验证本节代数，不是语言模型行为证据。

来源：[Xie et al.](https://arxiv.org/abs/2111.02080) · [Min et al.](https://arxiv.org/abs/2202.12837) · [证据账本](references.md)
