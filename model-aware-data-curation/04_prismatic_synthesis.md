# 04 — Prismatic Synthesis：从选数据到主动造数据

## 1. 为什么是总纲论文

静态选择只能从已有池中重排预算；当池子在某些学习方向上根本没有样本，selector 无法补洞。Prismatic Synthesis（2025）将数据覆盖变成生成控制信号：

\[
D_t \xrightarrow{\text{gradient map}} \text{sparse regions}
\xrightarrow{\text{generate}} C_t
\xrightarrow{\text{accept sparse}} D_{t+1}.
\]

这把三条已有路线接起来：目标价值（LESS-like）+ 集合多样性（Vendi-like）+ synthetic generation。

## 2. 论文三步算法

从 seed pool $D_0$ 开始，每轮：

1. **Cluster**：用 off-the-shelf proxy 计算 loss gradients，随机投影后做 $k$-means；
2. **Generate**：从当前池随机抽 few-shot examples，提示 generator 产生新样本；
3. **Diversify**：只接收落入稀疏梯度簇的候选，再加入池中迭代。

伪代码：

```python
D = seed_data
for round_id in range(T):
    G = projected_loss_gradients(proxy, D)
    clusters = kmeans(G, k=schedule_k(len(D)))
    sparse = least_populated(clusters)

    C = generator.generate(few_shots=sample(D), n=batch_size)
    C = quality_and_decontamination_gate(C)
    Gc = projected_loss_gradients(proxy, C)
    accepted = [z for z in C if assign(Gc[z], clusters) in sparse]
    D.extend(accepted)
```

论文说明示例为保留成员数最少的 top 20% clusters；其 NLI/math 实现动态设 $k=1\%\times |D|$，保留最小的 $k/2$ 个簇。它是 rejection sampling，不是要求 generator 直接反传梯度。

## 3. 已核实的实验数字

来自论文 v2 / NeurIPS 2025：

- 合成 data pool 超过 **3 million samples**，微调 **more than 300 models / training runs**，控制数据规模与质量；
- G-Vendi 与 unseen OOD 平均表现的 **Spearman $\rho\approx0.9$**，同时报告在 NLI 和数学推理；
- 梯度随机投影维度实验设为 **1024**，示例 proxy 包括 **0.5B instruction-tuned model**；
- 最终 **Nemotron-PrismMath 1.0M** problem-solution pairs，**PrismNLI 515K** input-label pairs；
- PrismMath-7B（32B generator）在论文列出的 7 个挑战 benchmark 中胜过比较对象 R1-Distill-Qwen-7B（其数据由 671B generator 生成）中的 **6/7**；
- PrismNLI 在 8 个 OOD benchmark 平均比论文最佳既有 mixture 高 **8 percentage points**；
- 朴素 few-shot/persona 生成在约 **50K–100K** 规模出现平台，而 Prismatic 在更大规模继续改善。

这些是论文特定设置的结果，不改写成通用 production SLA。

## 4. 质量门不能省

梯度稀疏可能来自：真正新技能、格式异常、错误答案、乱码、恶意样本。论文自身也在生成后做：

- 多答案 majority-vote filtering；
- benchmark 10-gram matching；
- LLM paraphrase decontamination。

coding data 还应加入：parse / compile、unit tests、sandbox execution、timeout、静态安全检查、license/provenance。

## 5. 与 target signal 合并

原始 Prismatic 更接近无显式目标分布下的 coverage maximization。对 coding flywheel，可以把接收条件改成：

\[
\mathrm{accept}(z)=
\mathbf 1[q(z)=1]
\mathbf 1[c(z)\ge\tau_c]
\mathbf 1[v(z)\ge\tau_v]
\mathbf 1[r(z)\le\tau_r].
\]

- $q$：正确性/执行/污染；
- $c$：稀疏簇或 G-Vendi marginal gain；
- $v$：与目标失败簇对齐；
- $r$：与保护能力冲突。

这一步是本专题的工程扩展，不冒充 Prismatic 原论文结论。

## 6. 最小可行实验

1. 收集 5k 已验证 coding traces；
2. 0.5B–1.5B proxy 上只取 LoRA/LM-head 梯度并投影到 256/1024 维；
3. 聚成 50–200 个方向簇，检查簇是否对应算法/错误类型而非模板风格；
4. 从最稀疏 20% 簇生成候选；
5. execution gate 后训练同一 base model；
6. 对照 random、embedding-diverse、target-only、G-Vendi-only、target+G-Vendi；
7. 报告 ID/OOD pass@1、失败簇覆盖、污染、GPU-hours 与重复率。
