# 05 Metrics — 从 Perplexity 到 Judge 0-5

## 为什么 perplexity 失效

> 【arxiv:2409.11233】Prune 50% 权重后，ppl 只涨 0.3，但 HellaSwag -12

Transfer to agent:

- `ppl(compressed_summary)` 低：summarizer 是个好 LM
- `probe_recall` 低：summary 丢信息

## 传统三级指标

### 1. LM-level

- **Perplexity**: `exp(avg NLL)` — 只能测 fluency
- **JS Divergence** $D_{JS}(P_{orig} || P_{comp})$: 测压缩后 next-token 分布偏移，来自 [2402.00861] compression as eval

  公式 $JS = 0.5 KL(P||M)+0.5 KL(Q||M)$，$M=(P+Q)/2$。JS 小 = 压缩保真度好。但仍不 capture artifact loss。

### 2. Retrieval-level

- **Exact Match recall**: artifact file list exact match？
- **ROUGE / BERTScore** vs gold summary

问题：gold summary 谁来写？agent session gold 很难定义。

### 3. Downstream task

- 压缩后直接跑最终任务成功率：SWE-bench pass？
- 太重、慢、不定位 failure mode

## Factory Judge 方法 (adopted by Hermes)

They use LLM-as-judge 0-5 discrete:

```
0 = completely missing / hallucinated
1 = mentions but wrong
2 = partial but missing critical detail
3 = usable but incomplete (the sweet spot threshold)
4 = correct with minor omission
5 = perfect / gold equivalent
```

每个 probe → judge prompt template：

```
You are evaluator for dimension {dim}.
Context (compressed): {compressed}
Question: {q}
Gold answer: {gold}
Model answer: {ans}
Score 0-5 per rubric + reason in < 20 words.
```

6 dims avg = `quality`. They deliberately keep ratio separate.

最终报告两栏：

```
compression_ratio: 80K→22K = 3.6x
latency_save: -1.2s prefill
quality: 3.7/5 (artifact_trail 2.9 lowest)
```

CI gate: `quality >= 3.5` 且 `artifact_trail >= 3.0` 才能 merge。

## 我们最小实践

- `toy_compressor_eval.py` 先算 EM recall，省 LLM judge cost
- 再接 `openai` judge stub ( 见 code )，跑 5 probes 验证 bias
- Long run 与 `vllm-rollout` 打通：当 compressor 换成 `keep-last-n tool outputs verbatim`，probe recall 如何从 2.9 → 4.1

## 参考 Threshold (来自 Hermes sample + Factory)

| 场景 | ratio | quality target | artifact_trail floor |
|------|-------|---------------|----------------------|
| SWE short 20K | 2x | 4.2 | 4.0 |
| SWE long 120K (我们场景) | 5-6x | 3.7 | 3.2 |
| 1M token research (GLM-5.2) | 10x | 3.5 | 3.0 |

> Lower ratio with high floor > aggressive ratio low quality — Factory 核心观点重复。

## Code Stub 思路

See `code/toy_compressor_eval.py`:

```python
def evaluate(compressor, fixture, probes):
    comp = compressor.compress(fixture) # 120K→20K
    scores = []
    for p in probes:
        ans = simple_qa(comp, p.q)
        s = judge(p, ans) if use_llm else em_score(p.gold, ans)
        scores.append(s)
    return {"ratio": len(fixture)/len(comp), "by_dim": aggregate(scores)}
```

JS divergence version: sample 100 next-token logits from orig vs comp for same prefix → JS.

But keep simple first.
