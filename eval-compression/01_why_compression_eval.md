# 01 Why Compression Eval — 为什么要单独评压缩

## 问题 Problem

Production agent session:

```
[10K] system / user task
[80K] tool calls: read → search → edit → bash → browser
[30K] artifacts: file diffs, test logs
[5K]  compressed summary? <- 这里是分水岭
```

当 `len(conversation) > CONTEXT_THRESHOLD`，系统会触发 `compress()`:

```python
if len(context) > 120_000:
    context = compressor.compress(context) # 120K → 20K
```

`compress()` 之后：

- 测试里：prompt 只有 8K，compress 从未触发 → test green
- 线上：真实 SWE 任务 200K+，compress 触发 3-4 次 → summary 丢了关键 file path / API key / edge case

**Manual prompt tuning 的盲区**: 大家改 prompt 都在 short 上改，没人测 compressed state。

This blind spot is why Factory made compression eval a first-class track.

## Perplexity 不够的例子

From [arxiv:2409.11233] Compression of LLMs as eval study:

> Magnitude pruning, SparseGPT, Wanda can keep perplexity almost flat at 50% sparsity, but downstream tasks (HellaSwag, ARC, MMLU) drop 8-15 points.

Transfer to agent context compression:

- `ppl(compressed_ctx)` low — summarizer 是个好语言模型
- 但 `recall(artifact_trail)` = 0 — summarizer 把关键 file edit 的 reasoning trail 丢了，agent 不知道自己改对了没

所以 need **behavioral probe**, not LM metric.

## 与其它压缩混淆的区别

| 维度 | Prompt Compression | Context Compression (agent) | KV-cache Compression |
|------|--------------------|----------------------------|----------------------|
| 目标 | 输入前 20K → 8K，降 latency | Session 200K → 20K，管 memory | KV N*H → N/2*H，管 GPU |
| 何时触发 | prefill 前一次性 | 每超过阈值触发，多轮 | decode 每步 |
| 痛点 | over-compress 掉 question | 丢掉 instruction / artifact | 精度掉后 answer drift |

> 只评 KV-cache 的 throughput 不看 context compression 的 completeness，等于只测 infra 不测 agent 能力。

## 结论 Takeaway

压缩评测必须：

1. 用真实 long fixture，不是 toy prompt
2. 压缩后 **再问问题**（probe），不是只比压缩前后 token 数
3. 用 judge LLM 多维度打分，而不是单一 exact match
