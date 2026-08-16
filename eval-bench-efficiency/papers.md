# papers — efficient eval 关键论文

1. **Metabench** — Sparse Benchmarking
   - arXiv: https://arxiv.org/abs/2407.12844
   - OpenReview: https://openreview.net/forum?id=6VX2Vk12Bs
   - GitHub: https://github.com/adkipnis/metabench
   - HF dataset: https://huggingface.co/datasets/HCAI/metabench
   - Title: Metabench -- Sparse Benchmarking of Large Language Models (ICLR 2025)

2. **Efficient Benchmarking (HELM DIoR)**
   - arXiv: https://arxiv.org/abs/2308.11696
   - Concept: Decision Impact on Reliability, x100 cost, fraction examples enough

3. **Efficient Benchmarking Is Just Feature Selection and Multiple Regression**
   - arXiv: https://arxiv.org/abs/2605.25773
   - Method: Kernel Ridge + mRMR information-theoretic feature selection, same questions across seeds, beats IRT/clustering, faster stable

4. **IRT-based math benchmark curation** (PMLR 273)
   - PMLR: https://proceedings.mlr.press/v273/ (search IRT math discrimination)
   - Method: select high discrimination $a_j$ items to improve ranking reliability, mirrors mRMR intuition

5. **Prompt Compression in the Wild**
   - arXiv: https://arxiv.org/abs/2604.02985
   - Related to eval-compression track, LLMLingua 18% speedup operating window

6. **Evaluating the Impact of Compression Techniques**
   - arXiv: https://arxiv.org/abs/2409.11233
   - Shows perplexity insufficient, JS divergence needed, task-specific calibration

7. **Factory — Evaluating Context Compression for AI Agents**
   - Blog: https://factory.ai/news/evaluating-compression
   - Hermes harness: https://github.com/ahuachen/hermes-compression-eval

8. **GLM-5.2 PPO comeback** (context: task-dependent RL)
   - Medium: https://medium.com/@ammanakhtar8/glm-5-2-the-ai-that-changed-agent-training-98f97f9f75c8
   - BigGo: https://finance.biggo.com/news/bd10eb10-5dad-4d89-9bfc-18f9ee21746d
   - VentureBeat: https://venturebeat.com/technology/z-ais-open-weights-glm-5-2-beats-gpt-5-5-on-multiple-long-horizon-coding-benchmarks-for-1-6th-the-cost

> All papers keep short Chinese takeaways in related notes, consistent with ICL / grpo-vs-ppo style.
