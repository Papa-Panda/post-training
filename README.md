# Post-training Systems & Data

A public, implementation-oriented knowledge base for post-training: objectives, data, distributed systems, rollout serving, evaluation, agent runtimes, and the GPU mechanisms underneath them.

The repository favors a consistent progression:

1. write down the objective and assumptions;
2. derive the relevant compute, communication, or memory cost;
3. connect the derivation to a system design;
4. run a small executable model or harness;
5. test semantic invariants, not only syntax.

## Start here

| Track | Primary question | Suggested entry point |
|---|---|---|
| [AI infrastructure](ai-infra/README.md) | How do training and inference systems trade compute, communication, and memory? | Current second-pass labs, then the 45-day map |
| [GPU architecture](gpu-architecture/README.md) | Which hardware and kernel mechanisms create those costs? | SIMT → memory → GEMM → collectives → profiling |
| [PPO vs. GRPO](grpo-vs-ppo/README.md) | How do the objectives and training-system requirements differ? | Objective derivations before infra trade-offs |
| [vLLM rollout](vllm-rollout/README.md) | How should rollout serving be measured and stress-tested? | TTFT/TPOT metrics → configuration → failures |
| [Model-aware data curation](model-aware-data-curation/README.md) | Which examples move the current model toward a target while preserving coverage and safety? | Attribution → gradient coverage → closed-loop selection |
| [AI data reading track](ai-data/README.md) | What do the major data-selection, synthesis, and filtering papers contribute? | Paper index and reading log |
| [In-context learning](ICL/README.md) | How can learning-like behavior arise from context without weight updates? | Bayesian, gradient-descent, and circuit views |
| [Harness engineering](harness-engineering/README.md) | How should a frozen model's context, workflow, tools, memory, and release gates be engineered? | Runtime loop → state/memory → evaluation/security |
| [Evaluation: context compression](eval-context-compression/README.md) | Does compressed context preserve task-relevant behavior? | Evaluation design and probe harness |
| [Evaluation: benchmark efficiency](eval-bench-efficiency/README.md) | Can a smaller benchmark preserve ranking and decision quality? | IRT/mRMR methods and practical checks |
| [Evaluation index](eval/README.md) | Where are the evaluation subtracks? | Navigation only |

## Topic boundaries

- **`ai-infra/`** is the training/inference systems spine: DDP, FSDP, sharding, collectives, checkpointing, rollout serving, and performance models.
- **`gpu-architecture/`** goes one layer lower: SIMT execution, memory hierarchy, Tensor Cores, CUDA, interconnects, virtual memory, and profiling. It does not duplicate end-to-end distributed-training labs.
- **`vllm-rollout/`** specializes in serving and rollout behavior; `ai-infra/` links to it rather than maintaining a second canonical copy.
- **`ai-data/`** is a paper-reading corpus. **`model-aware-data-curation/`** is a cross-paper synthesis and runnable model-in-the-loop selection system.
- **`ICL/`** studies behavior induced by context. **`harness-engineering/`** studies the executable system that constructs context, calls tools, manages state, and promotes changes.
- **`grpo-vs-ppo/`** owns optimization-objective comparisons; infrastructure tracks discuss only their systems consequences.
- **`eval-*`** tracks own measurement methodology and should not be treated as training or serving implementations.

## Repository status

This is a learning repository, not a benchmark leaderboard or a production framework.

- CPU analytical models and simulations are labeled as such.
- Hardware throughput, latency, memory, and scaling numbers are not considered measured unless the corresponding command, configuration, and environment are recorded.
- Recent papers and preprints are separated from mature mechanisms where the distinction matters.
- Primary papers and official documentation are preferred for technical claims.
- GitHub display math uses one-line `$$...$$` blocks.

## Quick checks

The topic directories with runnable suites can be checked independently:

```bash
# Whole-repository links, GitHub math, control characters, and Python parsing.
python3 tools/check_repo.py

# Topic-level semantic suites.
python3 -m unittest discover -s ICL/tests -v
(cd grpo-vs-ppo/05_code && python3 -m unittest -v test_rl_objectives.py test_docs.py)
python3 -m unittest discover -s vllm-rollout/tests -v
python3 -m unittest discover -s model-aware-data-curation/tests -v
python3 -m unittest discover -s harness-engineering/tests -v
python3 -m unittest discover -s gpu-architecture/tests -v
python3 -m unittest discover -s ai-infra/day-07-h100-beyond-7b -p 'test_*.py' -v
python3 -m unittest discover -s ai-infra/r2-day-03-topo-nccl -p 'test_*.py' -v
python3 -m unittest discover -s ai-infra/r2-day-04-ddp -p 'test_*.py' -v
python3 -m unittest discover -s ai-infra/r2-day-06-gpu-architecture -p 'test_*.py' -v
python3 -m unittest discover -s ai-infra/r2-day-07-cuda-programming-model -p 'test_*.py' -v
```

Some labs additionally require PyTorch, JAX, CUDA, NCCL, or a multi-GPU host. A successful CPU model is evidence for its formulas and control flow only; it is not evidence of accelerator performance.
