# AI Infra 45天 · 知识面覆盖版（纯 AI Infra，不含机房 PUE）

> 目标：45天走完 **纯 AI Infra** 全链路，能对别人讲清每层 **牺牲什么 换取什么**，面试/系统设计能画出链路即可。
> 对标：草帽路飞 《AI Infra学习路线》四层结构 + 你的 post-training / Agentic RL 背景。
> 约束：每天 30-60min 鸟瞰式，不做深度 kernel 手搓；公共 repo 不出现雇主标识；已有 day-11/14 机房线移至 side-track。

灵感源：https://zhuanlan.zhihu.com/p/2021970155182326008

---

## 核心思维（贯穿45天）

所有优化都是 **计算 / 通信 / 显存** 不可能三角取舍：

| 技术 | 牺牲 | 换取 |
| ZeRO | 通信 | 显存 |
| Activation Checkpoint | 计算 | 显存 |
| 量化 | 精度 | 显存+带宽+吞吐 |
| Speculative Decoding | Prefill 开销 | Decode 速度 |
| Prefill/Decode 解耦 | 系统复杂度+KV迁移 | 尾延迟+goodput |
| FlashAttention | 实现复杂度 | 显存+速度 |

问法：这个技术多做了什么？省了什么？什么时候不赚？

---

## 四层架构（草帽路飞原版）

```mermaid
graph TD
  subgraph 第零层 地基
    A[Transformer白板] --> B[PyTorch循环]
    B --> C[通信拓扑]
    C --> D[DDP]
    D --> E[JAX声明式]
  end

  subgraph 第一层 CUDA算子
    F[GPU架构] --> G[CUDA基础]
    G --> H[Reduce]
    H --> I[GEMM Tiling]
    I --> J[FlashAttention]
    J --> K[Triton/compile]
    K --> L[Profiling]
  end

  subgraph 第二层 分布式训练
    M[MHA/MQA/GQA/MLA] --> N[FSDP/ZeRO]
    N --> O[TP/PP/SP 3D]
    O --> P[混合精度/重计算]
    P --> Q[框架选型]
    Q --> R[Checkpoint/DCP]
  end

  subgraph 第三层 推理部署
    S[Prefill vs Decode] --> T[KV Cache算账]
    T --> U[PagedAttention+ContBatch]
    U --> V[PrefixCache/Chunked]
    V --> W[vLLM/SGLang实战]
    W --> X[量化决策树]
    X --> Y[量化实战]
    Y --> Z[Spec Decoding]
    Z --> AA[Spec实测]
    AA --> AB[Prefill/Decode解耦]
    AB --> AC[Goodput配比]
    AC --> AD[Benchmark 6指标]
    AD --> AE[回归门禁]
  end

  E --> F
  L --> M
  R --> S
```

### Phase 定义

- **第零层 01-05天 地基「够用即可」**：Transformer Decoder 手绘、PyTorch loop、topo NVLink 900GB/s vs PCIe 64GB/s、DDP 30min改造、JAX Mesh 声明式
- **第一层 06-13天 CUDA**：GPU存储 寄存器>ShMem>L1/L2>HBM，coalesced，最朴素Reduce→Shuffle，Tiled GEMM 50% cuBLAS，FlashAttention tiling+online softmax，Triton fused，Nsight System看host拖后+Compute看SOL
- **第二层 14-19天 分布式**：MHA→MLA省KV，7B FP16 14GB+Adam 56GB单80GB判断，ZeRO三句区分，64卡TP=8机内PP=4 DP=2为何TP不能跨机，BF16指数8位vs 5位，Megatron/DeepSpeed/FSDP一句选型，DCP async ckpt
- **第三层 21-32天 推理**：Compute vs Memory bound，7B 2*32*32*128*4096*16*2B≈32GB cache，vLLM Paged虚拟页，Static vs Continuous 30%→80%，Shared前缀复用，长prefill分块防拖慢，真机部署对比表，70B 2*A100 140GB→INT4 35GB压法，INT4慢于INT8边界，Spec实习生草稿+主编验证无偏，DistServe/Splitwise网约车分拣，Goodput才等于体验，GenAI-Perf一键6指标，TPOT P95退化5%即block
- **Portfolio 33-45天 连接到你的 post-training**：RLHF vs GRPO，RM ECE 0.0906→0.0881 σ0.045，ToolUse 5类失败，训练→vLLM流水 async省52%墙，$/有用，coding data飞轮，Star 150字 $200M迁移，E2E Demo，系统设计白板，复盘/LinkedIn honest/Final总结

---

## 45天日历（知识面覆盖版，已写入 ai_infra Sheet）

> 每行=鸟瞰问题+可验证产出，30-60min，不求手搓极致
> Sheet ID `1JxGiuanIdwHWD2_e6FRMiVLBvB2kNnD8QAree8ZXfUc` gid `ai_infra` A2:J46 已重写 45行

- 01 Transformer架构
- 02 PyTorch loop
- 03 通信拓扑 NVLink/IB/NCCL
- 04 DDP
- 05 JAX pjit
- 06 GPU架构 HBM 3.35TB/s
- 07 CUDA编程模型
- 08 Reduce三连
- 09 GEMM Tiling
- 10 FlashAttention白板
- 11 Triton/torch.compile
- 12 Profiling Nsight 双剑
- 13 Attention变种 MQA/GQA/MLA/MoE
- 14 FSDP/ZeRO 显存账
- 15 TP/PP/SP 64卡拓图
- 16 BF16混合精度/重计算
- 17 Megatron/DeepSpeed/FSDP选型
- 18 Checkpoint&Recovery DCP
- 19 复盘周
- 20-21 TTFT/TPOT & KV Cache 32GB手算
- 22-23 PagedAttention/Cont Batch/Prefix
- 24-25 vLLM实战部署对比表
- 25-27 量化决策树+实战 INT4 35GB
- 27-29 Spec Decoding + 实测场景
- 29-31 Disagg + Goodput配比1:3
- 31-32 Benchmark 6指标+门禁
- 33-40 RL芯连接：GRPO/RM σ/ECE/ToolUse 5失败/vLLM联动/async eval/$有用/coding flywheel/Star故事
- 41-45 E2E Demo+系统设计+复盘+LinkedIn honest+Final 300字

---

## 已有 side-track（原机房线）说明

- `day-11-paper2-mech-load` SSML→GPU Tj 82.49C throttle0.83% / `day-14-pue-cost` PUE 1.2576 25.76% overhead / `day-06-paper1-rl-infra` autoscaling 已保留，不在主干考核，属于 **ML for Infra→AI Infra可迁移思维**，面试可讲方法论迁移但不算AI Infra核心。

---

## 检验标准（粗略理解版，不求完美）

- 白板默写 Decoder Block (B,S,D) 无错
- 给7B hidden 4096 32层 手算总参误差<20%
- DDP改造不查文档30min跑通
- 8卡 topo -m 说清NV12 vs SYS
- 能画 AllReduce Ring通信量公式
- FlashAttention白板画 tiling外层KV内层Q SRAM完成
- 说清ZeRO-2 vs ZeRO-3 一句区分：2只AllReduce梯度，3连参数也切forward/back都要Gather通信翻倍但显存1/N
- KV Cache手算32GB判断80GB剩多少
- 能解释 Prefill快Decode慢 compute vs memory bound
- 讲清 PagedAttention虚拟页表思想像OS，Continuous Batching网约车拼单
- 量化选型：70B 2卡怎么压一句话到单卡的tradeoff
- Spec Sampling无偏保证 rejection sampling数学等价
- DistServe动机：混批TPOT P95被prefill拖慢3-5倍定量
- Benchmark能输出固定模板6指标复现config只改一变量
- `43-45` 讲出 Star 2分钟省$200M→RL稳定故事 队列68.8%热1.67pp成本22.1%（待H100真机替换待H100标记）

---

## 接下来怎么用这个 repo

1. 每天按 Sheet 完成 `Small Task`，笔记进对应 `rl-infra/day-0x/NOTES.md` 加一句 Connection to Prev: Day{prev}→Day{curr}
2. 45天后你有：DDP/FSDP/ZeRO账、FlashAttention白板、vLLM对比表、量化决策树、Spec/Disagg动机、6指标Benchmark模版、RLHF/GRPO一页、RM校准小跑、5失败库、Star故事
3. 这就是你想的“多少有一点粗略理解”——全链路都摸过一遍，面试能指到瓶颈，不露怯

> 关联：`ai_daily.csv` 同步需 manually 更新 status→done，`ai-data/` 30篇图谱互补（LESS/TRACIN/STAR...），`eval-xxx` 两条评测轨并行
