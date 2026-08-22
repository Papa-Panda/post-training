> Connection to Prev: r2-Day01 Transformer 架构 → r2-Day02 PyTorch 基础: 懂了Decoder Block的维度(B,S,D)后需要用PyTorch把它写成可跑的train loop才能验证；Day01的白板公式在今天用Tensor/autograd落地。

# r2-Day02 - PyTorch 基础（第二轮 Day2）

## 昨日复盘
r2-Day01 Transformer 白板：Self-Attention QKV O(N²) / FFN 2048升维 / RoPE / Pre-Norm，7B 32层手算总参≈6.7B误差<20%，白板可默写Decoder Block。

## 今日主题
**PyTorch train loop 完整闭环 + checkpoint + profiler 基础**

- Tensor / autograd / nn.Module / state_dict 关系
- 手写完整train loop：DataLoader → forward → loss → backward → step → scheduler → ckpt
- 用 torch.profiler / memory_summary 看一次iter的显存和host idle

## 最小可跑任务（30-60min）
在GPU（如无GPU用CPU proxy）跑通mnist 1 epoch，记录：
1. single-GPU time
2. 保存 `state_dict` 含 model+optimizer+epoch+rng
3. `torch.cuda.memory_summary` 或 CPU版显存估算

## 检验
- 能不查写出train loop 6步
- 说清 `state_dict` vs `model` 区别
- 能读C++ host代码中 malloc/memcpy/launch 的对应

## 资源
- PyTorch 60min Blitz
- Karpathy build GPT from scratch

## 待H100
CPU proxy，待H100 NCCL补 `torch.cuda.max_memory_allocated` + profiler trace
