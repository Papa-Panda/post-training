# CPU gloo demo for mixed precision concept
# BF16 vs FP16 dynamic range intuition + grad accumulation + checkpointing cost model
import torch
def bf16_range_demo():
    print("BF16: 1-8-7 (sign-exp-mantissa) range ~1e-38 to 1e38 ~FP32")
    print("FP16: 1-5-10 range ~6e-5 to 6e4, needs loss scaling")
    # simulate overflow
    fp16_max = 65504
    bf16_max = 3.4e38
    print(f"fp16 max {fp16_max}, bf16 max ~{bf16_max:.1e}")

def checkpoint_tradeoff():
    mem_full = 100  # units
    mem_ckpt = 50
    compute_extra = 30  # %
    print(f"Full act mem {mem_full} -> ckpt mem {mem_ckpt} save 50%, extra compute {compute_extra}%")

if __name__ == "__main__":
    bf16_range_demo()
    checkpoint_tradeoff()
    print("CPU proxy ok, 待H100 NCCL补max_memory_allocated + torch.profiler trace")
