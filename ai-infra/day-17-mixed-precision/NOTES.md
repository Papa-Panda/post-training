# NOTES - Day17 混合精度

CPU proxy，待H100 NCCL验证

- BF16指数8位 vs FP16 5位，动态范围近FP32，免Loss Scale
- 7B FP16 14GB + Adam FP32副本+动量 56GB = 70GB 单80GB临界，必须ZeRO或BF16
- 重计算：compute +30%换显存 -50%，ZeRO：comm +100%换显存 1/N

待H100：
- 跑BF16 vs FP16 对比，记录max_memory_allocated
- torch.profiler看comm占比 7-15%预期
