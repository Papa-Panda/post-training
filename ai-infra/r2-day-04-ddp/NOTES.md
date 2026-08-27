# NOTES - r2-Day04 DDP

CPU gloo proxy，待H100 NCCL验证

- DDP原理：每rank完整模型，DistributedSampler切数据，backward后AllReduce grad SUM/WS
- 通信量：≈ 模型参数量*2*(N-1)/N，7B fp16 14GB，8卡约12.25GB per iter
- bucket：默认25MB，拼小tensor，减少AllReduce启动次数 1000次→560次
- broadcast：rank0权重广播保证初始一致，否则发散
- 手写vs DDP：Day02手写 all_reduce grad就是DDP核心，DDP加了autograd hook自动overlap

CPU proxy跑通：
- single 2.31s loss 2.142 proxy
- 2-rank gloo 3.12s comm 35% loss 2.138 一致 proxy

待H100：
- torchrun --nproc_per_node=8 all_reduce_perf 14GB 真BW
- DDP vs FSDP显存对比 14GB vs 14GB/8=1.75GB per GPU
- MFU 7B 2k seq 8卡 NVLink真数
