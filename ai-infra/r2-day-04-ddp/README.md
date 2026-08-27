> Connection to Prev: r2-Day03 通信拓扑 → r2-Day04 DDP: 上一篇算清了 NVLink 900GB/s vs IB 50GB/s 差14倍，AllReduce耗时公式；今天把 DDP 落地，Day02 的 DistributedSampler + 手写 AllReduce 正是 DDP 的原型，DDP就是把这套自动化+bucket化。

# r2-Day04 - DDP 分布式数据并行

## 昨日复盘
r2-Day03 8卡1GB Ring 1.94ms NVLink vs 27.3ms PCIe，TP不能跨机因为每层2次同步AllReduce在关键路径，PP/DP可跨机因为可overlap。

## 今日主题
**DDP 多进程梯度AllReduce同步**

- 每GPU一份完整模型，不同数据
- backward完梯度 AllReduce SUM / world_size
- bucket化：多梯度拼成25MB bucket再AllReduce，省启动开销
- DistributedSampler 保证不重复
- DDP vs 手写：DDP自动hook，overlap backward与AllReduce，broadcast初始权重

## 最小可跑任务（30-60min）
把 r2-day-02 的 loop 改 DDP：
- `dist.init_process_group(gloo, rank, world_size)`
- `model = DDP(model)` 或手写 `all_reduce grad`
- 2进程 `torchrun --nproc_per_node=2` 跑通，验证 loss一致

## 检验
- 不查说出 DDP通信量 = 参数量*2*(N-1)/N ≈ 参数量
- 能说清为何 DDP 要 broadcast 权重
- 能说清 bucket 25MB 意义

## 资源
- PyTorch DDP tutorial
- https://pytorch.org/docs/stable/distributed.html

## 待H100
CPU gloo proxy，待H100跑 NCCL 2/4/8卡真数 `torch.cuda.max_memory_allocated` 补 MFU
