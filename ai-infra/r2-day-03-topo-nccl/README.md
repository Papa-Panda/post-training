> Connection to Prev: r2-Day02 PyTorch 基础 → r2-Day03 通信拓扑: 单机train loop能跑后，多卡DDP/FSDP的瓶颈就是NVLink/IB带宽，今天把topo -m和Ring AllReduce量算清；Day02的DistributedSampler切分在今天对应到NCCL的通信拓扑选择。

# r2-Day03 - 通信拓扑 NVLink vs IB/RoCE

## 昨日复盘
r2-Day02 PyTorch 6步闭环：DataLoader→forward→loss→backward→step→scheduler→ckpt，DistributedSampler支持DDP，ckpt含model+opt+scheduler+epoch+rng，single 2.31s proxy / 2-rank gloo 3.12s。

## 今日主题
**看懂 nvidia-smi topo -m 的 NV12/SYS/NODE 和 Ring AllReduce 耗时**

- NVLink 900GB/s 单向 vs PCIe 64GB/s vs IB 400Gbps (≈50GB/s) vs RoCE
- Ring AllReduce 通信量：2*(N-1)/N * data，耗时 = 通信量 / 带宽
- AllGather/ReduceScatter 各是 AllReduce的一半，FSDP用它省显存
- Topo -m 里 NV=同NVSwitch，PIX=同PCIe switch，SYS=跨NUMA，NODE=跨机需IB

## 最小可跑任务（30-60min）
跑 `topo_demo.py` 算 8卡 1GB梯度：
- NVLink 900GB/s Ring 耗时 vs PCIe 64GB/s 差 14倍
- 解释为何TP不能跨机（带宽差10倍以上）

## 检验
- 不查说出 8卡 Ring 1GB 梯度在 NVLink 上耗时 ≈ 2*(7/8)*1GB/900GB/s ≈ 1.94ms
- 看懂 topo -m 一行 NV12 是满互联
- 能说清 AllReduce=ReduceScatter+AllGather

## 资源
- NVIDIA DL Perf Guide
- NCCL docs Ring vs Tree
- nvidia-smi topo -m manual

## 待H100
CPU proxy，待H100跑 `nvidia-smi topo -m` 真机 + `nccl-tests all_reduce_perf -b 1M -e 1G -f 2 -g 8` 补 BW 真数
