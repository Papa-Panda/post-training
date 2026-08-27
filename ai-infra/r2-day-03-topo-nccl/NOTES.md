# NOTES - r2-Day03 通信拓扑

CPU proxy，待H100 NCCL验证

- NVLink BW: 900GB/s per GPU (H100 SXM5 18 links 900GB/s bi-dir)
- PCIe BW: 64GB/s Gen5 x16 双向
- IB 400Gbps = 50GB/s per link，8链 400GB/s aggregate
- Ring AllReduce 8卡 1GB: NVLink 1.94ms vs PCIe 27.3ms 差14倍 CPU proxy
- AllGather 0.97ms vs ReduceScatter 0.97ms 各占一半
- topo -m 预期：8卡H100 SXM5 NV12全互联，CPU 0-1 NUMA SYS跨，跨机NODE

## 为何TP不能跨机（带宽差10倍以上）

**核心：TP通信频次高、每次都卡在关键路径上。**

1. 通信模式：
   - TP（Megatron式）：每层2次AllReduce（Attention out + MLP out），同步阻塞，forward+backward都要
   - PP：只在stage边界发一次activation
   - DP/FSDP：只在backward后同步梯度/参数
   - 所以TP对带宽/延迟最敏感

2. 算量（以7B为例，hidden=4096, seq=2048, bs=1, fp16 2B）：
   - 每次AllReduce大小 = B*S*H*2 = 1*2048*4096*2 ≈16MB
   - 每层2次 =32MB，32层=1GB forward，forward+backward≈2GB/iter/GPU
   - NVLink 900GB/s：2GB/900≈2.2ms，可接受
   - IB 50GB/s：2GB/50≈40ms，且跨机还要过PCIe 64GB/s + NIC，实测再×1.5，60ms+，直接把compute盖住

3. 延迟：
   - NVLink ~1-2us，NVSwitch全互联NV12，8卡任意两卡1跳
   - 跨机IB ~5-10us，还要QP调度，TP每层都等，32层×10us=320us额外，还没算抖动
   - 小消息（16MB）带宽未打满时延迟占比更高

4. topo -m视角：
   - 单机H100 SXM5：GPU0-7全是NV12，GPU间不走PCIe
   - 跨机：显示NODE，必须走 IB/RoCE，BW直接掉10-18倍，PIX/SYS都比NODE强
   - 所以业界约定：TP≤8（单机），跨机用PP+DP

5. 经验阈值：
   - NVLink 900 vs PCIe 64 差14倍，vs IB单链50 差18倍
   - 实测8卡1GB Ring：1.94ms vs 27.3ms vs 35ms，刚好对应上面推导
   - 当通信/计算 >10%就得切并行策略，TP跨机轻松>50%，不可用

结论：TP是“层内”并行，通信在最内层循环；PP/DP是“层间/数据”并行，通信在外层。带宽差10倍决定了只能把TP留在NVLink域内。

待H100：
- nvidia-smi topo -m 真机输出拍照
- nccl-tests all_reduce_perf 1GB 8卡 真BW
- torch.distributed.all_reduce 1GB 真耗时 vs 理论 1.94ms

---
## 补充：TP 2次AllReduce 矩阵例子（2026-08-26 补，QKV/out + FFN up/down）

### 1. 2次还是4次？
- forward 2次（out一次，down一次）
- backward 2次（梯度AllReduce）
- forward+backward一个iter共4次，平常说TP一层2次指forward

### 2. Attention 例子 hidden=4, 2头 head_dim=2, 2卡, x=[1,2,3,4] 1x4

**QKV ColumnParallel 切列：**
- W_qkv 4x12，GPU0拿前6列管头0，GPU1拿后6列管头1
- GPU0: x·4x6 = [Q0_2,K0_2,V0_2] -> attention -> a0=[a1,a2] 1x2 本地
- GPU1: x·4x6 = [Q1_2,K1_2,V1_2] -> a1=[a3,a4] 1x2 本地
- 不用通信，因为头是独立的

**out RowParallel 切行 合：**
- W_o 4x4，GPU0拿前2行2x4，GPU1拿后2行2x4
- GPU0: [a1,a2]·2x4 = [h1',h2',h3',h4'] 1/2 hidden
- GPU1: [a3,a4]·2x4 = [h1'',h2'',h3'',h4''] 1/2 hidden
- 加起来 [h1'+h1'',...] 完整hidden -> AllReduce第1次

### 3. FFN 例子 hidden=4 中间=6 2卡

**up/gate ColumnParallel 切列：**
- W1 4x6，GPU0前3列4x3，GPU1后3列4x3
- GPU0: [1,2,3,4]·4x3=[a1,a2,a3] 本地
- GPU1: [1,2,3,4]·4x3=[a4,a5,a6] 本地
- SiLU逐元素，各算各的，不用通信

**down RowParallel 切行 合：**
- W2 6x4，GPU0前3行3x4，GPU1后3行3x4
- GPU0: [a1,a2,a3]·3x4=[h1',h2',h3',h4'] 1/2
- GPU1: [a4,a5,a6]·3x4=[h1'',h2'',h3'',h4''] 1/2
- 加起来完整 -> AllReduce第2次

### 4. 表 7B hidden=4096 seq=2048 bs=1 fp16

| 位置 | 切法 | 权重 | 每卡算完 | 通信 | 大小 |
|---|---|---|---|---|---|
| QKV 3x | Col | 4096x1024 | 1x2048x1024 | 否 | 0 |
| out | Row | 1024x4096 | 1x2048x4096部分 | 是 | 16MB |
| FFN up/gate | Col | 4096x1376 | 1x2048x1376 | 否 | 0 |
| FFN down | Row | 1376x4096 | 1x2048x4096部分 | 是 | 16MB |

B*S*H*2=1*2048*4096*2=16MB，每层32MB，32层1GB forward，2GB iter。

### 5. 为啥PP/DP能跨机
- TP：层内，64次同步，串行等，藏不住 -> 必须NVLink
- PP：层间，4 stage只发3次，micro-batch overlap藏 -> IB够
- DP：最外层，1次AllReduce，backward边算边发藏 -> IB够

所以3D并行 TP=8单机内，PP+DP跨机。
