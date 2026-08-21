# Day 5/6 - JAX pjit / sharding

**Learning Goal**: 理解 JAX 比 PyTorch 多解决了什么

PyTorch DDP/FSDP 是命令式：你告诉它怎么并行（wrap、fully_shard、all-reduce）  
JAX pjit 是声明式：你告诉它数据怎么切，编译器自己去排通信

## sharding 和 device mesh 是咋回事

1. **device mesh** - 把你的卡摆成网格
   - 比如 8 卡摆成 `('data','model') = 2x4`
   - 就像你搭机架，决定横向是数据并行，纵向是模型并行

2. **PartitionSpec** - 说数组的每一维贴到 mesh 的哪一轴
   - `P('data', None)` = 行切到 data 轴，列不切
   - `P(None, 'model')` = 列切到 model 轴
   - `P('data','model')` = 2D 切

3. **效果** - matmul A(8,4) @ B(4,2)
   - 声明 `A` 按行切 `P('data',None)`，`B` 不切，`C` 按行切
   - 编译器自动在需要时插 all-gather / reduce-scatter
   - 类比：物理建模里多变量耦合，mesh 是变量图，sharding 是哪个变量在哪算

## 10行跑通
`python jax_pjit_matmul.py` CPU 也能跑，真机再看 device 8+ 上的分片。

Status: CPU验证ok，待H100验证真分片，逻辑已通。
