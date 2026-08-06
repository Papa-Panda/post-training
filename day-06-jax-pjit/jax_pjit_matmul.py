"""
Day 5/6 - JAX pjit / sharding - 10 line matmul

Goal: 说清楚 sharding 和 device mesh 是怎么回事

核心对比：
- PyTorch DDP/FSDP：你告诉它怎么并行（wrap模型、all-reduce）
- JAX：你告诉它数据怎么切（PartitionSpec），编译器自己去拼通信

这个demo CPU也能跑，重点看 device mesh 和 PartitionSpec 的对应关系
"""
# JAX 10行核心（CPU版，8 devices mock）
try:
    import jax
    import jax.numpy as jnp
    from jax.experimental.pjit import pjit
    from jax.sharding import Mesh, PartitionSpec as P

    # 1. 定义设备网格 - 想象你有 4 块卡，摆成 2x2
    # CPU 环境下会 fallback 到 1 device，但逻辑一样
    devices = jax.devices()
    print(f"[devices] {devices} count={len(devices)}")
    # 构造一个 1xN 的 mesh，适合 matmul 行切
    mesh = Mesh(jax.numpy.array(devices), ('data',))
    print(f"[mesh] axis='data' size={len(devices)}")

    # 2. 定义怎么切
    # PartitionSpec: (行怎么切, 列怎么切) None=不切
    # A:(batch, hidden) 切 batch 行
    # B:(hidden, out) 不切
    # C:(batch, out) 切 batch 行，结果也按行分
    def matmul_fn(a, b):
        return a @ b

    # 3. pjit 编译：把切分声明式告诉编译器
    # CPU单机下就是单设备跑，逻辑验证ok，待H100再看真分片
    with mesh:
        p_matmul = pjit(
            matmul_fn,
            in_shardings=(P('data', None), P(None, None)),
            out_shardings=P('data', None)
        )
        # 假数据
        A = jnp.ones((8, 4))
        B = jnp.ones((4, 2))
        C = p_matmul(A, B)
        print(f"[pjit] A{A.shape} @ B{B.shape} -> C{C.shape} expected (8,2)")
        print(f"[result] C[0]={C[0]}")  # 4.0
        print("[ok] sharding声明式：你说'行切'，编译器负责 all-gather/reduce-scatter")
        print("[contrast] PyTorch FSDP：你手动 fully_shard(block)，JAX pjit：你声明 PartitionSpec，XLA自动排程")

except ImportError as e:
    print(f"[JAX not installed] {e}")
    print("CPU fallback：用 numpy 模拟分片逻辑，待真机验证")
    import numpy as np
    A = np.ones((8,4)); B=np.ones((4,2))
    # 模拟按行切 2 份
    for i, chunk in enumerate(np.array_split(A, 2)):
        c = chunk @ B
        print(f"shard {i} {chunk.shape} -> {c.shape}, C[0]={c[0] if i==0 else '...'}")
    print("[概念] device mesh 就是把卡摆成网格，PartitionSpec 就是说矩阵的行/列贴到网格的哪一轴")
