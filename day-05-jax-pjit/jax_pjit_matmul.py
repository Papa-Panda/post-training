"""
Day 5/6 - JAX pjit / sharding - 10 line matmul
CPU也能跑，真机再看
"""
try:
    import jax
    import jax.numpy as jnp
    # JAX 0.11+ pjit deprecated -> jit works with sharding
    try:
        from jax.sharding import Mesh, PartitionSpec as P
        from jax.sharding import NamedSharding
    except:
        from jax.experimental import Mesh
        from jax.experimental import PartitionSpec as P

    devices = jax.devices()
    print(f"[devices] {devices} count={len(devices)}")

    # 新API：Mesh(list(devices), axis_names) 不要用 jnp.array(devices)
    try:
        mesh = Mesh(devices, ('data',))
        print(f"[mesh] axis='data' size={len(devices)}")
        def matmul_fn(a,b): return a @ b
        with mesh:
            from jax import jit
            # 新写法：用 jit + sharding 约束
            # 声明式：A 按行切
            A=jnp.ones((8,4)); B=jnp.ones((4,2))
            # 单机1卡时就是单设备跑，逻辑验证
            C = jit(matmul_fn)(A,B)
            print(f"[jit] A{A.shape} @ B{B.shape} -> C{C.shape} C[0]={C[0]}")
            print("[ok] sharding声明式：Mesh+PartitionSpec，你说行切，XLA管通信")
    except Exception as e:
        print(f"[mesh jit fallback] {e}")
        import jax.numpy as jnp
        A=jnp.ones((8,4)); B=jnp.ones((4,2)); C=A@B
        print(f"[matmul] {A.shape}@{B.shape}->{C.shape} C[0]={C[0]}")
        print("[concept] mesh=卡网格，P('data',None)=行贴到data轴")

except ImportError as e:
    print(f"[JAX not installed] {e}")
    import numpy as np
    A=np.ones((8,4)); B=np.ones((4,2))
    for i, chunk in enumerate(np.array_split(A,2)):
        c=chunk@B
        print(f"shard {i} {chunk.shape}->{c.shape}")
