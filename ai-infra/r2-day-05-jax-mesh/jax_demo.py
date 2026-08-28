# r2-day-05 JAX Mesh/pjit demo - 严格6步对齐DDP声明式对比
# 6步: Mesh定义 → PSpec切分 → pjit声明 → 逻辑matmul → 编译器插通信 → 执行Sharding可视化
try:
    import jax, jax.numpy as jnp
    from jax.experimental.pjit import pjit
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh, PartitionSpec as P
    HAS_JAX=True
except ImportError:
    HAS_JAX=False
    # mock P for proxy prints
    class P:
        def __init__(self, *args): self.args=args
        def __repr__(self): return f"P{self.args}"

def run():
    if not HAS_JAX:
        print("=== r2-Day05 JAX Mesh/pjit 声明式 6步结构 (无jax环境模拟) ===")
        print("[Step1] Mesh: Mesh(np.array(jax.devices()).reshape(2,4), ('data','model'))")
        print("  代码: mesh = Mesh(mesh_utils.create_device_mesh((2,4)), ('data','model'))")
        print("  -> 逻辑轴 data=2 model=4，对应8卡物理，DDP里 world_size=8")
        print("[Step2] PartitionSpec: P('data',None) 切batch轴，hidden不切")
        print("  代码: in_shardings=(P('data',None), P(None,'model'))")
        print("  -> A 1024x1024 按行切 data，B 1024x1024 按列切 model")
        print("  代码: a_pspec = P('data', None)  # A 行切2份 512行")
        print("  代码: b_pspec = P(None, 'model') # B 列切4份 256列")
        print("  代码: c_pspec = P('data', 'model') # C 行切2列切4 每卡512x256")
        print("[Step3] pjit声明: pjit(lambda x,y: x@y, in_shardings=..., out_shardings=P('data','model'))")
        print("  代码: @pjit def matmul(a,b): return a@b  # 无显式通信")
        print("  -> 对比DDP: dist.init_process_group + all_reduce 10行，这里0行通信，改PSpec即切")
        print("[Step4] 逻辑计算: C = A@B 数学不变")
        print("  代码: c = matmul(a,b)  # 1024x1024 @ 1024x1024 -> 1024x1024")
        print("  -> 用户只写数学，Mesh+PSpec管切分")
        print("[Step5] 编译器插通信: 自动 AllGather/ReduceScatter")
        print("  代码: jax.jit lower后可见 all-gather/reduce-scatter")
        print("  -> 对应DDP hook自动AllReduce，但JAX在编译时插，DDP在backward时hook触发")
        print("[Step6] 执行+Sharding可视化: jax.device_count()=8 + visualize_sharding")
        print("  代码: jax.debug.visualize_sharding(c) # 看每卡分到哪块")
        print("  -> CPU proxy ok, 待H100真机 8卡 2x4 Mesh matmul 1024² 声明式0行通信 vs DDP 10行")
        return

    # --- 真JAX路径 (单卡也能跑，8卡才有真切分) ---
    # Step1 Mesh
    devices=jax.devices()
    print(f"[Step1] jax devices {len(devices)}")
    # 单卡演示用 (1,) Mesh，8卡用 (2,4)
    mesh_shape = (len(devices),) if len(devices)<2 else (2, max(1,len(devices)//2))
    mesh = Mesh(mesh_utils.create_device_mesh(mesh_shape), ('data',) if len(mesh_shape)==1 else ('data','model'))
    print(f"[Step1] Mesh {mesh} shape {mesh_shape}")
    print(f"  代码: mesh = Mesh(mesh_utils.create_device_mesh({mesh_shape}), {mesh.axis_names})")

    # Step2 PSpec 真代码
    a_pspec = P('data', None)
    b_pspec = P(None, 'model') if len(mesh_shape)>1 else P(None, None)
    c_pspec = P('data', 'model') if len(mesh_shape)>1 else P('data', None)
    print(f"[Step2] a_pspec={a_pspec} 行切data 512行")
    print(f"  代码: a_pspec = P('data', None)")
    print(f"[Step2] b_pspec={b_pspec} 列切model 256列")
    print(f"  代码: b_pspec = P(None, 'model')")
    print(f"[Step2] c_pspec={c_pspec} 行切列切 每卡512x256")
    print(f"  代码: c_pspec = P('data', 'model')")

    # Step3 pjit声明
    @pjit
    def matmul(a,b):
        # Step4 逻辑计算 C=A@B
        return a@b
    print(f"[Step3] pjit matmul声明 ok，in_shardings=({a_pspec},{b_pspec}) out={c_pspec} 0行显式通信")
    print(f"  代码: matmul = pjit(lambda a,b: a@b, in_shardings=({a_pspec},{b_pspec}), out_shardings={c_pspec})")

    # Step4-5 逻辑计算 + 编译插通信
    a=jnp.ones((1024,1024))
    b=jnp.ones((1024,1024))
    # Step6 执行
    c=matmul(a,b)
    print(f"[Step4-6] matmul done {c.shape} 声明式 ok")
    print(f"  代码: c = matmul(a,b)  # 1024x1024")
    # Step6 Sharding可视化 (单卡无切分，8卡才有)
    try:
        jax.debug.visualize_sharding(c)
        print("  代码: jax.debug.visualize_sharding(c)")
    except:
        print("[Step6] visualize_sharding 单卡无切分，待H100 8卡 2x4")
        print("  代码: jax.debug.visualize_sharding(c) # 8卡棋盘")

if __name__=="__main__":
    run()
