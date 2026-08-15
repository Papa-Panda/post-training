"""
Day 11 — Paper2 机械负载 → GPU 热/功耗 映射
CPU gloo 2-rank 可跑，H100 待补 max_memory_allocated

对应 ai_daily 2026-08-11:
  Track: Reasoning Data / Papers
  Topic: Paper2拆解-机械负载
  Knowledge: 非线性物理系统建模技巧
  Goal: 提炼技巧，写状态空间，对应到 GPU 热/功耗
  Small Task: 写 Paper2 状态空间模型，思考对应 GPU

Run:
  torchrun --nproc_per_node=2 paper2_mech_to_gpu_thermal.py
  python paper2_mech_to_gpu_thermal.py   # fallback single rank
"""
import os, math, time, random, json
import torch
import torch.distributed as dist

def is_dist():
    return dist.is_available() and dist.is_initialized()

def setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        try:
            dist.init_process_group(backend="gloo")
            return True
        except Exception as e:
            print(f"[setup] gloo init fail {e}, fallback single")
            return False
    return False

class MechLoadSSM:
    """
    Paper2 复刻：数据中心机械负载 非线性状态空间

    状态 x = [Q_mech (kW), T_chw_s (°C), T_chw_r (°C)]
      Q_mech: 当前机械制冷量需求
      T_chw_s: 冷冻水供水温度
      T_chw_r: 冷冻水回水温度

    输入 u = [Q_IT (kW), T_wb (°C, wet bulb), setpoint_T_s]

    动力学:
      Q_mech_{t+1} = (1 - 1/τ_Q) Q_mech_t + (1/τ_Q) * Q_IT_t * (1 + α*(T_wb - 25)) + f_stage
      f_stage 非线性：冷水机台数阶梯 = ceil(Q_IT / Q_rated_chiller) * hysteresis
      T_chw_s_{t+1} = T_chw_s_t + dt/C_w * (Q_mech_t - m_dot*cp*(T_chw_r - setpoint) + γ*(T_chw_s - T_chw_r)^2)
      二次项 γ 体现换热非线性

    输出 y = P_mech = Q_mech / COP, COP = COP_ref * (1 - β*(T_wb - 25) - κ*PLR^2)

    非线性技巧（Paper2 提炼）：
      1) bilinear IT * 外温耦合
      2) quadratic/cub fan/pump law → 二次项
      3) hysteresis 防止频繁加减机
      4) 热容 τ 一阶惯性 + 噪声
    """
    def __init__(self, dt=1.0, n_chiller=3, Q_rated=500.0, seed=42):
        self.dt = dt
        self.n_chiller = n_chiller
        self.Q_rated = Q_rated  # kW per chiller
        random.seed(seed)
        self.tau_Q = 5.0  # min
        self.C_w = 4180 * 1000 / 3600  # water thermal mass approx
        self.alpha = 0.015
        self.beta = 0.02
        self.kappa = 0.15
        self.gamma = 0.008
        self.COP_ref = 6.5
        self.hyst_on = 0.85
        self.hyst_off = 0.35

    def step(self, x, u, stage_state):
        Q_mech, T_s, T_r = x
        Q_IT, T_wb, set_s = u
        # chiller staging with hysteresis
        load_ratio = Q_IT / self.Q_rated
        desired_n = min(self.n_chiller, max(1, math.ceil(load_ratio / self.hyst_on)))
        # hysteresis off
        if stage_state["n"] > desired_n and (Q_IT / (stage_state["n"]*self.Q_rated) < self.hyst_off):
            n_active = max(1, stage_state["n"]-1)
        else:
            n_active = max(stage_state["n"], desired_n)
        stage_state["n"] = n_active

        # Q_mech dyn + nonlinearity
        f_stage = 0.02 * (n_active - 1) * self.Q_rated * 0.05  # small overhead
        Q_mech_next = (1 - self.dt/self.tau_Q)*Q_mech + (self.dt/self.tau_Q)*(Q_IT*(1+self.alpha*(T_wb-25)) + f_stage)
        Q_mech_next += random.gauss(0, 2.0)  # process noise

        # water temps
        m_dot_cp = 50.0  # kW/K approx
        T_s_next = T_s + self.dt/self.C_w*(Q_mech - m_dot_cp*(T_r - set_s) + self.gamma*(T_s - T_r)**2*100)
        T_r_next = T_r + self.dt/3.0*( (Q_IT/ (m_dot_cp*1.2) ) - 0.3*(T_r - T_s) ) + random.gauss(0,0.05)

        T_s_next = max(5.0, min(12.0, T_s_next))
        T_r_next = max(8.0, min(18.0, T_r_next))
        Q_mech_next = max(50.0, Q_mech_next)

        # COP & power
        PLR = min(1.0, Q_mech_next / (n_active*self.Q_rated))
        COP = self.COP_ref * (1 - self.beta*(T_wb-25) - self.kappa*PLR**2)
        COP = max(2.0, COP)
        P_mech = Q_mech_next / COP

        return (Q_mech_next, T_s_next, T_r_next, P_mech, COP, n_active)

class GPUThermalSSM:
    """
    GPU 热/功耗 SSM — Paper2 迁移

    对应：
      Q_IT (IT负载 kW)  → P_gpu (W) rollout power
      Q_mech (机械制冷) → Q_cool (散热需求)
      T_chw (水温)      → T_j (junction), T_hs (heatsink)
      COP 非线性        → 热阻非线性 + 风扇立方定律
      hysteresis (冷机启停) → 风扇/节流 hysteresis (防抖)

    状态 x_gpu = [T_j, T_hs] °C
    输入 u_gpu = [P_gpu (W), T_ambient, fan_rpm_ratio, throttle_flag]
    输出 y = throttled? + power cap hit?

    动力学 (两节点热容模型):
      C_j dT_j/dt = P_gpu - (T_j - T_hs)/R_jh + noise
      C_hs dT_hs/dt = (T_j - T_hs)/R_jh - (T_hs - T_amb)/R_hs(fan) - hyst_cooling
      R_hs(fan) = R0 * (1 / (fan_ratio^0.8 + 0.1)) — 风扇曲线非线性，立方定律影子
      hysteresis: T_j>85C 触发降频 150ms 内 P_gpu *=0.7，T_j<75C 恢复，类似 cold start 防抖
    """
    def __init__(self, dt=0.1, seed=42):
        random.seed(seed+1)
        self.dt = dt
        self.C_j = 28.0      # J/K - larger thermal cap = realistic 70-90C
        self.C_hs = 220.0
        self.R_jh = 0.06    # K/W
        self.R0 = 0.09
        self.throttle_on = 82.0
        self.throttle_off = 72.0
        self.throttled = False

    def step(self, x, u):
        T_j, T_hs = x
        P_gpu, T_amb, fan_ratio, _ = u
        # throttle hysteresis
        if not self.throttled and T_j > self.throttle_on:
            self.throttled = True
        elif self.throttled and T_j < self.throttle_off:
            self.throttled = False

        eff_P = P_gpu * (0.68 if self.throttled else 1.0)

        R_hs = self.R0 * (1.0 / (fan_ratio**0.8 + 0.15))
        # two-node
        dTj = self.dt/self.C_j * (eff_P - (T_j - T_hs)/self.R_jh + random.gauss(0,0.3))
        dThs = self.dt/self.C_hs * ((T_j - T_hs)/self.R_jh - (T_hs - T_amb)/R_hs + random.gauss(0,0.2))

        T_j_next = max(30.0, T_j + dTj)
        T_hs_next = max(25.0, T_hs + dThs)

        power_overhead = (T_hs_next - T_amb)/R_hs * 0.04  # fan power ~ cubic law proxy

        return (T_j_next, T_hs_next, self.throttled, eff_P, power_overhead, R_hs)

def simulate(steps=120, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    mech = MechLoadSSM(seed=seed)
    gpu = GPUThermalSSM(seed=seed)

    x_mech = (300.0, 7.0, 12.0)
    stage = {"n":1}
    x_gpu = (45.0, 35.0)

    mech_hist=[]; gpu_hist=[]; power_hist=[]
    true_Q=[]; pred_Q=[]

    # simple naive predictor for eval: EWMA of Q_IT
    alpha=0.3
    ewma_Q=300.0

    for t in range(steps):
        # IT load: base 400kW + burst + sinusoid
        burst = 200.0 if (t%40>35) else 0.0
        Q_IT = 350 + 80*math.sin(t/15.0) + burst + random.gauss(0,10)
        T_wb = 22 + 4*math.sin(t/30.0) + random.gauss(0,0.4)
        set_s = 7.0

        Q_mech_next, T_s_next, T_r_next, P_mech, COP, n_active = mech.step(x_mech, (Q_IT, T_wb, set_s), stage)

        # EWMA pred vs true for RMSE
        pred = ewma_Q
        ewma_Q = alpha*Q_IT + (1-alpha)*ewma_Q
        true_Q.append(Q_IT)
        pred_Q.append(pred)

        # Map Q_IT (kW rack) → P_gpu per GPU (~ 700W H100 * G)
        # 假定 1 rack ~ 32 GPU H100, 400kW → ~ 700W per GPU scaling
        P_gpu = 450 + (Q_IT-350)*1.8 + random.gauss(0,8)  # 450-750W range
        T_amb = 24 + (T_wb-22)*0.6
        fan_ratio = 0.6 + 0.4*max(0, int(gpu.throttled or P_gpu>650)) + 0.1*math.sin(t/10)
        fan_ratio = max(0.3, min(1.0, fan_ratio))

        T_j, T_hs, throttled, eff_P, fan_over, R_hs = gpu.step(x_gpu, (P_gpu, T_amb, fan_ratio, 0))

        x_mech = (Q_mech_next, T_s_next, T_r_next)
        x_gpu = (T_j, T_hs)
        mech_hist.append((Q_mech_next, P_mech, COP, n_active))
        gpu_hist.append((T_j, T_hs, throttled))
        power_hist.append(P_mech + fan_over)

    # metrics
    rmse = math.sqrt(sum((t-p)**2 for t,p in zip(true_Q, pred_Q))/len(true_Q))
    avg_P_mech = sum(m[1] for m in mech_hist)/len(mech_hist)
    throt_rate = sum(1 for g in gpu_hist if g[2])/len(gpu_hist)
    Tj_max = max(g[0] for g in gpu_hist)
    Tj_avg = sum(g[0] for g in gpu_hist)/len(gpu_hist)
    p_mech_std = math.sqrt(sum((p - avg_P_mech)**2 for _,p,_,_ in mech_hist)/len(mech_hist))

    return {
        "rmse_Q_pred_kW": rmse,
        "avg_P_mech_kW": avg_P_mech,
        "p_mech_std_kW": p_mech_std,
        "throttle_rate": throt_rate,
        "Tj_max_C": Tj_max,
        "Tj_avg_C": Tj_avg,
        "mech_samples": mech_hist[:3],
        "gpu_samples": gpu_hist[:3],
        "steps": steps
    }

def main():
    started_dist = setup()
    rank = dist.get_rank() if is_dist() else 0
    world = dist.get_world_size() if is_dist() else 1

    t0 = time.time()
    metrics = simulate(steps=120, seed=42+rank)

    if is_dist():
        # gather metrics to rank0 via all_reduce average for demo
        throt_t = torch.tensor([metrics["throttle_rate"]], dtype=torch.float32)
        dist.all_reduce(throt_t, op=dist.ReduceOp.SUM)
        throt_avg = throt_t.item()/world
        if rank==0:
            metrics["throttle_rate_dist_avg"] = throt_avg
        # barrier demo for sync eval pattern
        if rank==0:
            print(f"[rank0] gloo {world}-rank mechanical→GPU simulation done {time.time()-t0:.3f}s")
        dist.barrier()

    if rank==0 or not is_dist():
        print("=== Day 11 Paper2 mech→GPU thermal (CPU true numbers) ===")
        for k in ["rmse_Q_pred_kW","avg_P_mech_kW","p_mech_std_kW","throttle_rate","Tj_max_C","Tj_avg_C","steps"]:
            print(f"{k}: {metrics[k]}")
        # json for NOTES auto-parse
        print(json.dumps({k: metrics[k] for k in ["rmse_Q_pred_kW","avg_P_mech_kW","p_mech_std_kW","throttle_rate","Tj_max_C","Tj_avg_C"]}, ensure_ascii=False))

        # 待H100 NCCL 提示
        if not torch.cuda.is_available():
            print("待H100验证: torch.cuda.max_memory_allocated / NCCL real thermal Tj  + power smoothing ilp")
            print("CPU 模拟已跑通，GPU 真机需补 R_jh 温度+风扇立方律实测 + 节流真实延迟")

if __name__ == "__main__":
    main()
