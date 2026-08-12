# Day 6 - 2026-08-07 - Paper1 autoscaling → RL Infra

**Goal:** 把你 Paper1 自动扩缩容的 5 个核心方法，翻译成 RL Infra 的语言，做成面试能讲的 bridge。

## 5 条映射 (Paper1 → RL Infra)

1. **burst 预测 → rollout 到达预测**
   - Paper1: 用 5min 级时序预测突发流量，提前 10min 扩容，避免 SLO 跌。
   - RL: rollout worker 到达是泊松+突发 (长思维链 500→5000 tokens)。同样用 EWMA + trend 预扩 vLLM 集群，避免训练集群空转等 rollout。
   - 算账: 训练集群 $3.2/GPU-hour，空转 15% = 每月浪费 x。

2. **SLO 定义 → GPU 利用率下限 SLO**
   - Paper1: 定义 P95 延迟 <200ms, 成功率 >99.9%。
   - RL: 定义小集群 3 条 SLO：作业成功率 >95%，排队时长 <2min，GPU 利用率 >60% 下限 (不是上限)。低了说明调度烂，探针就是 $/有用 rollout。
   - 复用你写过那套 SLO 故障库。

3. **成本模型 → $/有用 rollout 成本模型**
   - Paper1: 成本 = 机型成本 + 功率 * PUE，优化目标是 $/QPS。
   - RL: 成本 = 训练 $ + vLLM $ + 失败重试 $，除以有用 rollout 数 (pass filter)。PUE 建模思路直接套，目标 $/有用 rollout 降 20% = 下一个 $200M 故事雏形。
   - 类比你 Paper3 PUE modeling，COST = new PUE。

4. **抗抖动 / 冷却 → reward 抖动处理**
   - Paper1: 加冷却窗口 10min + 滞后阈值，避免抖动反复扩缩容。
   - RL: reward 噪声大，GRPO 组内相对 + EMA baseline 就是冷却，用 OAS 校准法压方差，别让抖动触发无意义重训。
   - 你金融 OAS 压噪思路这里直接用。

5. **nowcasting 短时预测 → eval 延迟预测**
   - Paper1: 用 nowcasting 预测 5-10min 后负载。
   - RL: 用同样方法预测 eval 排队延迟，eval 是瓶颈，80% 墙钟耗在 rollout+eval。提前预测，把 coding eval 的 3 个 flaky 点加重试/分级，P95 eval <5min。
   - ML for Infra 里的 nowcasting 就是 RL 的 eval 预测器。

## 面试一句
> 我过去用时序预测 + SLO + 成本建模省 $200M，这套方法平移到 RL，就是把 burst 预测换成 rollout 到达，把 $/QPS 换成 $/有用 rollout，GPU 利用率从上限制转下限 SLO。

## 待H100 验证清单
- eval 用法留空，待小 eval 上机再补数。
