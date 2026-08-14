# Paper 模板

## 元信息
- Title: LIMR: Less is More for RL Scaling
- Authors / Org: Xuefeng Li, Haoyang Zou, Pengfei Liu - SJTU, SII, GAIR
- Link / arXiv: https://arxiv.org/abs/2502.11886
- Date read: 2026-08-13
- Tags: [rl-data, coding-data, data-selection, less, influence, reasoning]

## 一句话总结
RL 阶段不用堆 8k 题，用 Learning Impact Measurement 按学习轨迹挑 1,389 题硬题，7B RL 效果反超全量，证明 RL 里少即是多就是影响力对齐。

## 核心
1.  **Motivation**: o1, R1 都说 RL 能提推理，但数据要多少说不清。SFT 少量好用的 LIMO/s1 在 7B 上拉胯，说明 SFT 的少即是多不能直接搬到 RL。
2.  **Data Pipeline**: 从 base 模型直接起 RL -> 跑全量 8,523 题轨迹 -> 用 LIM 评估每题和学习轨迹的对齐度 -> 挑 1,389 -> 再训 RL 对比。LIM 是自动化度量，算每题对策略提升的贡献。
3.  **Key Tricks**: 
    - 不蒸馏直接从 base RL：避免老师天花板
    - LIM 按轨迹对齐度选，不是按难度主观：1,389 vs 8,523 打赢【8617116200514826664†L20-L23】
    - 小样本 RL 泛化更好：AIME24 +16.7%，MATH500 超 LIMO 13% / s1 22.2%【8617116200514826664†L23-L26】
4.  **Results**: 1,389 题 LIMR-7B 在 AIME/MATH 上持平或超过 8,523 全量【8617116200514826664†L21-L24】，开源了 LIM 实现+数据+模型【8617116200514826664†L26-L29】。

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：
  - 把你 coding 50k 的 RL 阶段，用 LIM 思路：先跑一遍 50k 的 rollout，用 pass/fail 轨迹的相关度来重排，留 20% 高对齐的再训第二轮 RL
  - 把 LESS 的梯度库换成 LIM 的轨迹对齐度，ADA 里的 7B base 直接 RL 的思路可试
- Infra 视角：RL 洗数据比 SFT 洗便宜太多，1.3k vs 8k 意味着 PPO 采样开销降 6 倍，值得做

## 和之前工作的关系
是 LESS 的 RL 版。LESS 说 SFT 5% 打赢全量，LIMR 说 RL 16% 打赢全量，都是少即是多。但 LESS 用梯度相似度，LIMR 用学习轨迹对齐度，解决了你之前问的 LESS 到 RL 的 gap。补了 Day 4 LESS 只管 SFT、Day 8-10 pretrain 太重的短板，回到 post-train coding 数据的本质。

## 疑问 / 下一步
- LIM 具体怎么算相关度？和 DataInf 的闭式影响比，哪个更便宜可做成在线版？

## 原文金句 (1-2句)
> a strategically selected subset of just 1,389 samples can outperform the full 8,523-sample dataset【8617116200514826664†L20-L23】

> precise sample selection, rather than data scale, may be the key【8617116200514826664†L25-L27】
