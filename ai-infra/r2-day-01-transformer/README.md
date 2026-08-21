# r2-Day01 - Transformer 架构（第二轮 Day1）

> r2 表示第二轮纯 AI Infra，重走草帽路飞四层。
> Connection: r2-Day01 地基起点，后续所有 CUDA/分布式/推理都围着它转。

## 目标（第零层 前置）

- Self-Attention Q K V 含义与计算 QK^T → scale → softmax → PV，复杂度 O(N²)，是 FlashAttention 优化前提
- FFN 两层线性+激活，模型参数大头
- RoPE 位置编码为何需要额外位置信息
- LayerNorm Pre-Norm vs Post-Norm，为何大模型普遍用 Pre-Norm
- 完整前向：token embedding → Masked Self-Attention → Add&Norm → FFN → Add&Norm，标注每步输入输出维度 (B,S,D)

## 小任务 30-60min（知识面覆盖）

1. 跟着 The Illustrated Transformer 手画一遍 Decoder Block，标 (B,S,D)
2. 给定 7B 配置 hidden=4096 heads=32 layers=32 vocab=32000，手算总参 ≈ Attention+FFN+Embedding 误差<20%
3. 不看资料说出 Q K V 如何线性投影得到

## 检验（够用即可）

- 白板默写 Decoder Block 无错
- 维度推导 (B,S,H) x (H,V) → (B,S,V) 立刻反应
- 能说清为何需要位置编码

## 资源

- Attention Is All You Need
- Jay Alammar The Illustrated Transformer
- 琳琅阿木 图文详解LLM inference

## 产出

- 手绘图拍照存 assets/（可选）
- 1页笔记：QK^T scale因子 sqrt(dk) 为何
