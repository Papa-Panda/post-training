# NOTES - r2-Day01 Transformer

第二轮 Day1 地基，粗略理解即可

- Attention O(N²) 是后续 FlashAttention/量化/解耦的动机
- FFN 参数大头，GQA/MLA 省KV不省FFN
- RoPE 相对位置，支持外推
- Pre-Norm 训练更稳，梯度直通

待H100：无，此日纯理论

---
## 补充：Forward FLOPs per trajectory 2·N·D + 2·L·D²·d_model 来源（2026-08-26 补）

**1. 2·N·D 线性层部分**
- N 参数量，D trajectory token数
- y=xW，x D×d_in，W d_in×d_out，FLOPs 2·D·d_in·d_out，2=1乘+1加
- 累加所有 QKV/out/FFN 线性层 = 总线性参数 N，每token用一次 → 2ND
- 即 Chinchilla forward每token 2N

**2. 2·L·D²·d_model Attention二次项**
- L层，每层两次激活×激活（不含参，不在N里）
- QKᵀ：D×d_model × d_model×D → D×D，2·D²·d_model
- Attn·V：D×D × D×d_model → 2·D²·d_model
- 合计 4·L·D²·d_model，乘加算1 FLOP时简写 2·L·D²·d_model
- D²特性：trajectory一长就爆，长rollout RL瓶颈

总前向 = 线性主导 + 注意力二次项，backward≈2×forward。

---
## Paper 来源（2026-08-27 补）

- Attention Is All You Need (Vaswani et al., NeurIPS 2017): https://arxiv.org/abs/1706.03762 - 提出 QKᵀ/√d_k → softmax → PV，Multi-Head，Positional Encoding，Encoder-Decoder
- The Illustrated Transformer (Jay Alammar): https://jalammar.github.io/illustrated-transformer/ - 图解 Decoder Block 维度流 (B,S,D) 前向
- RoPE (Su et al., 2021): https://arxiv.org/abs/2104.09864 - 相对位置旋转编码，支持外推
- Pre-Norm vs Post-Norm (Xiong et al., 2020): https://arxiv.org/abs/2002.04745 - 为何大模型用Pre-Norm更稳
- Chinchilla Scaling (Hoffmann et al., 2022): https://arxiv.org/abs/2203.15556 - forward 2ND，6ND total，MFU定义来源
- LLaMA Architecture (Touvron et al., 2023): https://arxiv.org/abs/2302.13971 - SwiGLU FFN 11008，RMSNorm，RoPE实例
