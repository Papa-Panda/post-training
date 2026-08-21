# NOTES - r2-Day01 Transformer

第二轮 Day1 地基，粗略理解即可

- Attention O(N²) 是后续 FlashAttention/量化/解耦的动机
- FFN 参数大头，GQA/MLA 省KV不省FFN
- RoPE 相对位置，支持外推
- Pre-Norm 训练更稳，梯度直通

待H100：无，此日纯理论
