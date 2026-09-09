# r2-Day13 — Attention 变种：问答归档

用户在 ai 课程聊天中提出的问题与总结。问题原文保留，答案按聊天记录整理；
英文术语（MHA、MQA、GQA、MLA、KV cache、latent、RoPE、absorption、head、
up-projection、down-projection、uptraining）保留。

## 2026-09-08（周二）

以下 5 问均为 2026-09-08 晚（PDT）提出，紧接 r2-Day13 课程推送之后，
全部围绕 MHA→MQA/GQA/MLA 的数学原理。

### Q1：MHA→MQA/GQA/MLA 大概原理介绍一下 可能的话用数学语言

**A：** 先统一符号： $x_t \in \mathbb{R}^{d_{model}}$ 为第 $t$ 个 token 的层输入；
$h_q$ = query head 数， $h_{kv}$ = key/value head 数， $d_h$ = 每 head 维数，
$d_{model} = h_q d_h$ ； $L$ = 层数， $S$ = 序列长度， $B$ = 每元素字节数。

**MHA 基线**：每个 head $i$ 独立投影，
$$q_t^{(i)} = x_t W_Q^{(i)},\quad k_t^{(i)} = x_t W_K^{(i)},\quad v_t^{(i)} = x_t W_V^{(i)} \in \mathbb{R}^{d_h}$$
即 $h_{kv} = h_q$ 。decode 时每层 cache 存 $2 \cdot S \cdot h_{kv} \cdot d_h$
个元素——后面三招全都是在砍这个数。

**MQA**（Shazeer 2019）：把 $h_{kv}$ 直接砍到 1，所有 query head 共用同一组 K/V：
$$\text{head } i \text{ 的输出} = \mathrm{softmax}\!\left(q_t^{(i)} k_{1:t}^T / \sqrt{d_h}\right) v_{1:t}$$
每 token 每层 cache 从 $2h_q d_h$ 降到 $2d_h$ ，省 $h_q$ 倍。牺牲表达能力，
换取 decode 的 KV 字节和 HBM 流量成倍下降。

**GQA**（Ainslie et al. 2023）：MQA 与 MHA 之间的连续插值。取组数 $G$ ，
$h_{kv} = G$ ，每 $h_q/G$ 个 query head 共用一组 KV。 $G = h_q$ 时恒等于 MHA，
$G = 1$ 时恒等于 MQA（Day13 代码数值验证过）。落地的关键是 uptraining：
把 MHA checkpoint 的 K/V head 按组取平均转成 GQA，再续训约 5% 预训练算量。

**MLA**（DeepSeek-V2）：不做 head 级共享，做**低秩压缩**。
$$c_t^{KV} = h_t W^{DKV} \in \mathbb{R}^{d_c},\quad d_c \ll h_q d_h$$
cache 里只存 $c_t^{KV}$ 。吸收恒等式：
$$q^T k = q^T W^{UK} c = (W^{UK T} q)^T c = \tilde{q}^T c$$
decode 时把 $W^{UK}$ 吸进 query，直接对 $c^{KV}$ 做 attention，
全尺寸 K 永远不 materialize。RoPE 必须解耦：RoPE 是随位置 $t$ 变化的旋转
$R_t$ ，矩阵乘法不可交换，所以单独留一条小维度 RoPE 通道
$k_t^R = \mathrm{RoPE}(h_t W^{KR})$ （V3 取 $d_r=64$ ），所有 head 共用。

四者每 token 每层 cache（元素数）：
$$2h_q d_h \;\;\xrightarrow{MQA}\;\; 2d_h \;\;\xrightarrow{GQA}\;\; 2Gd_h \;\;\xrightarrow{MLA}\;\; d_c + d_r$$
V3 维度下 MHA $2\cdot128\cdot128 = 32768$ ，MLA $512+64 = 576$ ，约 $56.9\times$ 。

共同逻辑：decode 是 memory-bound，省字节 = 省时间；prefill 是 compute-bound，
$QK^T$ 的 FLOPs $= 2S^2 h_q d_h$ 只和 $h_q$ 有关——这三招都没动 $h_q$ ，
所以 **prefill 一点不赚**。MLA 的代价是复杂度税：RoPE 解耦 + 吸收改变 TP 切分面
（Day15 的议题）。

**同轮补充（吸收恒等式的结合律视角）：** 真正的把戏只有一句话——
矩阵乘法结合律，换个括号位置：
$$q^T \underbrace{(W^{UK} c_j)}_{\text{先展开 } S \text{ 个 key}} = \underbrace{(q^T W^{UK})}_{\text{先变换 1 个 query}} c_j = \tilde{q}^T c_j$$
左边把 $S$ 个 cached 隐向量全展开成 $d_h$ 维的 key（ $S$ 次 $d_c \times d_h$ 乘法），
右边只变换**一个** query（1 次），然后 $S$ 次点积直接在隐空间里做——
省的就是这 $(S-1)$ 次。注意 $q$ 不是被"压缩"了，而是从 128 维**升**到 512 维；
"所有计算都是 512 维"只对随 $S$ 线性增长的那部分成立。定量（ $d_c=512, d_h=128, S=128k$ ）：naive 展开 key 约 $2S d_c d_h \approx 1.7\times 10^{10}$
FLOPs，吸收后约 $2S d_c \approx 1.3\times 10^8$ ——差的正好是 $d_h = 128$ 倍。
论文更彻底：把 $W^{UK T} W^{UQ}$ 离线合并成一个矩阵存权重里，decode 时
全尺寸 K/V 在数学上就不存在了。

### Q2：MLA中 h_t W^DKV是关键 W^DKV 是一个压缩矩阵 h_t是啥？是K or V吗？

**A：** 都不是。 $h_t$ 是 K 和 V 的"共同祖先"——attention 层的**输入**
hidden state（上一层传下来的那个向量），K、V 都是从它投影出来的。

管线：
$$h_t \;\xrightarrow{\;W^{DKV}\;}\; c_t^{KV} \;\xrightarrow{\substack{W^{UK} \\ W^{UV}}}\; \begin{cases} k_t \\ v_t \end{cases}$$

标准 MHA 里 $k_t = W_K h_t$ 、 $v_t = W_V h_t$ ，是 $h_t$ 的两个独立胖投影。
MLA 把这两个胖投影做低秩分解：先用共享的下投影 $W^{DKV}$ 把 $h_t$
压成隐向量 $c_t^{KV}$ ，再用两个瘦的上投影 $W^{UK}$ 、 $W^{UV}$ 分别展开回 K、V。
**cache 里存的就是中间这个** $c_t^{KV}$ ，K、V 需要时才现场重建
（decode 时连重建都省了，直接吸收进 query）。

V3 维度： $h_t \in \mathbb{R}^{7168} \xrightarrow{W^{DKV}} c_t^{KV} \in \mathbb{R}^{512} \xrightarrow{W^{UK}} k_t^{(i)} \in \mathbb{R}^{128}$ （每 head），压缩比 $7168/512 = 14$ 。"Latent"（隐）的含义： $c_t^{KV}$ 同时是 K 和 V 的压缩表示——
K 和 V 不再各自独立存一份，而是共享同一个隐祖先。这也是 MLA 比 MQA/GQA 更狠的
原因：MQA/GQA 只是在 head 数上做共享（ $h_{kv}$ 从 $h_q$ 降到 $G$ ），
MLA 直接把 $h_q \cdot d_h = 16384$ 维的 K/V 联合空间压到 512 维。

顺带：query 侧也有压缩（ $c_t^{\prime Q} = W^{DQ} h_t$ ），但 query 不用 cache
（decode 每步只需要当前 token 的 q），所以 query 压缩省的是计算不是显存——
cache 账本只算 KV 这边。

### Q3：合理 W^DKV 怎么算出来的呢？这得是对所有h_t做的分解？

**A：** 不是。 $W^{DKV}$ **不是**对 $h_t$ 做 SVD/PCA 分解算出来的，
它就是个普通的可学习参数矩阵，随机初始化、跟着整个模型一起 backprop
端到端学出来。 $W^{UK}$ 、 $W^{UV}$ 、 $W^{DQ}$ 同理。训练流程里没有任何
"先收集 $h_t$ 再分解"的步骤。

它"看起来像分解"的原因是**秩约束**：第 $i$ 个 head 的有效 key 映射是
$$M_i = W^{UK}_i W^{DKV} \in \mathbb{R}^{d_h \times d_{model}},\quad \mathrm{rank}(M_i) \le d_c$$
所有 head 共享同一个右因子 $W^{DKV}$ ——所有 head 的 key 映射被强制经过
同一个 $d_c$ 维子空间。这是架构上的归纳偏置，不是对训好矩阵的事后分解。
梯度流过 $W^{UK}_i W^{DKV}$ 时，LM loss 会自己把"对 attention 最有用的信息"
塞进这 $d_c$ 维里。有点像 autoencoder 的 bottleneck，但没有重构 loss——
塑造隐空间的是语言建模目标，而不是 $h_t$ 的方差（所以它**不是**主成分）。

对比两个"真分解"的例子：
- **GQA uptraining**：拿训好的 MHA 的 $W_K$ 、 $W_V$ ，按组取平均初始化 GQA，
  再续训——那才是对已有模型做转换。
- **SVD 权重压缩**（如 LoRD）：真对训好的矩阵做 SVD 截断。

MLA 两边都不沾，DeepSeek-V2/V3 是从 pretrain 第一天就按这个架构训的。
代价也在这：**没法把一个现成的 MHA 模型"转成" MLA** ，低秩结构必须在
pretrain 时就 baked in——这也是 MLA 落地门槛比 GQA 高的根本原因。

### Q4：这里面MLA的部分的维度解释一下

**A：** 先修一个符号约定问题：MHA/GQA 的定义用的是**行向量**
（ $x_t W_Q^{(i)}$ ），但 MLA 的吸收恒等式和手算用的是**列向量**
（ $q^T W^{UK} c$ ）。两种都对，但混着写维度就看似对不上了。
统一成行向量约定（跟 MHA 定义和 PyTorch 一致），把 V3 的真实数字代进去：

- $x_t \in \mathbb{R}^{1 \times 7168}$ ：层输入，行向量
- $W^{DKV} \in \mathbb{R}^{7168 \times 512}$ ：**全层唯一**的下投影， $h_q = 128$ 个 head 共用
- $c_t^{KV} = x_t W^{DKV} \in \mathbb{R}^{1 \times 512}$ ：KV latent，**cache 里存的就是它**
- $W^{UK}_i \in \mathbb{R}^{512 \times 128}$ ：第 $i$ 个 head 的 key 上投影，共 128 份
- $W^{UV}_i \in \mathbb{R}^{512 \times 128}$ ：第 $i$ 个 head 的 value 上投影，共 128 份
- $W^{KR} \in \mathbb{R}^{7168 \times 64}$ ：RoPE 通道投影，**全 head 共享一份**；
  $k_t^R = \mathrm{RoPE}(x_t W^{KR}) \in \mathbb{R}^{1 \times 64}$ ，cache 里也要存它

每 token 每层 cache $= 512 + 64 = 576$ 个元素，fp16 就是 1152 bytes。
压缩的真相是**跨 head 共享**：MHA 里 128 个 head 各有各的 128 维 key
（共 16384 个有效维度），MLA 把它们全锁进同一个 512 维子空间。
注意单个 head 的 $d_h = 128 < d_c = 512$ ，压的不是单个 head，是 head 之间的冗余。
等价 key 矩阵 $W^{DKV} W^{UK}_i \in \mathbb{R}^{7168 \times 128}$ 的秩不超过 512，
这就是 rank bottleneck。

行向量约定下的吸收恒等式：
$$s_{tj}^{(i)} = q_t^{(i)} (k_j^{(i)})^T = q_t^{(i)} (W^{UK}_i)^T (c_j^{KV})^T = \tilde{q}_t^{(i)} (c_j^{KV})^T$$
其中 $\tilde{q}_t^{(i)} = q_t^{(i)} (W^{UK}_i)^T \in \mathbb{R}^{1 \times 512}$ 。
维度链： $(1 \times 128)(128 \times 512) = 1 \times 512$ ，
再和 $(c_j^{KV})^T \in \mathbb{R}^{512 \times 1}$ 点积得标量。

同一组数字按行向量重做手算（ $q = [1,2]$ ， $c = [5,6]$ ，
$W^{UK} = \begin{bmatrix}1&2\\3&4\end{bmatrix}$ ）：
- 直接算： $k = cW^{UK} = [23, 34]$ ， $qk^T = 23 + 68 = 91$
- 吸收算： $\tilde{q} = q(W^{UK})^T = [5, 11]$ ， $\tilde{q}c^T = 25 + 66 = 91$ ✓

（列向量版本算出来是 95；数字不同只是因为约定换了，关键是两边依然相等。）

value 侧同理： $o_t^{(i)} = (a_t^{(i)} C) W^{UV}_i$ ，其中
$C \in \mathbb{R}^{S \times 512}$ 是全部缓存的 latent 矩阵，attention 权重复接乘在
$C$ 上，全尺寸 $V$ 同样不 materialize。每 head 还可以把 $W^{UV}_i$ 折进
output projection 预合并： $\hat{W}^O_i = W^{UV}_i W^O_i \in \mathbb{R}^{512 \times 7168}$ 。

V2 里 query 侧的压缩： $c_t^Q = x_t W^{DQ}$ （ $W^{DQ} \in \mathbb{R}^{7168 \times 1536}$ ），
$q_t^{(i)} = c_t^Q W^{UQ}_i$ ，所以
$\tilde{q}_t^{(i)} = c_t^Q (W^{UQ}_i (W^{UK}_i)^T)$ 可以预先合并成一个
$1536 \times 512$ 的矩阵——decode 时 query 直接从 1536 维 latent 一步到 512 维。

### Q5：q_t^(i) 这里这是怎么来的？

**A：** 它来自 query 侧自己的压缩路径，和 KV 那条是对称的：
$$x_t \xrightarrow{W^{DQ}} c_t^Q \xrightarrow{W^{UQ}_i} q_t^{(i)}$$

- $W^{DQ} \in \mathbb{R}^{7168 \times 1536}$ ：query 下投影，**全层共享**，所有 head 共用
- $c_t^Q = x_t W^{DQ} \in \mathbb{R}^{1 \times 1536}$ ：query latent，
  **不进 cache**——decode 时每次只需要当前这一个 token 的 query，算完即用即扔
- $W^{UQ}_i \in \mathbb{R}^{1536 \times 128}$ ：第 $i$ 个 head 的 query 上投影，每 head 独立
- $q_t^{(i)} = c_t^Q W^{UQ}_i \in \mathbb{R}^{1 \times 128}$ ：吸收恒等式里的 $q_t^{(i)}$ ，
  是第 $i$ 个 head 的 **content 部分**（不带 RoPE）

上标 `(i)` 是 head 编号，下标 $t$ 是 token 位置，跟 MHA 定义里的记号完全一致。

每个 head 的完整 query 其实是两段拼起来：content 部分 $q_t^{(i)} \in \mathbb{R}^{128}$ ，
加上 RoPE 部分 $q_t^R = \mathrm{RoPE}(x_t W^{QR}) \in \mathbb{R}^{64}$ （全 head 共用一份）。
key 侧同理 $[k_j^{(i)},\, k_j^R]$ 。attention score 是两段点积之和：
content 点积那段才能做吸收，RoPE 那段不行。

为什么 query 也要压缩？跟省显存没关系（query 不缓存）。纯粹是省参数和训练时的
activation： $W^{DQ}$ 全 head 共用一份 $7168 \times 1536$ ，而不是 128 个 head
各自搞 $7168 \times 128$ 。顺带的好处是吸收可以直接做到 query latent 上：
$\tilde{q}_t^{(i)} = c_t^Q (W^{UQ}_i (W^{UK}_i)^T)$ ，decode 时一步从 1536 维到 512 维，
中间 128 维的 $q_t^{(i)}$ 同样不 materialize。
