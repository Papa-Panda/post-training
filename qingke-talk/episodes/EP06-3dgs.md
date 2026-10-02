# EP6 — 实时渲染 3DGS 中的反走样及逆渲染应用

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP06-3dgs.html

> "solving the integral of Gaussian signals within the pixel window area as intensity responses is crucial for both anti-aliasing and capturing details." —— Analytic-Splatting 论文 §1，点明本期反走样主线的问题定性

## 元信息

- 期号：6（官网期与 B站合集期号一致，Talk06）
- 标题：实时渲染 3DGS 中的反走样及逆渲染应用
- BV：BV1NntyzRExN（有 B站视频，但视频无自有字幕轨：浏览器内约 25 次抓取只返回空字幕列表或串台字幕，故本纪要走论文还原路线）
- 时长：01:06:03
- 直播时间：2024-05-21（周二）19:00–20:00（官网预告）
- 提炼日期：2026-10-02
- 分享嘉宾：梁智灏（华南理工大学几何感知与智能实验室博士，导师贾奎；Analytic-Splatting 工作于腾讯 AI Lab 实习期间完成；官网预告嘉宾介绍 + 论文作者页）
- 相关论文：
  - Zhihao Liang, Qi Zhang, Wenbo Hu, Ying Feng, Lei Zhu, Kui Jia, *Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic Integration*，https://arxiv.org/abs/2403.11056（v2, 2024-04-03）
  - Zhihao Liang, Qi Zhang, Ying Feng, Ying Shan, Kui Jia, *GS-IR: 3D Gaussian Splatting for Inverse Rendering*，https://arxiv.org/abs/2311.16473（v3, 2024-03-28）
- 相关代码：Analytic-Splatting https://github.com/lzhnb/Analytic-Splatting ；GS-IR https://github.com/lzhnb/GS-IR
- 官网预告：https://qingkeai.online/blog/t9m9Uukj
- B站链接：https://www.bilibili.com/video/BV1NntyzRExN/

> ⚠️ 提炼方式说明：本期 B站视频无自有字幕轨（字幕接口只回空列表或他视频的串台字幕，已多轮抓取确认），无法逐字还原。本纪要根据该期对应的公开材料还原——官网预告（含讲者与四点讲授提纲）+ 讲者两篇论文原文。提纲第 3 节「3DGS 反走样」以 Analytic-Splatting（arXiv:2403.11056）为主线详写；第 2 节「逆渲染」以 GS-IR（arXiv:2311.16473）为据压缩呈现；第 1 节（从 NeRF 到 3DGS 的方法概述）与第 4 节（落地难点及未来方向探讨）没有独立的书面材料，只交代脉络、不虚构讲者具体表述。特别说明取舍：**Mip-Splatting（Zehao Yu 等）不是讲者团队的工作**，在本期中只是 Analytic-Splatting 的对比基线，纪要按这个定位处理，不与主线混写。现场 Q&A 与讲者口头发挥未覆盖。

## 一句话总结

针对 3DGS 在变分辨率/变焦渲染时出现模糊与锯齿的问题，讲者主线工作 Analytic-Splatting 把像素响应从「中心一点采样」改成「像素窗口内高斯信号的解析积分」：用条件 logistic 函数近似高斯 CDF、以 CDF 差分得到窗口积分，再对投影协方差做特征对角化把二维积分拆成两个一维积分乘积，同时放进颜色与透射率计算——抗走样与保细节兼得，在多尺度协议下全面超过 3DGS 与 Mip-Splatting；同场的另一条线 GS-IR 则把 3DGS 第一次引入逆渲染，用深度推导的法线正则与烘焙遮挡体积完成材质/光照分解，速度比 NeRF 式逆渲染快一个量级。

## 核心

### 背景/问题：3DGS 为什么一变倍率就露馅

按官网预告，讲授分四段：（1）从 NeRF 到 3DGS 的三维重建渲染方法概述；（2）基于物理的逆渲染框架 GS-IR；（3）实现 3DGS 反走样的 Analytic-Splatting；（4）3DGS 落地难点及未来方向探讨。

走样的根源在 3DGS 的像素着色方式（Analytic-Splatting 论文 §3）：3DGS 把每个像素当作一个孤立的点，只取投影 2D 高斯在**像素中心**处的值作为该像素的强度响应。当相机距离、焦距或输出分辨率改变时，像素在世界空间覆盖的足迹（footprint）随之改变，但中心采样对这种变化不敏感——采样带宽受限，混叠不可避免：拉近时出现细长高斯与锯齿，拉远时发糊（论文 Abstract、§1）。论文给出的两个候选修法都不理想：超采样（论文中基线 3DGS-SS：先以 2 倍分辨率渲染再平均池化，§5.2）计算代价高；同期最强的 Mip-Splatting 用预滤波（3D 平滑 + 2D Mip 滤波，本质是把窗口当成一个低通高斯核与信号做卷积）压制高频，代价是过平滑、细节丢失（论文 §1、§2 的定位）。讲者的判断是：**正解既不是采更多点、也不是先把高频滤掉，而是老老实实把窗口内的积分算出来**。

### 方法/设计：Analytic-Splatting 如何把积分闭式化

从一维情形出发（论文 §4.1）：高斯信号 $g(x)$ 在窗口 $[x_{1}, x_{2}]$ 内的积分等于其累积分布函数（CDF）之差 $G(x_{2}) - G(x_{1})$ ；但高斯 CDF 就是误差函数 erf，没有闭式解。论文用一个「条件 logistic 函数」 $S(x)$ 解析近似标准高斯的 CDF（Eq. 9）：

$$S(x)=\frac{1}{1+\exp\left(-1.6 \cdot x-0.07 \cdot x^{3}\right)}$$

对任意标准差 $\sigma$ ，把自变量按 $1/\sigma$ 缩放即可适配；于是窗口宽为 1 的积分直接写成 $S(u+\frac{1}{2}) - S(u-\frac{1}{2})$ （Eq. 11）。这个近似单调、关于 $(0, \frac{1}{2})$ 中心对称、可求导，能直接进入可微渲染管线反向传播。

二维情形的障碍是投影 2D 高斯的协方差含交叉相关项，二重积分不可直接解（论文 §4.2）：论文对 $\hat{\Sigma}$ 做特征分解，把像素积分域旋转到两个特征向量构成的坐标系（这一步引入可控的近似，见下文误差分析），二维高斯随之拆成两个独立的一维高斯，窗口积分变成两个一维 CDF 差分的乘积：

$$\mathcal{I}_{g}^{\text{2D}}(\bm{u}) \approx 2\pi \sigma_{1} \sigma_{2} \left[ S_{\sigma_{1}}(\tilde{u}_{x}+\frac{1}{2})-S_{\sigma_{1}}(\tilde{u}_{x}-\frac{1}{2}) \right] \left[ S_{\sigma_{2}}(\tilde{u}_{y}+\frac{1}{2})-S_{\sigma_{2}}(\tilde{u}_{y}-\frac{1}{2}) \right]$$

实现上只动一处：把这个积分响应 $\mathcal{I}_{g}^{\text{2D}}$ 替换掉原 3DGS 着色公式里的中心采样值，同时进入颜色累积与透射率 $T_{i}$ 的计算（Eq. 15），渲染结果便天然随像素足迹变化。工程基底是原版 3DGS + 自定义 CUDA shading 模块，训练参数、训练日程与损失函数与 3DGS 完全相同（论文 §5.2）——即改动集中在 shading 一个环节，不动表示与优化。与 Mip-Splatting 的本质区别也在这里：**它不压制高频成分，而是按像素足迹做真实的面积积分**，所以保细节（论文 §1）。

### 实验/实战：Analytic-Splatting（多尺度训练与测试，MTMT）

- **近似误差先行验证（§5.1）**：在 3DGS 训练实际维持的 $\sigma \in [0.3, 6.6]$ 范围内，条件 logistic 对 CDF 与窗口积分的近似误差以 $10^{-4}$ 量级呈现，显著优于其他近似方案； $\sigma$ 越小（即信号越高频）优势越明显，这正是它保细节的数学来源。把积分域旋转 $0^{\circ}$ 到 $45^{\circ}$ 会略微增大误差，但仍优于其他方案（Fig. 4、Fig. 5）。
- **多尺度 Blender 合成集（Table 1）**：平均 PSNR 35.03，对比 3DGS 的 29.77 与 Mip-Splatting 的 34.56；平均 LPIPS 0.018（Mip-Splatting 0.019、3DGS 0.040）；在走样最严重的 1/8 分辨率下 PSNR 36.00 对 3DGS 的 27.98，低分辨率端差距最大。
- **多尺度 Mip-NeRF 360（Table 2）**：平均 PSNR 29.51，优于 3DGS (27.63) 与 Mip-Splatting (29.12)，是可实时渲染方法中最好的；仅低于 Zip-NeRF (30.58) ——但 Zip-NeRF 非实时且渲染阶段用超采样（论文 §5.2 正文）。平均 LPIPS 0.123，对 Mip-Splatting 的 0.134。
- **2 倍超分辨率设置（Table 3，Mip-NeRF 360）**：平均 PSNR 26.90，依次高于 3DGS-SS (26.68)、Mip-Splatting (26.46)、3DGS (25.95)，说明积分响应在放大渲染时同样保细节，而预滤波路线在该设置下垫底于超采样，印证了过平滑的代价。

### 逆渲染支线：GS-IR（提纲第 2 节，压缩呈现）

GS-IR 解决的是另一个问题：给定未知光照下拍摄的多视图图像，把场景分解为几何（法线）、材质（反照率/金属度/粗糙度，Cook-Torrance BRDF 模型）与环境光照，进而支持重光照与材质编辑；论文自述这是把 3DGS 引入逆渲染的第一件工作（§1）。把 3DGS 搬进逆渲染有两个天然障碍，各对应一个设计：

- **法线不可信**：3DGS 的自适应密度控制使几何松散、原生不出合理法线。GS-IR 用深度推导（depth-derivation）正则去优化每个高斯所存的法线，其中深度由线性插值策略生成；消融（Table 3）显示该策略的法线平均角度误差（MAE）为 4.948，远优于体积累积深度的 16.347，且避免了体积累积的漂浮问题与峰值选择的圆盘状走样（disc aliasing）。
- **遮挡算不了**：前向 splatting 不能像光线追踪那样追踪遮挡。GS-IR 借鉴实时渲染里的 Indirect Lighting Cache，把遮挡烘焙进一个遮挡体积（occlusion volume）做缓存，用它建模间接光照；消融（Table 4）显示去掉遮挡或去掉间接光都会拉低新视角合成 PSNR（完整模型 35.333，去遮挡 34.997，去间接光 35.186，TensoIR-Synthetic）。

效果定位（论文 §5.1 与 Table 2）：在 TensoIR-Synthetic 上，GS-IR 的新视角合成与反照率质量优于基线方法，重光照仅次于 TensoIR，法线重建略逊于 TensoIR，但平均训练时间相对基线方法约有 5 倍加速；在真实场景 Mip-NeRF 360 上，新视角 PSNR 为 25.381，低于专职新视角合成的 3DGS (27.21) ——这部分差距买来的是材质/光照分解能力；其训练约 45 分钟，对比 Mip-NeRF 360 的约 48 小时（Table 2）。论文同时自述局限：球谐只能表达低频，遮挡体积目前只建模间接光的漫反射项，镜面间接项未解，后续方向是屏幕空间全局光照（SSGI）（§6）。

### 结论/观点

- 对反走样：像素不是点而是一块面积，强度响应应该是窗口内的信号积分；高斯是连续函数，这个积分可以被解析近似到可用的精度——这是 Analytic-Splatting 相对超采样（贵）与预滤波（糊）的核心主张（论文 §1）。
- 对逆渲染：瓶颈不在「要不要光线追踪」，而在表示与遮挡建模；前向 splatting + 预烘焙缓存能把 NeRF 式逆渲染的速度拉进实用区间（GS-IR §1、§5.1）。
- 提纲第 1 节（方法概述串讲）与第 4 节（落地难点与未来方向）属于讲者的现场内容，没有书面出处，本纪要只记其存在、不复述具体观点（见疑问节）。

## 关键数字

| 指标 | 基线/对照 | 结果 | 来源 |
|---|---|---|---|
| 条件 logistic 对高斯 CDF 的近似误差 | 蒙特卡洛/卷积等其他近似方案 | 误差以 $10^{-4}$ 量级呈现且最小， $\sigma$ 越小优势越大 | 论文 §5.1、Fig. 4 |
| 多尺度 Blender 平均 PSNR | 3DGS 29.77 / Mip-Splatting 34.56 | Analytic-Splatting 35.03 | 论文 Table 1 |
| 同上 1/8 分辨率 PSNR | 3DGS 27.98 | Analytic-Splatting 36.00 | 论文 Table 1 |
| 多尺度 Blender 平均 LPIPS | 3DGS 0.040 / Mip-Splatting 0.019 | Analytic-Splatting 0.018 | 论文 Table 1 |
| 多尺度 Mip-NeRF 360 平均 PSNR | 3DGS 27.63 / Mip-Splatting 29.12 | Analytic-Splatting 29.51（实时方法中最好；Zip-NeRF 30.58 非实时） | 论文 Table 2、§5.2 |
| 2 倍超分 Mip-NeRF 360 平均 PSNR | 3DGS 25.95 / Mip-Splatting 26.46 / 3DGS-SS 26.68 | Analytic-Splatting 26.90 | 论文 Table 3 |
| GS-IR 法线平均角度误差（MAE） | 体积累积深度 16.347 | 线性插值深度推导 4.948 | GS-IR Table 3 |
| GS-IR 新视角 PSNR（TensoIR-Synthetic） | 去遮挡 34.997 / 去间接光 35.186 | 完整模型 35.333 | GS-IR Table 4 |
| GS-IR 新视角 PSNR（Mip-NeRF 360） | 3DGS 27.21（无分解能力） | GS-IR 25.381，训练约 45 分钟（Mip-NeRF 360 约 48 小时） | GS-IR Table 2 |
| GS-IR 训练速度 | NeRF 式逆渲染基线 | 平均训练时间约 5 倍加速 | GS-IR §5.1 |

## 可迁移

- **凡是「点采样代替面积积分」的地方都会在变倍率时露馅**：多分辨率评测、不同设备像素比的预览、缩略图金字塔、跨分辨率推理 serving。用 3DGS（或任何 splatting/点式表示）做重建与数据生成时，评测协议应单独设多尺度/多距离档位，单分辨率 PSNR 会系统性掩盖走样问题；Analytic-Splatting 的 MTMT 协议（同时在全分辨率到 1/8 分辨率上训练与测试）是现成模板。
- **「用闭式近似换数值采样」是一个可复用的数值技巧**：条件 logistic 近似 erf/CDF 这一套（解析、可导、误差 $10^{-4}$ 量级）不只用于高斯泼溅——任何软光栅化、概率渲染或需要对窗口做平滑积分的可微管线，都可以用类似的闭式 CDF 近似替代高倍率超采样，把 $O(N)$ 次采样换成 $O(1)$ 次函数求值。
- **「贵在线计算就烘焙成缓存」与「能解析就别采样」是同一课的两种形态**（Infra 视角）：GS-IR 把前向渲染算不了的遮挡预烘焙成体积缓存，Analytic-Splatting 把超采样换成解析积分；两者都是在为实时性让路时，先分清哪部分计算可以预计算、哪部分可以闭式化，再决定剩下的采样预算花在哪。放到 ML infra 里，这与「把逐样本的昂贵计算摊销为离线特征/缓存、用闭式近似替代蒙特卡洛估计」是同构的取舍。
- **超采样基线要算全账**：论文中 3DGS-SS（2 倍分辨率渲染 + 平均池化）的质量仍不及解析积分，但显存与算力开销随分辨率平方增长；凡是系统里用「提高采样率」兜底画质的地方，都值得先问一句有没有解析或预滤波的替代式。

## 疑问 / 下一步

- 现场内容不可还原：提纲第 1 节的方法史串讲与第 4 节「落地难点及未来方向探讨」是讲者的口头内容，尤其第 4 节（显存、速度、真实数据的坑）对做工程的人可能最有价值，但无书面材料，只能等字幕/录像可得时补录，现阶段不臆测。
- Analytic-Splatting 把积分域旋转到特征向量坐标系引入额外近似，误差随旋转角度（ $0^{\circ}$ 到 $45^{\circ}$ ）略增（论文 §5.1）；对强各向异性的细长高斯，这项误差在什么条件下会反噬渲染质量，论文给了误差曲线但没有逐角度的渲染质量消融。
- Analytic-Splatting 在已核对的表格里没有给出渲染帧率与显存的定量对比（只说明实时性继承自 3DGS 这一路方法、Zip-NeRF 非实时）；相对原版 3DGS 的实际速度折损，落地前需自行实测。
- GS-IR 的主对比表（Table 1，TensoIR-Synthetic 上与 TensoIR 等方法的逐项数值）本纪要只转述了论文 §5.1 的定性结论与 5 倍速度口径，逐方法数字未转写；若后续要做逆渲染方案横向选型，需回读 Table 1 原表。
- 本纪要只覆盖 2024 年 5 月 talk 时点的工作状态；3DGS 反走样与逆渲染在 2024 年之后的后续演进不在本期范围。

## 原文金句（1-2句）

> "3DGS treats each pixel as an isolated, single point rather than as an area, causing insensitivity to changes in the footprints of pixels." —— Analytic-Splatting 论文 Abstract，一句话点清走样的病根

> "Inspired by the 'Indirect Lighting Cache' used in real-time rendering, we attempt to bake the occlusion into volumes for caching." —— GS-IR 论文 §1，用实时渲染的工程传统解决前向 splatting 算不了遮挡的问题
