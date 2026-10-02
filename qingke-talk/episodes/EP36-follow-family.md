# EP36 — Follow Family：可控视频生成方法探索与应用
> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP36-follow-family.html

> 「ControlNet 本质上就是对 attention map 的影响，进而对整体的条件信息进行一个很好的引导。」——讲者复盘 Follow Your Handle 时的总结（字幕）

## 元信息

- 期号：36
- 标题：Follow Family：可控视频生成方法探索与应用
- BV：BV1ZeNZeYEfe
- 时长：00:35:29（工作介绍约 22 分钟 + 科研分享与 Q&A 约 13 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：马悦（香港科技大学博士生，2024 年清华大学毕业后入读；字幕自述导师为「陈立峰」老师、研究方向为图像与视频生成；曾在腾讯 AI Lab、混元团队实习，正式头衔以其主页为准）
- 相关论文（均为讲者自述的 Follow 系列工作，编号与版本以论文原文为准）：
  - Follow Your Pose（姿态可控的人物视频生成，两阶段训练 + 姿态数据集）
  - Follow Your Click（局部图像动画，TPAMI 2024；由 Dream2 一类 regional motion brush 启发）
  - Follow Your Handle（通过编辑首帧实现视频编辑，attention 特征注入）
  - Follow-Emoji（人脸版 Animate Anyone，表情 landmark 表示 + Emoji-Bench）
  - Follow Your Canvas（高分辨率视频 outpainting，TPAMI；区域 embedding + 多卡分块推理）
- 相关代码：Follow Your Pose 训练与推理代码已全部开放（讲授口径，GitHub 1.3K stars）
- B站链接：https://www.bilibili.com/video/BV1ZeNZeYEfe/
- 官网预告：青稞Talk EP36（官网链接未确认，略）
- 字幕原文存档：本地 `transcripts/EP36.txt`（850 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP36，已存档）清洗提炼，信息来源为 AI 字幕原文（经清洗，专名可能有识别误差：如 Follow-Emoji 在字幕中作「佛罗 emoji」、LoRA 作「LAURA」、评审分数「7769」为字幕口径且含义存疑等；引用具体数字前建议查论文与项目页核对）。

## 一句话总结

讲者以「Follow 家族」五件工作为主线，复盘了可控视频生成一条清晰的方法论：在 ControlNet、IP-Adapter 都还没有的时期，他们用「条件与时序解耦的两阶段训练」把稀缺的人物视频数据用起来，之后沿着 pose、click（局部运动）、handle（首帧编辑传播）、emoji（表情 landmark）、canvas（高清外扩）逐个把控制信号做细；他给的统一解释是——所有控制的本质都是对 attention map 施加影响，而基模型能力决定控制效果的上限。

## 核心

### 背景：控制信号五花八门，缺的是注入方式

讲者先给可控视频生成做了一次分类：camera/motion control 一系、pose 与 landmark、prompt 文本注入、depth、box（如字节 Boximator 一系）、frame 条件（插帧、图生视频本质都是条件）。这些信号本身不难拿，难的是在上游视频生成模型之上把信号有效注入、且不破坏时序一致性。Follow 家族就是按控制信号逐个攻的系列。

### Follow Your Pose：条件与时序解耦的两阶段训练

这是家族起点，动机很现实：带人物的姿态视频数据极难获取，直接训人物视频模型没有数据。他们的解法是两阶段：第一阶段用图像驱动学条件控制（图像数据充足），第二阶段再用视频信号学时序（cross-frame self-attention + temporal self-attention 做帧间与时序控制），把「条件信息」与「时序信息」解耦后最大化利用现有数据。配套还发布了姿态数据集（字幕作「line pose 数据集」，名称以论文为准）。讲者给的采用度：引用一百二十多次、GitHub 1.3K stars（字幕口径，截至讲授当日早晨）。

### Follow Your Click：局部动画难在数据构造

目标是 Dream2 的 regional motion brush 式局部动画：用户涂抹一块区域，那块动起来。真正的难点不是模型而是数据构造——人工挑局部运动样本太慢。他们用一个通用管线绕过去：用光流找出局部动作幅度最大的区域自动生成 mask，再把光流均值做成 motion 信号（延时摄影即使帧率高、动作也慢，光流能把真实运动强度提出来），配一个 motion 增强模块与短 prompt，得到可批量生产的训练数据。这件工作讲者自述中了 TPAMI 2024。

### Follow Your Handle：编辑首帧，靠 attention 传播到全片

思路与现在的 drag 操作同构：很多编辑本质是对首帧做线性/非刚性形变，只要形变能统一地应用到整个视频，就能得到一致的编辑。实现分两段：第一阶段先用 LoRA 微调把物体与 sketch/pose 的对应关系「绑」进模型；第二阶段在 inversion 与去噪两个过程中同时把编辑后的 condition 注入 self-attention——消融很干净：只在 inversion 注入，网络知道要编辑哪两个区域但编不出来；只在去噪注入，网络不知道源区域与目标区域在哪；两处都不注入则完全无效；两处都注入才成立。讲者由此提炼出全场最有迁移价值的一句：ControlNet 这类控制机制的本质，就是改 attention map。mask 的抠除则是为了显式告诉网络「哪块是待编辑区、哪块是控制区」。

### Follow-Emoji 与 Follow Your Canvas：把控制做细、做大

Follow-Emoji 冲着人脸版 Animate Anyone 去：表情表示上不用原始 landmark——去掉脸部轮廓（轮廓会把身份形状泄漏进驱动信号、干扰生成）、去掉鼻子、保留两只眼睛，再加 facial mask 与 expression mask 两路局部控制 loss；训练只用纯人数据，不构造杂物场景，靠生成模型自身的先验就得到不错的泛化（讲者举例：训练时嘴闭合，生成张嘴时牙齿仍是像样的牙齿）。配套发布 Emoji-Bench 补齐这一方向的评测缺失。

Follow Your Canvas 做高分辨率视频 outpainting：输入 $256 \times 256$ 、输出可达 $4096 \times 4096$ 。训练侧用两个不同位置的 anchor（一中一边）驱动生成其余区域，推理侧因显存必须分块，用 regional embedding 让网络知道每块在全局 layout 中的位置，再用多 GPU 分块推理后拼接。讲者自述该工作 TPAMI 评审分数不错（字幕作「7769」，含义存疑，未入数字表）。

### 讲者的判断：基模型决定上限，控制只是引导

Q&A 里讲者把话说得很直：控制信号并没有「限制」生成能力，效果差异主要来自基模型——混元 Video 基模型变强之后，Follow 系列在其上的效果整体抬升。所以做 research 资源不够时，他建议先在 SD 一系上做而不是直接上 DiT（太耗资源）。趋势判断：生成会往自回归（AR）走，AR 与扩散结合做大一统（下游任务本质都是「给 condition 生成」，可打成一个 batch），他正做的 3D/4D 生成也是这个逻辑的延伸（明年下半年将去 Meta 实习做该方向，字幕口径）。

### 科研分享（节选）

后半段讲者给在读研究生三点：个人能力与团队能力缺一不可（先自己分析再拿给老师确认，形成正向反馈）；做代表性工作要「多挖坑、少填坑」，A+B 式改进没有 novelty，图像/视频已是红海，去找 AR、3D/4D 这类蓝海；idea 成型后多找老师师兄聊，不存在「被偷 idea」，聊的过程就是在验证大方向。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| Follow Your Pose 引用量 | — | 120+ 次（截至讲授当日早晨） | 字幕 |
| Follow Your Pose GitHub stars | — | 1.3K | 字幕 |
| Follow Your Canvas 输入分辨率 | $256 \times 256$ | 可输出至 $4096 \times 4096$ | 字幕 |
| Follow Your Click 发表 | — | TPAMI 2024（讲者自述） | 字幕 |
| Follow Your Canvas 发表 | — | TPAMI（讲者自述，评审分数「7769」字幕口径存疑） | 字幕 |

## 可迁移

- 「两阶段解耦」是数据稀缺时的通用招：先在便宜数据上把条件对齐学会，再在稀缺数据上只学时序/动态——对应到 RL 训练就是先用离线数据教表示、再用在线 rollout 教时序信用分配，顺序反了两边都学不好。
- 控制即 attention 干预：做可控生成或可控 agent 时，优先想「我的条件信号应该改哪一层的 attention、怎么显式 mask 出待编辑区」，比在输入端堆条件编码器更直接可控。
- 自动数据管线先行：Follow Your Click 的真正贡献是光流自动挖 mask 的数据管线——评测/数据类工作里，「如何低成本造出带标注的训练对」往往比模型结构更决定成败。

## 疑问 / 下一步

- 各工作的确切发表状态与编号需逐一核对：讲者口径里 Click 与 Canvas 均为 TPAMI，Pose 的数据集名称字幕识别不清（「line pose」），引用前查项目页。
- Follow Your Handle 的「第一阶段 LoRA 绑定」细节（绑定的是哪种对应关系、在哪些层放开）字幕讲得较跳跃，需对论文方法节确认。
- 讲者判断「控制效果上限由基模型决定」给了混元 Video 的例子，但没有量化对比（如同一控制方法在弱/强基模型上的指标差），值得找后续工作验证。

## 原文金句

> 「ControlNet 本质上实际上就是对 attention map（施加影响），进而对整体的条件信息进行一个很好的引导。」——讲者复盘 Follow Your Handle 消融后（字幕约 15:50）

> 「多做挖坑的工作，而少做填坑的工作……如果只是一个简单的 A 加 B，那么实际上它并没有什么 novelty。」——讲者给研究生的选题建议（字幕约 26:05）

> 「（控制信号）本质上都是给一个 condition，然后让它来生成一些东西，所以本质上是一样的东西……都可以打成一个 batch 进行一个整合。」——讲者论大一统生成（字幕约 34:10）
