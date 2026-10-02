# EP08 — VideoBooth：文本和图像提示共同驱动的视频生成

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP08-videobooth.html

> 「至于说他只比如他的鼻子这里、眼睛这里它的花纹具体是长什么样的，很多网络不知道。」——讲者谈 coarse 层 CLIP 特征为什么不够、必须再补 fine 层细粒度注入

## 元信息

- 期号：8
- 标题：VideoBooth：文本和图像提示共同驱动的视频生成
- BV：BV1une9zME2m
- 时长：00:38:06
- 提炼日期：2026-10-02
- 分享嘉宾：南洋理工大学博士生（字幕未口播姓名；自述导师为刘子伟老师与李建勤老师；正式姓名以论文作者页为准）
- 相关论文：VideoBooth（CVPR 2024；字幕自述发表于 CVPR 2024，与上海 AI Lab、北京大学合作）
- 相关代码：推理代码已开源；数据与训练代码讲者称将陆续同步到 GitHub（字幕口径）
- B站链接：https://www.bilibili.com/video/BV1une9zME2m/
- 官网期号：8
- 字幕原文存档：本地 `transcripts/EP08.txt`（820 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP08，已存档）清洗提炼，信息来源为 AI 字幕原文（经人工清洗；个别专名可能有识别误差，如「video booth」在字幕中统一作「video boost/boos」、「coarse-to-fine」作「cost to find」、「cross-frame attention」作「cross fattention」、「UNet」作「unit」、「ELITE」作「依赖」、「WebVid」疑作「YYD」，均以论文与公开资料为准）。凡讲授口径与论文版可能不同，以下标注（字幕口径）。

## 一句话总结

VideoBooth 解决纯文本控制的视频生成无法定制主体外观的问题：用户给一段文字加一张主体参考图，模型生成既符合文字动作、又保持这只具体主体 appearance 的视频；方法是 coarse-to-fine 双层注入（CLIP 图像特征经 MLP 对齐文本空间给高层语义，UNet 特征经 cross-frame attention 给低层细节），并且训练也必须两阶段，否则强细粒度特征会让语义层学不到有意义的表征、采样时 appearance 直接崩掉。

## 核心

### 引入：为什么只用文字定制不了主体

文字驱动的视频生成效果已经很好，但定制化场景里不够用。讲者用「一只具体的狗在笼子里吃零食」举例：把文字改成对这只狗外貌的长篇枚举（耳朵、花纹等属性逐条写）再生成，结果只是大致相似、并不是那只狗。原因是两条：一张图片的属性根本无法用文字枚举完整；现有模型对长文本里各项 attribute 的捕捉也不精准。更直接的办法就是把图片本身送进去——文字管动作、图片管 appearance，两路一起驱动生成。

任务定义就是：给一段文字和一张主体 reference 图，输出同时符合文字内容与图片 appearance 的视频。讲者演示了同一只狗喝水/游泳/在公园/望车窗，以及同一动作换不同熊猫主体，两种维度都能保持主体特征。

### 方法：coarse-to-fine 双层注入

基座是经典 UNet 式视频 diffusion，每个 block 有三类 attention：cross-frame attention（用第 0 帧与前一帧做 key/value 保帧间连续性）、cross attention（接收文本的 CLIP embedding）、temporal attention（捕捉合理动作）。VideoBooth 在此之上加两条注入支路：

1. **Coarse 层（语义）**：冻结的 CLIP image encoder 提取图像特征，经 MLP 映射到文本空间，与文本 embedding 拼接后一起送入 cross attention。CLIP 靠图文联合训练天然有图文理解能力，这支负责告诉网络「这只狗大概长什么样」的高层概念。
2. **Fine 层（细节）**：CLIP 把图片压成一维向量会丢 spatial 信息，花纹、鼻子眼睛长什么样都不知道。所以再用 image prompt 过 UNet 提特征（这类特征本来就保留 spatial 信息），作为额外的 key/value 注入 cross-frame attention，让生成过程回看参考图的低层 appearance 来 refine 细节。

可训练的只有两处：CLIP 特征到文本空间的 MLP，以及两个 attention 里的 KV projection。

cross-frame attention 还做了额外设计：先用 image prompt 作为额外 key/value 更新第 0 帧，再用已更新过的第 0 帧特征去更新后续帧，把参考图信息先固化到第一帧、再传播出去。

### 训练也必须 coarse-to-fine：顺序错了语义层就废了

这是讲者全场最强调的工程发现。image prompt 的 UNet feature 是非常强的 visual cue：如果两支一开始就合在一起训，网络直接抄 fine 层答案，coarse 层学到「完全没有意义的表征」，训练 loss 还是能降得很小——因为 diffusion 训练是加噪还原，加噪并没有完全破坏原图信息，fine 支等于自带答案。但真正 sampling 是从纯高斯噪声开始的，没有语义层锚点，网络直接生成一个不相关的物体，fine 层再 refine 也无从下手。

所以训练分两阶段：先只训 MLP 与 KV projection 让语义层立住，再加入 attention 注入模块学细节。消融三连证实了这条链：

- 只有 coarse：种类、花纹大概能给，土司帽（字幕词）这类具体细节搬不过来。
- 只有 fine：直接把第一帧抄过来，后续帧完全靠第一帧 propagate，细节照样崩。
- 非 coarse-to-fine 训练：表现与「只有 fine」几乎相同——后几帧 appearance 直接崩掉，反证语义层在混训里确实学空了。

这个结论值得单独记：**训练 loss 低不等于采样能用，强支路会把弱支路的梯度信号吃光**，这是多条件注入时很通用的一种失败模式。

### 定性对比与局限

当时视频领域还没有现成的 customized 生成基线，讲者把三个图像方法（Textual Inversion、DreamBooth、ELITE）搬上视频基座改造后作对比。反直觉的现象是：fine-tuning 类的 Textual Inversion / DreamBooth 反而更差。讲者的解释是：图像设置里能找同一物体在不同场景的大量照片，网络能抽象出表征；视频设置里很难凑齐同一物体不同动作的视频，只能把长视频切段训练，fine-tune 就 overfit 到见过的场景，生成时偏差大。

VideoBooth 自身的局限：生成动作的 pose 有时会像 image prompt 的 pose（数据集把每段视频第一帧抠前景当 reference，训练对本身就很像），训练时用随机 masking 做 augmentation 缓解；想让 pose 真正自由还需要 image-to-3D 做 novel view 扩增（讲者列为思路）。另外训练数据（字幕作「YYD」，应为 WebVid 类）motion 偏小，动作幅度大的长视频受基座限制。

### Q&A 要点

- **风格与背景**：风格迁移暂不做，只保主体内容一致；给图的背景会被抠掉，只保前景，背景多样性靠文本控制。
- **图文不一致怎么办**：管线里「a dog」这个词会被 image prompt 的 embedding 直接替换掉，所以不存在图文打架；动态能力减弱确实存在，一半是数据集 motion 小、一半是 UNet 基座长视频算力限制。
- **计算成本**：相对基模型几乎不加额外成本；长视频/高分辨率的上限还是基座本身，讲者认为方法可迁移到 DiT 架构。
- **应用**：现有小规模数据上泛化偏弱（人体结构保持不好），大规模高质量数据上才有内容创作潜力；真要定制动作仍需关键点/骨骼这类显式控制补进来。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| 训练阶段数 | 一次混训（语义层学空，采样崩） | 2（先语义层，后细粒度注入） | 字幕（方法/消融） |
| 可训练参数 | 全网可训 | 仅 MLP + 两个 attention 的 KV projection，CLIP image encoder 冻结 | 字幕（方法） |
| 额外计算成本 | 基座 UNet 模型 | 讲者称相较基模型「没有引入很多额外计算成本」 | 字幕（Q&A） |
| 数据开源时间 | 推理代码已开源 | 数据约两周、训练代码约下月同步 GitHub | 字幕（Q&A） |

注：本期以定性演示与消融为主，字幕未给任何 benchmark 数值，本表不编造量化指标。

## 可迁移

- **条件注入的 curriculum 原则**：多路条件强度差异大时，先让弱支路（语义锚点）单独站住再放进强支路，否则强支路会靠「抄答案」把训练信号吃掉。这是「训练 loss 正常、采样失败」的一个很具体的成因模板，比笼统说分布偏移有用。
- **reference 传播的最简结构**：先更新第 0 帧再向后 propagate，是把静态条件缝进时序模型的最小改动；任何图生视频/条件视频生成任务都可先试这种「锚点帧先固化」的做法再谈复杂机制。
- **消融设计顺序**：单 coarse、单 fine、非分阶段训练，三格消融直接把「配合方式」和「模块本身」分开验证，比堆模块加表格更省轮次。

## 疑问 / 下一步

- pose 锁定问题最终解得怎么样：novel view 扩增（image-to-3D）是讲者给的思路，但字幕未提后续验证结果。
- coarse 层把主体词 embedding 直接替换掉，等于默认图文主体永远一致；真实用户给错主体词时（猫图配「狗」文本）行为边界是什么，字幕只答了「暂不存在此问题」，未做过压力测试。
- 训练数据 motion 小是事实，但 mask augmentation 对「生成时只见 partial view」有帮助、对 pose 解锁是否同等有效，没有数字支撑。

## 原文金句（1-2句）

> 「如果我们直接把这两个部分合在一起训练，那么他得到的结果就是说我这边的这个 coarse level 的这个 image embedding，就会学到一些完全没有意义的表征。」——讲者解释为什么训练必须分阶段（字幕 15:34–15:45 附近，干净口径转写）

> 「那我们在做真正的做 sample 的时候，我们其实是完全从一个高斯噪声来开始做 sampling 的。」——紧接其后点出训练与采样的不对称：训练里抄得到的答案，采样时根本不在场。
