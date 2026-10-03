# CUA — CUA-Suite（ICLR 2026 CUA Workshop）

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/CUA-suite.html

## 元信息

- 专题：ICLR 2026 CUA Workshop（青稞合集 sid=7971490）
- 标题：CUA-Suite
- BV：BV11fo5BtENp
- 时长：约 28 分钟（1679 秒）
- 观看/提炼日期：2026-10-02
- 讲者：字幕未自报姓名（CUA-Suite 作者；自述原 ServiceNow AI Research、已离职，与 Salesforce/ServiceNow AI Research 及 Mila、Oxford、NUS 的合作者共同完成，此处不猜姓名）
- B站链接：https://www.bilibili.com/video/BV11fo5BtENp/
- 字幕原文存档：本地 `transcripts/CUA-SUITE.txt`（747 条，带时间戳）
- 说明：基于 B站 AI 字幕整理；工作名（字幕作 UI-Vision、GroundCUA、GrounNext 等）按读音转写，未逐一核对官方拼写；个别基线名（字幕作 UGound、UI-TARS 等）以可辨者为准。

## 一句话总结

CUA-Suite 是讲者团队把同一套人类示范数据做成三件套并全部开源：评测基准 UI-Vision（专攻企业/专业软件的 grounding 评测）、训练数据与模型 GroundCUA/GrounNext（高密度 keyframe 标注 + SFT 后再 RL 的桌面 grounder），以及 55 小时连续视频 VideoCUA（字幕口径）——核心主张是企业级 CUA 的瓶颈在长尾专业软件的数据与 grounding，不是再刷一遍常见 App。

## 核心

1. **背景/问题**：web agent 数据与 benchmark 已经很丰富，但 ServiceNow、Salesforce 这类 enterprise software 的数据和模型关注有限；而企业场景的问题天然是长尾的。讲者团队自去年初起收集了 111 套（字幕口径）人类示范数据，围绕它做了评测、训练、视频释放三条线。

2. **UI-Vision（评测）**：87 个应用、12 个类别（早期版本 83 个），约 1 万个 task；keyframe 上可获得几乎全部可交互元素的准确 bounding box 与 label。评测分三类 grounding 场景（basic / functional / spatial）、layout parsing、以及把轨迹任务原子化成单步的 action prediction。发现：(a) 当年 SOTA（字幕作 UGound、UI-TARS）在专业软件上大量归零——长尾效应极其显著；(b) 影响 grounding 的两个可辨因素是 element 相对尺寸与每屏 confounding element 数量；(c) 典型错误是细粒度歧义（horizontal vs vertical）、domain knowledge 缺失（如 widget 的特殊含义）、小元素密集场景，以及训练分布偏置（macOS 窗口按钮在左，模型永远预测右上角）。

3. **GroundCUA / GrounNext（训练）**：从 1 万条视频轨迹的每个 action keyframe 做高密度标注，平均 64 个 element/截图、最多 542 个；纯 vision 输入（很多软件拿不到 accessibility tree）。指令构造沿用 UI-Vision 的三类逻辑，数据 recipe 为 direct 50% / functional 35% / spatial 15%（字幕口径）；released instruction data 做过 deduplication（pHash + text match）。训练为 SFT 后再做一轮 RL（RLOO 式相对策略优化，字幕口径），reward 不用连续距离也不用纯 0/1，而是分段的离散化函数区分 near miss 与 far miss。结论：同尺寸下性能与当年强基线可比，且跨域（ScreenSpot-Pro、OSWorld-G 等，UI-Vision 为 in-domain）表现好；但 SFT 足够强时 RL 的 headroom 有限，增益约 1.7（字幕口径，未指明单位）；把 GrounNext 3B 接到 o3 planner 上测 OSWorld，多类任务效果很好。

4. **VideoCUA（释放）**：团队离职后把 55 小时连续视频、约 600 万帧与处理代码全部开源（比对对象为 OpenCUA、ScaleCUA 等，字幕口径）；连续 video 不只有 keyframe，还带完整轨迹，可平滑转成 OpenCUA/ScaleCUA 式训练格式，也可支撑 video-based reward model、visual world model 等后续方向——这部分明确「交给社区」。

## 关键数字

| 指标 | 数字 | 口径 |
|---|---|---|
| 应用 / 任务规模 | 87 个应用、12 类，约 1 万个 task | 字幕（[02:16] 起） |
| 视频总量 | 55 小时、约 600 万帧 | 字幕（[02:56]、[24:31]） |
| keyframe 元素密度 | 平均 64 个/截图，最多 542 个 | 字幕（[15:46]） |
| 指令数据 recipe | direct 50% / functional 35% / spatial 15% | 字幕（[17:50]） |
| RL 增益 | 约 1.7，且 SFT 强时 headroom 有限 | 字幕（[23:05]，单位未明） |

## 可迁移

- 评测任何 GUI grounding 模型时，把「每屏 confounding element 数」和「element 相对尺寸」单独分层出分——总体准确率会掩盖专业软件上的长尾崩塌。
- grounding 的 RL reward 别用连续距离也别用纯 0/1：分段离散化（命中框内、near miss、far miss 分档）训练更稳、信号更密，讲者称简单但有效。
- 数据采集优先选 license 干净的专业/创意软件并做密集 keyframe 标注：一套数据同时支撑 benchmark、grounder 训练和视频预训练，比三套数据各行其是便宜得多。

## 疑问 / 下一步

- UI-Vision 上新一代模型（如讲者提到的 MAI-UI 已把成绩翻倍）之后，spatial grounding 与专业软件长尾还剩多少差距？讲者只给了「翻倍」的口头说法，后续 workshop 的 MAI-UI 专场正好可以对照。

## 原文金句

> 「在 enterprise 的很多 scenario 里，问题就是长尾的。」
