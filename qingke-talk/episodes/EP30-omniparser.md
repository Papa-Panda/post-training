# EP30 — OminiParser：基于纯视觉的 GUI Agent

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP30-omniparser.html

## 元信息

- 期号：EP30（官网 EP30，2024-11-30）
- 标题：OminiParser：基于纯视觉的 GUI Agent（官网登记标题作「OminiParser」，论文与项目原名拼写为 **OmniParser**，疑官网笔误，以下按论文作 OmniParser）
- 讲者：鲁亚东（微软研究院 AI Frontiers 实验室高级研究员；据官网预告嘉宾介绍，且为论文第一作者 Yadong Lu）
- BV：无（official-only，官网有预告、B站合集无对应视频）
- 时长：未知（官网预告 2024-11-30 上午 11:00 直播，未见回放存档）
- 提炼日期：2026-10-02
- 相关论文：Yadong Lu, Jianwei Yang, Yelong Shen, Ahmed Awadallah, *OmniParser for Pure Vision Based GUI Agent*，arXiv:2408.00203（2024-08-01 提交），https://arxiv.org/abs/2408.00203
- 相关代码：https://github.com/microsoft/OmniParser
- 官网预告：https://qingkeai.online/blog/ZXoBVSOE

> 📝 提炼方式说明：本期为 official-only，B站合集无对应视频、无字幕可用，本纪要基于对应论文（arXiv:2408.00203）还原，**非逐字稿**；讲授环节的展开顺序、口头补充与 AMA 内容均无法确证，以下内容以论文为准，凡涉及「讲者观点」的表述均指论文作者在文中给出的判断。

## 一句话总结

GPT-4V 做 GUI Agent 跨平台跨应用不行的瓶颈不在推理，而在没有可靠的纯视觉屏幕解析：OmniParser 用 67k 网页 DOM 框微调一个「可交互元素检测模型」+ 7k 图标描述对微调一个「图标功能描述模型」+ OCR，把截图解析成带编号框与局部语义的结构化文件，让不依赖 HTML/view hierarchy 的 GPT-4V 在 ScreenSpot 上从 16.2% 涨到 73.0%，并在 Mind2Web、AITW 上超过需要额外结构化输入的 GPT-4V 基线。

## 核心

1. **背景/问题**：VLM 做跨平台（Windows/macOS/iOS/Android）、跨应用 GUI Agent 的核心短板是 action grounding——GPT-4V 给不出精确的点击 xy 坐标。Set-of-Mark（SoM）用带编号的框把「输出坐标」换成「输出框 ID」绕开了坐标问题，但框从哪来？已有方案从网页 DOM 或手机 view hierarchy 取真值框，桌面应用与任意截图场景拿不到。讲者一方的判断（论文原话）：GPT-4V 作为通用 agent 的能力被严重低估，缺的只是一个稳健的屏幕解析技术——(1) 可靠识别可交互图标，(2) 理解各元素语义并把目标动作关联到正确区域（来源：arXiv:2408.00203，§1 与摘要）。
2. **方法/设计**：三件套流水线。① 检测：从热门网页 DOM 树抽可交互元素框，构建 67k 张截图的检测数据集，微调 YOLO 系检测模型（论文称 interactable region detection model），与 OCR 的文字框合并、去重（重叠 > 90% 去一）并给每框编号；② 描述：用 GPT-4o 造 7k 图标-描述对，微调 BLIP-v2 给每个检出图标生成一句功能语义（如「设置」「最小化」），OCR 文字直接作语义；③ 喂给 GPT-4V 的是 SoM 标注截图 + 局部语义文本 prompt，模型只需输出框 ID（来源：arXiv:2408.00203，§3.1、§3.2）。
3. **实验/实战**：三个基准全用现成 GPT-4V 不微调。ScreenSpot（600+ 跨平台截图）：纯 GPT-4V 平均 16.2%，OmniParser（局部语义 + 微调检测）73.0%，压过专门在 GUI 数据上微调的 SeeClick（53.4%）与 CogAgent（47.4%）；消融显示微调检测模型比现成 Grounding DINO 多 +4.3%。自建 SeeAssign 诊断集：加上局部语义后 GPT-4V 选对框 ID 的准确率 0.705 → 0.938。Mind2Web（网页多步导航）：仅截图输入的 OmniParser 在 Cross-Domain/Cross-Website/Cross-Task 三个切分的步骤成功率（Step SR）为 42.0% / 36.5% / 39.4%，全面超过用 HTML 的 GPT-4（36.2% / 30.1% / 26.4%）与用真值框 SoM 的 GPT-4V。AITW（手机导航）：总分 57.7%，比最佳 GPT-4V+history 基线（53.0%）高 4.7 个百分点（来源均同上，§4.1–§4.4）。
4. **结论/观点**（论文口径，非讲者原话）：作者认为「屏幕理解好的小模型 + 通用大模型」是比分工端到端微调更可扩展的路线——解析质量（可交互框准不准、语义给没给）是上限，谁掌握解析层谁就解锁了冻结大模型的 agent 能力。

## 关键数字

| 指标 | 基线 | 结果 | 来源 |
|---|---|---|---|
| ScreenSpot 平均准确率（GPT-4V） | 16.2%（纯 GPT-4V） | 73.0%（OmniParser：局部语义+微调检测） | arXiv:2408.00203，Table 2 |
| ScreenSpot 对比微调型模型 | SeeClick 53.4% / CogAgent 47.4% | OmniParser 73.0% | arXiv:2408.00203，Table 2 |
| 检测模型消融 | Grounding DINO 现成检测 | 微调检测模型再 +4.3% | arXiv:2408.00203，§4.2 |
| SeeAssign 选对框 ID 率 | 0.705（无局部语义） | 0.938（加局部语义） | arXiv:2408.00203，Table 1 |
| Mind2Web 步骤成功率（跨域/跨站/跨任务） | GPT-4（用 HTML）：36.2% / 30.1% / 26.4% | OmniParser（纯截图）：42.0% / 36.5% / 39.4% | arXiv:2408.00203，Table 3 |
| AITW 总分 | GPT-4V+history 53.0% | OmniParser 57.7% | arXiv:2408.00203，Table 4 |
| 训练数据 | — | 检测集 67k 截图（网页 DOM 框）；描述集 7k 图标-描述对（GPT-4o 生成） | arXiv:2408.00203，§3.1、§3.2 |

## 可迁移

- 环境观测的结构化中间层是 agent 的「感知 API」：OmniParser 的关键动作是把原始截图变成「编号元素 + 功能语义」的结构化文件再喂策略——同样的分工可搬到 coding/terminal agent：把终端输出、DOM、日志先解析成可寻址的结构化状态（元素 ID + 语义注解），让冻结的大模型只做选择与规划，不让策略模型自己兼任感知。感知层可以独立评测（SeeAssign 式「指对了吗」的最小诊断集，先量解析层上限，再谈端到端）。
- 「小而专的解析模型放大冻结大模型」是数据飞轮的一种形态：67k/7k 这种可从程序化来源（DOM、真值结构）自动造的标注，比端到端行为克隆便宜一个量级；做 post-training 数据管线时，先问有没有可以从结构真值自动生成监督的解析子任务。

## 疑问 / 下一步

- 论文自陈三类失败：重复图标/文字无法消歧、OCR 框粒度粗导致点击点落在框外、图标描述模型只看裁剪小图会误读（如三点图标读成「加载中」）——纯视觉解析在高密度桌面 UI 上的准确率天花板与误检如何随分辨率缩放，论文未系统分析，等待后续工作（该项目已有 V2 迭代，未包含在本期与本篇范围）。

## 原文金句（1-2句）

> （论文原句，arXiv:2408.00203 §1）"we argue that the power of multimodal models like GPT-4V as a general agent ... is largely underestimated due to the lack of a robust screen parsing technique"

> 说明：本期无字幕，上引为论文原文，非讲者口述。
