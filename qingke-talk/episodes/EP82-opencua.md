# EP82 — OpenCUA：用于构建 Computer-Use Agent 的开源框架

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP82-opencua.html

> 「我们希望给大家构建一整套这个 CUA 的 Open foundations。」——讲者对本工作定位的一句话

## 元信息

- 期号：82
- 标题：OpenCUA：用于构建 Computer-Use Agent 的开源框架
- BV：BV14isozHErN
- 时长：00:52:09（讲授约 40 分钟 + Q&A 约 12 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：王欣远（香港大学博士生；讲者字幕自述姓名，主持人介绍作「王新远」，以自述为准；字幕中实验室名称识别不清，不具名）
- 相关论文：*OpenCUA: Open Foundations for Computer Use Agents*（讲者称已被 NeurIPS 2025 接收；港大团队与 Kimi 等机构合作，历时一年多；字幕口径）
- 相关代码：OpenCUA 系列模型（7B / 32B / 72B）与数据、评测工具均已开源（讲者口径，具体地址以项目页为准）
- B站链接：https://www.bilibili.com/video/BV14isozHErN/
- 官网期号：82
- 字幕原文存档：本地 `transcripts/EP82.txt`（1003 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP82，已存档）清洗提炼，信息来源为 AI 字幕原文（经人工清洗；个别专名可能有识别误差，如 OpenCUA 在字幕中作「open CV / open ca」、AgentNet 作「internet / aj nine」、Qwen2.5-VL 作「千万25VL」、NeurIPS 作「NEX2025」，均以论文与公开资料为准）。凡讲授口径与论文版可能不同，以下数字均按字幕口径记录。

## 一句话总结

OpenCUA 为 computer-use agent（CUA）补齐开放基建：用自研录制工具把真人操作录成带像素级动作的轨迹，构建 22K 任务的 AgentNet 数据集；用「生成器 + 反思器」流水线合成带反思环节的结构化 CoT，配合多层级数据增强与历史信息压缩，训出 7B/32B/72B 三档模型（72B 在 OSWorld-Verified 达 45 分，开源最佳）；并给出离线评测基准与 pass@1/pass@N 差距分析，指向 RL 是下一步的关键。

## 核心

### 背景：computer 是最通用的 digital environment

讲者先把概念摆正：一切由程序构成的虚拟环境都是 digital environment（coding、手机、web），而 computer 是其中最通用的一个——理论上其他数字任务都能在一台电脑里完成。当前事实标准环境是 OSWorld：真实的 Ubuntu 电脑，可获取截图与 accessibility tree（无障碍树），可注入鼠标/键盘动作，含 369 个相对真实的用机任务。

近一年各大厂都把 computer use 当作重要 feature 发布（Claude 3.5/3.7、OpenAI Operator、UI-TARS、Claude Sonnet 4.5、ChatGPT Atlas，字幕口径）。但讲者引出实验室同期工作 ComputerArena 的众包真人评测（评测员在两台一模一样的 Windows 虚拟机上给两个匿名 agent 派同一任务、判优劣、ELO 汇总）：榜单上 Claude 4 在只需十几步 GUI 动作的任务上 correct rate 也才勉强过 50%。benchmark 分数不断涨、真实成功率却不高——原因是什么、怎么提升、以及 privacy/safety 怎么研究，都需要开放基建，这就是 OpenCUA 的 motivation。

### observation 与 action 的设计空间

- **observation 从文本转向视觉**。早期用 accessibility tree（按钮文字描述 + 坐标）喂 LLM，问题有四：获取需遍历整棵树、慢；依赖软件开发者是否认真填写；界面大量无用信息即噪声；解析后仍占大量 token。从 Aguvis（字幕作「AQU维斯」）开始社区转向纯截图：更符合人类操机直觉，代价是想看屏幕外信息必须滚动。
- **action 从原子动作扩展到工具调用**。最底层是 pyautogui（字幕作「pilot to g o i」）式的 click/type 等原子操作，或人工封装的 move/click 动作集；再往上是人工实现的 API / MCP / tool calls。讲者的对比很直白：纯 GUI 打开一个软件要「点菜单→搜索→点开」好几步，写成一个 tool 一步到位，还减少执行期出错的机会；但代价是每个环境都要人工去支持这些工具。

### 数据：22K 条真人录制轨迹（AgentNet）

讲者强调 CUA 数据是 embodied agent data，必须与真实环境交互才能获得，标注复杂度远高于图文数据；网上虽有大量 YouTube 教程，但没有 grounded action——只说「第一步做这个」，不告诉你点屏幕哪个像素。实验室的 VideoAgentTrek（字幕作「video ation track」）把无标注 YouTube 视频转成可训练轨迹，是同一方向的补充，但总体开源数据仍非常稀少。

OpenCUA 的解法是自研录制软件：标注员在自己电脑上装好后录制完成任务的全过程，软件同步记录视频与鼠标键盘操作，并切分 scroll、click 等离散动作。由此建成 AgentNet——讲者称是第一个 CUA 数据集，含 22K 个复杂多样的真实用机任务，覆盖 140 多个 app 与网站、三种主流操作系统（Windows、macOS、Ubuntu），覆盖电商、社交、办公与专业软件等日常场景。

### 建模：带反思的结构化 CoT 是质量关键

有了数据后怎么用，讲者先给了一个负结果：仿 Aguvis 的做法，让 GPT 拿截图 + 已标注动作补一段简单 CoT，在 20K 数据上测下来「并不能很好地带来 benchmark 上的提升」。他们的判断是必须用好 VLM 预训练里已有的 GUI 与软件先验，而语言（reasoning）部分是关键——这正是 ReAct 的教训：先 reason 再 act 能显著提升表现。

建模分两侧：

- **输出侧**：CoT 按模块组织——perception（理解当前状态）、reflection（判断之前动作有没有错、错了怎么改）、memory、plan，最后落到 grounded action。讲者特别强调 reflection 的重要性，给了一个通俗算术：单步成功率 99% 的 agent，做 50 步任务整体成功率至多 $0.99^{50}$ ，约 60%；学会执行中纠错才能把这个数提上去。数据合成流水线是「generator + reflector」：generator 生成 CoT，reflector 拿到动作前后的两张截图，判断动作是否正确完成、带来了什么变化、错的话正确动作应是什么，逐条轨迹循环生成。例子：插入五行三列表格时第 8 步把 5 误输成 1，第 9 步的 CoT 反思到「我应该输入 5」，plan 里决定选中改掉，后两步完成纠正。结构化 CoT 还顺带支持数据增强：一份轨迹派生三层标签——L3（observation + thought + action）、L2（thought + action）、L1（只有 action）；层级越低与动作的直接相关性越强，训练时他们提高了 action 与 thought 的权重。
- **输入侧**：历史怎么塞进有限 context。一张高清截图约一两千 token，不可能全塞。实验结论：保留最近 3 张图（3 张与 5 张差异不明显），更早的步只用一句话描述「这一步做了什么」（L1）；把历史 thought 全文放进去反而不如只放 action 描述——thought 长且有效信息密度低，对 VLM 是大海捞针。讲者坦承这块的已知缺陷：任务中途的信息会被遗忘（比如要从三个 Excel 里分别取信息，只能记住最近一个）。

训练侧的其余配方：与大量 agent grounding 数据混合提升定位能力；不同层级 CoT 与通用 SFT 数据混合提高稳定性；base model 从 Qwen2-VL 换到 Qwen2.5-VL（高分辨率理解更好、已预训练不少 GUI 数据）。最终产出 OpenCUA-7B、32B、72B。

### 实验：数字全部按字幕口径

- **主结果**：72B 在 OSWorld-Verified 上 45 分，讲者称超过 Claude Sonnet（字幕作「cos sonnet」），且是目前开源最好；在 UI-Vision（grounding 基准）拿到 SOTA（32B 在 grounding 与 planning 上也有好表现）。
- **Scaling**：在 Qwen2-VL 上数据从 10K 扩到 27K，performance 基本稳定提升；换 Qwen2.5-VL + 更多数据得到 7B/32B，72B 用了更多数据。
- **跨域迁移**：只用 Ubuntu 数据训 vs 只用 Windows/Mac 数据训，分别在 OSWorld（Ubuntu）、Windows Agent Arena（Windows）、AgentBench（字幕作「ahi bench」，含 Windows 与 Mac）上测——跨 OS 也能涨。讲者的解释：各软件的 GUI 操作逻辑相通，在一个软件里学到的 workflow 对别的软件有帮助；但操作逻辑高度专业的软件迁移能力弱。
- **Test-time scaling 与 pass@N 差距**：最大步数从 15 放宽到 100 有明显提升（部分任务本就需要更多步，模型在长任务上也能做）。更关键的发现是 pass@1 与 pass@N 的巨大差距：72B pass@1 为 45 分、pass@3 就有 53 分（+8）；一个基于 Qwen2.5-7B 的模型在 OSWorld 上跑 16 遍，pass@1 只有 18 分，pass@36 比它高一倍还多。讲者的归因观察有两条：一是点击不稳定——同一按钮这次点中、下次差几个像素点错，点进错误页面还回不来，任务直接失败；二是同一任务有多条难度不同的完成路径（恢复浏览器页面：会快捷键的人 Ctrl+Shift+T 一步到位，不会的人要翻历史记录），模型每次采样选的路径不同，成败随之摇摆。结论：pass@1/pass@N 的 gap 正是 RL 可以缩小的空间，近期已有多篇工作在做。
- **离线评测基准**：从 AgentNet 里选 100 个任务做 offline test set，人工检查每一步正确性并列出所有可能选项；离线结果与在线结果存在相关性。讲者也提醒：OSWorld 上分数已经很高，但真实生活里大家还不用 CUA，未来需要更能反映用户实际体验的 benchmark。

### 未来方向（讲者列的研究议程）

agent pre-training（把 agent 数据占比提上去，VideoAgentTrek 即此方向）；更好的学习算法（RL，如苹果的 UltraCUA：在 OpenCUA-32B 上先 SFT 再 RL，15 步预算下把分数从 29.7 提到 41，讲者记得 50 步约 43 分但不确定，字幕口径存疑）；更高效的 action space（把写代码本身做成 action、大段代码省掉大量 GUI 动作；scale 工具调用已被证明有效）；universal digital agents（把各种 agent 能力熔进一个 base model，Claude 4.5、UI-TARS-2 在做）；agent framework（在强基模型上按任务设计模块，组合后明显强于模型裸跑）；模型结构（历史图片的 context 利用效率，如 JoyKV 一类从 KV cache 角度压缩）；以及性能之后的问题——reliability、security、personalization、人与 agent 的交互形式，乃至通用 agency 何时到来。

### Q&A 要点

- **3 张图 vs 5 张图差异为何不大？** 越靠后的图片与当前动作关系越弱、重要性越低。Claude 4.5 用了 10 张图但把图片 resize 到 1280x720；未来方向是降分辨率、塞更多图，外加把超出窗口的历史图片信息提取成文本。
- **截图冗余信息能否压缩 image token 省 context？** 可以探索：JoyKV 从 KV cache 角度压缩历史图片，或者用文本代表图片中的重要信息。
- **自动化构建数据的方法？** VideoAgentTrek（视频转轨迹）之外，还有 trajectory 合成类工作，如 AgentSynth（字幕作「agent since」）从易到难自动生成任务。
- **任务描述能否从语言扩展到用户演示？** 讲者很认可：自己先录屏做一遍，下次把视频给 agent、任务略有变动它照着做——本质是 in-context learning，也是 personalization 的形态。
- **vision vs CLI/代码哪种交互是未来？** 各有优劣、必然结合：CLI 适合把任务托管后台长时间跑（如 24 小时持续工作），vision 适合与人交替操作的直观场景。被追问「AI 必须像人一样用视觉吗」，讲者认为视觉不可或缺：数字环境本就是为人眼设计的，建网站这类任务还涉及美学判断，纯文本表示给不了这种判断力。
- **L1/L2/L3 是同一条轨迹同时产出吗？** 是，一份数据增强成三份；UI-TARS 这类只有一种 CoT 格式的工作不需要这么做。
- **GUI + API 的 benchmark / agent 有意义吗？** 有意义，但不应局限在某个 domain；Claude 4.5 已把所有动作统一进 tool-call 的 action space，给它 API 它就会用；UltraCUA 自研的工具也可通俗理解为 API。
- **模型压缩（量化/稀疏/低秩分解）的意义？** 72B 直接跑资源量很大，压到 1/10 就能用少得多的资源做几乎同样的事；7B 级模型压缩后未来可能直接装进手机。
- **用代码解决电脑问题的 agent 推荐？** SWE-agent 一类专攻 coding 的 agent，看 SWE-bench 即可。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| OSWorld 任务数 | — | 369 个任务 | 字幕（环境介绍） |
| ComputerArena 真人评测（Claude 4） | 十几步 GUI 动作的任务 | correct rate 勉强过 50% | 字幕（同期工作转述） |
| AgentNet 数据集规模 | — | 22K 任务、140+ app/网站、3 种操作系统 | 字幕（数据部分） |
| 简单 CoT 消融 | Aguvis 式 GPT 补 CoT，20K 数据 | 未带来明显 benchmark 提升 | 字幕（建模部分） |
| 历史输入配置 | 3 张图 vs 5 张图差异不明显；更早步只留一句话动作描述 | 选定 3 张图 + L1 文本 | 字幕（输入建模实验） |
| 单步 99% 成功率的连乘上限 | 50 步任务 | $0.99^{50}$ ，约 60% | 字幕（reflection 动机） |
| 数据 scaling（Qwen2-VL） | 10K 条 | 27K 条时 performance 稳定提升 | 字幕（实验部分，未给具体分数） |
| OpenCUA-72B（OSWorld-Verified） | 开源模型此前成绩 | 45 分（pass@1），讲者称超过 Claude Sonnet、开源最佳 | 字幕（实验部分） |
| pass@3 vs pass@1（72B，OSWorld-Verified） | pass@1 = 45 | pass@3 = 53（+8 分） | 字幕（pass@N 分析） |
| pass@N 上限探针（Qwen2.5-7B 基座，OSWorld，16 遍采样） | pass@1 = 18 | pass@36 高一倍还多 | 字幕（pass@N 分析） |
| 最大步数放宽（test-time scaling） | 15 步上限 | 放宽到 100 步有明显提升 | 字幕（实验部分） |
| 离线评测基准规模 | 从 AgentNet 选取 | 100 个任务、逐步人工校验 | 字幕（评测部分） |
| UltraCUA（苹果，OpenCUA-32B 上 SFT+RL，15 步预算） | 29.7 分 | 41 分（讲者对 50 步约 43 分不确定） | 字幕（未来方向转述，存疑已标注） |

注：讲者提到 72B 在 UI-Vision 拿 SOTA、跨 OS 迁移涨点、离线与在线评测相关，均未在字幕中给出具体数值，故只记结论不记数字。

## 可迁移

- 对 post-training / agent 数据工作的直接启发：**轨迹数据的价值不在「成功」本身，而在是否含有错误—反思—纠正的完整回路**。只录成功轨迹合成简单 CoT 涨不动点；把 reflector（对比动作前后状态、判对错、给出正确动作）做成数据流水线的一环，CoT 质量才真正上去。这条对 coding agent 的轨迹合成同样成立。
- Infra 视角：CUA 的 context 经济学很硬——一张图一两千 token、长任务几十步，history 管理（近图全留、远步压成一句话、再往后靠文本提取）本质是给 agent 做的 KV/上下文预算分配；pass@1 与 pass@N 差一倍以上这个观察，把「评测用多次采样估上限、再用 RL 把 pass@1 往上限推」变成了可量化的立项依据。另外讲者把混合层级 CoT + 通用 SFT 数据当作训练稳定性手段，与 RL 数据配比调稳定性是同一类工程手法。

## 疑问 / 下一步

- AgentNet 与 OpenCUA 模型/数据的确切开源地址、论文 arXiv 编号：字幕只说已开源、NeurIPS 2025 接收，待从论文或项目页补录。
- 讲者自陈的两个未解点值得继续追：history 压缩会丢中途信息（跨多个文档取信息的任务），以及 OSWorld 分数与真实用户体验的脱节——新评测怎么设计。
- UltraCUA 在 15 步预算 29.7→41 的细节（SFT 数据构成、RL 算法）讲者只是转述，且 50 步分数他自己不确定；若后续要引用应查 UltraCUA 原文。

## 原文金句（1-2句）

> 「我们希望给大家构建一整套这个 CUA 的 Open foundations。」——讲者给出本工作的定位（字幕 10:56–11:08）

> 「其实可以用一些 RL 方法去缩小这个 gap，提升它的 pass@1 能力。」——讲者由 pass@1/pass@N 差距分析得出的判断（字幕 33:08–33:18，按干净口径转写）

> 「数字环境本身就是为人的视觉去构建的这些环境，所以说视觉可能是一个不可或缺的存在吧。」——Q&A 中被追问 AI 是否必须用视觉交互时的回答（字幕 46:41–47:11，节录）
