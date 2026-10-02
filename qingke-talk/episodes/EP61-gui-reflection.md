# EP61 — GUI-Reflection：让多模态 GUI 智能体获得反思纠错能力的训练框架

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP61-gui-reflection.html

> 「如果使用现在这种 error-free 的数据 SFT 之后的 GUI 模型，在每一步的 thinking 过程中，他都不会对上一步的行为进行真正的验证。」——讲者点出公开轨迹数据的结构性缺陷

## 元信息

- 期号：61
- 标题：GUI-Reflection：让多模态 GUI 智能体获得反思纠错能力的训练框架
- BV：BV1SA4CzWEph
- 时长：01:00:39（讲授约 46 分钟 + Q&A 约 12 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：吴鹏浩（南洋理工大学 MMLab 博士；主持人介绍口径，字幕中实验室作「m m lab」；本工作为南洋理工大学 MMLab 与商汤研究院合作）
- 相关论文：GUI-Reflection（讲者称论文信息、数据与模型均在项目页公布；字幕未给出 arXiv 编号）
- 相关代码：项目页（地址待从论文页补录）
- B站链接：https://www.bilibili.com/video/BV1SA4CzWEph/
- 官网期号：61
- 字幕原文存档：本地 `transcripts/EP61.txt`（1085 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP61，已存档）清洗提炼，信息来源为 AI 字幕原文（经人工清洗；个别专名可能有识别误差，如 GUI 在字幕中作「guy / GY / JI」、InternVL3 作「interview3」、thought 作「salt」、press back 作「press spect / PRESPECT」、AndroidWorld 作「android word」，均以论文与公开资料为准）。实验部分的若干表格数字讲者在字幕中未逐项口播，以下只记讲者口头给出的结论与数字，不从幻灯片推测数值。

## 一句话总结

公开 GUI 轨迹数据几乎全是无错的成功轨迹，SFT 之后模型在每一步都默认「上一步是对的」，丢掉了自我反思与纠错行为——而这恰是后续 online RL 能否采到反思样本、决定其上限的前提。GUI-Reflection 是一个无需人工标注的自动化框架，把「发现错误、回退消除影响、吸取教训重做」这三种子能力分别做成预训练任务套件、离线 SFT 的合成反思数据、以及迭代式 online reflection tuning，自建移动端环境闭环注入模型。

## 核心

### 背景：GUI agent 的三条路线与共同盲区

讲者先给分类学：

1. **基于强基模型的 agentic 框架**。围绕 GPT-4o、Gemini、Claude 等强基模型，拿 XML / accessibility tree 理解页面，配复杂 prompt 与 workflow，多为精简版 ReAct（先 reasoning 再 action，reasoning 内含任务拆解、行为反思、进度追踪，可挂外部知识库）。代表如 AppAgent（执行前先探索、构建知识文档供执行时参考）、Mobile-Agent-V2（planning / decision / reflection 三 agent 加 memory 模块）。好处是直接借强基模型能力；代价是管线复杂、prompt 调试重、级联误差、token 与时间开销大。
2. **解耦 planning 与 grounding**。强基模型只做 planner、给出 low-level 指令，专门的 grounding 模型预测像素级动作，研究重心在 grounding 数据与模型。代表如 UGround（字幕作「You grant you ground」，跨平台海量 grounding 数据训出强泛化模型，配 GPT-4 规划）、Aria-UI（字幕作「ARRAUI」，把整体任务信息与操作历史也喂给 grounding 模型，预测更好）。
3. **端到端单一 GUI 模型**（本工作所属）。直接从截图预测像素级 action，最像人操作设备。主流范式是：通用多模态大模型 → GUI pretraining（注入 grounding、页面理解、元素功能知识）→ 用轨迹数据做 offline SFT（模仿学习）→ 部分工作再做 post-training / online learning。代表如 SeeClick、OS-Atlas（字幕作「c click os aos」）一系；SOTA 的 UI-TARS 进一步整合大量 pretraining 与 SFT 数据，用 online bootstrapping 靠人工标注纠错与反思数据迭代微调，再用正负样本对做 DPO post-training。

盲区在第三类的 SFT 数据：现有公开 GUI action 数据集几乎全是人采集的成功轨迹，中间步骤几乎 error-free。在这种数据上训完，模型在每一步的 thinking 里都会输出「上一步操作正确、符合预期」式的 thought，丧失自我验证。这带来两个后果：真实部署中环境分布与训练不同、模型一犯错就无法自知自纠；更深一层，**反思行为决定 online RL 的上限**——RL 阶段几乎采不到「主动承认错误」的样本，online training 就无法有效进行。UI-TARS 靠大量人工标注才拿到纠错/反思数据，其余方法基本没有这条数据通路。

### 方法一：预训练阶段的反思任务套件

讲者把「能反思纠错」拆成三个子能力，各配一个自动构造的预训练任务：

1. **Action Verification（发现错误）**。输入是动作前后的两帧截图和一个 action purpose 短语，模型判断该目的是否被成功完成（例：purpose 是「搜索已输入内容」则完成、是「follow 这个用户」则未完成，因为第二帧里点的是搜索键不是 follow 按钮）。关键设计：**故意不把实际执行的 action 给模型**。因为真实犯错时，模型正是凭自身知识和第一帧判断不出正确动作的，它必须像部署时一样，对照执行结果（第二帧）来验证预期是否达成。
2. **Action Reversal（回退消除影响）**。给前后帧截图与第一帧对应的 action，模型从多选题中选出正确的回退操作，消除该动作的影响。
3. **Mistake-informed Reattempt（吸取教训重做）**。用 low-level instruction 的 grounding 数据构造：先在其他数据上训一个 grounding 模型，在保留集上跑 inference，把它真实犯的错收集起来；构造时把错误预测作为输入并明确告诉模型「你之前有一次错误尝试」，要求它重新预测（例：目标是 favorite 按钮，第一次误点 dislike，第二次被告知后改对）。

套件的实验结论（字幕口径）：大规模闭源多模态模型与 Qwen2.5-VL-72B 本身就有可观的反思能力且能迁移到 GUI 域；小模型则很有限；而且只做传统 GUI pretraining（grounding + 理解）会让这方面能力进一步丢失，预训练阶段就加入反思数据可以很大程度保住并提升它。

### 模型架构：InternVL3-8B 上的四个输入与两个新动作

本工作的 agent 模型基于 InternVL3-8B（字幕作「interview3」），聚焦移动端。输入包含：过去 4 帧 + 当前帧截图、整体任务 instruction、memory 内容、action history（每步的 action description 与 atomic action）。输出三段：action thought（对上一步的反思、整体规划、当前思考）→ action description（固定格式，如「click 某个 element to 完成某个目的」）→ 具体 atomic action。动作空间含 click、长按、scroll、type、wait 与 press back/home/enter 等快捷键，并额外定义两个动作：

- **answer**：信息获取类任务（查天气、查信息）做完后，用它把获取到的答案以结构化格式返回，再标记任务完成。
- **memorize**：把执行中必须留存的信息（比如在小红书查到的攻略要点）显式记下来供后续步骤参考——因为只留 4 帧历史，几十步的跨 app 任务里远处信息会丢，必须靠这个动作补上长期记忆。

### 方法二：离线 SFT 阶段——没有犯错后截图，就合成犯错

offline SFT 的核心困难是：现有轨迹里根本不存在「错误动作执行后的截图」，反思数据无从标注。讲者设计了两种不需要真实错误截图的构造法：

1. **改写任务目标（goal rewriting）**。采样一条轨迹，随机取 step t，用多模态大模型改写任务 goal，使原本在 t 步的正确动作在新 goal 下变成一个「容易犯的、真实的」错误。例：原任务是打开 app 搜三人座沙发，t 步点搜索框本是对的；新 goal 改成「用 camera icon 以图搜图找沙发」后，点搜索框就成了错误。于是可以在 t+1 步构造反思 thought（意识到该用相机图标、刚才点错了），并让 t+1 的动作是 press back——由于回退后界面回到 t 步状态，可直接复用 t 步截图，再构造 t+2：总结错误、在新 goal 下做出正确尝试（点 camera icon）。
2. **插入无效动作（ineffective action insertion）**。不改 goal，在 step t 之前插入一个对屏幕没有任何影响的错误动作（点不可点击的背景元素、在页面底部继续下滑）。执行后页面不变，原截图可复用；把原本 t 步 thought 改写为对这个插入错误的反思。例：任务是查看商品 details，插入「点商品标题文字」（不可点击，无变化），t 步反思意识到该元素不可点、想看详情更应该 scroll down。

生成细节上有两个工程要点：改写 goal、错误动作与反思 thought 都由多模态大模型生成，但这类模型给不出像素级 grounded action——解法是先在已有数据上训一个 GUI action agent，标注时把大模型生成的 thought 与 description 作为它输出的前缀拼进输入，让它续生成 grounded action，再加一步一致性 filter（检查 grounded action 与 thought/description 是否相符）。Q&A 中讲者补了数据精度：早期构造框架产出的数据大约只有 80% 是好的，会出现 false positive、把错误反思学歪；他们在改 goal 后加了额外 check（验证原 action 在新 goal 下确实是错的），把数据精度提到约 95%，模型才比较稳地学会判断上一步是否真的错了。

### 方法三：迭代式 online reflection tuning

online learning 需要环境。讲者认为公开移动端环境任务模板过于简单重复，于是自建了一套：11 个 app、215 个任务模板，每个模板可随机实例化出大量具体任务，按难度与步数分两个等级。架构是分布式 host + worker：worker 只有 CPU、只跑安卓模拟器；所有 inference 与训练都在 host 上做——与以往把 inference 放 worker 的环境不同，因为现在的大模型参数量与推理开销大，worker 普遍没有足够显存与算力。验证器两路：程序式 verifier（直接读底层数据库与设备状态，精确判分）与多模态大模型 verifier（覆盖难以程序化验证的中间过程，可扩展为 stepwise verifier；Q&A 口径称其定位犯错步的精度在 90% 以上）。

迭代流程（每轮）：模型采样任务做 rollout；成功轨迹里筛掉犯错步、只留成功步；失败轨迹自动定位犯错步 t，标注两类纠正——pre-error correction（t 步本应执行的正确动作）与 post-error correction（错误执行后 t+1 步应做什么），若 t+1 是 press back 同样可复用截图构造 t+2。每轮结束后按各任务成功率动态调整下一轮的采样比例。课程安排上前三轮只训 easy 任务、初始权重均等，随后向成功率低的任务倾斜，三轮后再把仍低分的 easy 任务与 level 2 任务混合继续迭代。Q&A 给的收敛口径：简单任务（十步到二十步、乃至十步以内）大约 3–4 轮迭代就有较大提升并收敛；更难的（20–30 步）任务需要先学会基础操作，再叠加 3–4 轮。

### 实验与案例（字幕口径）

讲者口头给出的结论链条：offline SFT 加入反思数据后任务成功率有很大提升；再叠加 online reflection tuning 进一步提升；随 tuning 轮次增加，两个难度等级的任务性能都稳定上升；最终模型在 AndroidWorld（字幕作「android word」）在线评测中在同规模端到端 baseline 里结果很不错（字幕未逐项报表格数字，本纪要不臆测数值）。

两个行为案例能说明「学会了什么」：闹钟任务要打开某时段的闹钟，模型第一步误点闹钟时间、打开了编辑界面（本应点右侧开关），下一步它意识到自己点的是 alarm time 而不是开关，明确说需要点 cancel 回到之前界面——及时识别错误并消除影响。文件管理任务要在 app 里找文档导出 PDF，模型在入口误点「All files」banner（以为能进浏览界面，实际只是个 label、毫无反应），第二步意识到这只是个标签、正确操作应点「select file to open」按钮，随后成功进入文档界面。

### 未来方向（讲者自列）

有反思行为后做大规模多轮端到端 RL，并把 RL 与 supervised reflection tuning 结合：很长、很难的任务靠随机采样 + outcome reward 极难走通，做法是把模型卡住的点直接告诉它正确操作应是什么，避免它长期停滞；扩展到移动端之外的更多平台与数据；用生成式模型造犯错后的截图（直接生成页面截图，或生成页面代码再渲染），产出更多样、更真实的犯错—反思数据。

### Q&A 要点

- **怎么避免 agent 过分谨慎、总是回滚？** 关键在 action verification 这一能力的准确性（预训练任务正是为此设计）；且犯错类数据在 offline SFT 里占比相对较少，模型一般不会过分谨慎。进一步可用正负样本对做 preference learning，专门提升「上一步是否真错了」的判断。
- **自动构造数据会引入 false positive 吗？研究过精度/噪声影响吗？** 这是讲者团队踩过的坑：早期框架精度约 80%，模型的对错判断能力学得弱、常出现 false positive；加了改 goal 后的二次校验等流程后精度到约 95%，此后模型能较好学会判断。结论是构造数据的精度直接决定反思机制学得对不对。
- **online 阶段真实犯错数据好拿吗？怎么做 verify？** 好拿——随机采样的轨迹里很大一部分最终失败，失败轨迹必然在某一步犯了错；再用多模态大模型逐歩 check 定位犯错步，精度 90% 以上。
- **基座模型选什么？** 目前大家常用 Qwen2.5-VL 或 InternVL3 效果较好；若用现成 GUI 模型，直接在 UI-TARS 给出的模型上继续训练提升会更好。
- **课程学习怎么分级？** 只分两级：先让人执行一遍估计步数，再加启发式（是否需要中途记忆内容、是否含困难操作）人工定级；迭代中按成功率动态调采样权重，而不是固定分级表。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| 模型基座 | — | InternVL3-8B；输入过去 4 帧 + 当前帧截图 | 字幕（模型架构） |
| 自建 online 环境规模 | 公开移动端环境模板过于简单重复 | 11 个 app、215 个任务模板、2 个难度等级 | 字幕（online 环境） |
| 合成数据精度 | 早期构造框架约 80% 为好数据 | 加二次校验后约 95% | 字幕（Q&A） |
| online 犯错步定位精度 | 多模态大模型 stepwise check | 90% 以上 | 字幕（Q&A） |
| online 迭代收敛轮次 | 简单任务（≤10–20 步） | 约 3–4 轮迭代明显提升并收敛；难任务（20–30 步）再叠加 3–4 轮 | 字幕（Q&A） |
| 预训练套件现象 | 传统 GUI pretraining 后反思能力进一步丢失 | 加入反思任务数据后可保住并大幅提升 | 字幕（实验结论，未给逐项数字） |
| SFT / online 叠加效果 | error-free SFT 基线 | 加反思 SFT 成功率大涨，online reflection tuning 再进一步提升 | 字幕（实验结论，未给逐项数字） |
| AndroidWorld 在线评测 | 同规模端到端 baseline | 讲者称结果「很不错」 | 字幕（实验结论，未给数值） |

## 可迁移

- 对 RL / post-training 的直接启发：**online RL 之前先审计 SFT 数据的行为分布**。如果训练数据里从不存在「承认上一步错了」的样本，RL 阶段就采不到这类行为、探索不到纠错策略，性能上限被数据分布提前锁死。本讲把「验证—回退—重做」拆成三个可自动构造的监督任务，是把行为先验注入模型、再让 RL 放大的通用配方，对 coding agent 的调试/回滚行为同样适用。
- Infra 视角：这套 online 环境的 host/worker 分工值得抄——worker 只跑 CPU 模拟器、inference 与训练集中在 host，避免给每个环境节点配大显存；程序 verifier 与多模态 verifier 分工（能读状态就读状态、读不了再上模型判），外加按成功率动态重加权采样（本质是给 rollout 做 curriculum 调度），都是 agent RL 环境层可复用的设计。

## 疑问 / 下一步

- 讲者口头未给的数字：三个预训练任务的逐项准确率、SFT/online 各阶段在 AndroidWorld 上的确切成功率——需要查论文表格补录，本纪要只按字幕口径记结论。
- 「改写 goal 构造错误」造出的错误分布与模型真实部署时犯的错是否同分布，讲者没有直接论证；这决定合成反思数据的上限，是下一步最该看的消融。
- 讲者自陈的边界：当前只做了移动端；犯错后截图靠「回退复用」与「无效动作」绕开，真实改变界面的错误尚不能合成——他给的解法（生成式模型造截图）还未在本工作中落地。

## 原文金句（1-2句）

> 「如果使用现在我们这种 error-free 的数据 SFT 之后的 GUI 模型，在每一步的 thinking 过程中，他都不会对上一步的行为进行真正的验证，或者说是他总是输出一个认为上一步操作是正确的、并且符合预期的这样的一种 thought。」——讲者论证公开数据为何训不出反思行为（字幕 13:59–14:21，按干净口径转写）

> 「反思的这种行为，对于强化学习的阶段是至关重要的，并且一定程度上决定了最终这种 online training 性能的上限。」——讲者把数据问题接到 RL 上的关键判断（字幕 13:42–13:57，节录）
