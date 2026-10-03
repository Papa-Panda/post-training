# MEETUP-RLITE — RLite：用 20 行代码从头写 RL
> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/MEETUP-rlite.html

> 「我希望传递一个 idea：这事情非常非常简单，keep things simple。」——讲者收尾判词（字幕 16:56–17:06，干净转写）

## 元信息

- 活动归属：2025-08-24「LLM RL&RL Infra」线下 Meetup（青稞社区）
- 合集：sid=6759789
- 标题：RLite: 用20行代码从头写RL
- BV：BV1AkkoBgEXr
- B站链接：https://www.bilibili.com/video/BV1AkkoBgEXr/
- 时长：约 1105 秒（字幕末条 18:23）
- 提炼日期：2026-10-02
- 分享嘉宾：张涵（字幕结尾主持人致谢作「张涵老师」；讲者本人在字幕中未自报姓名与单位，用字待核，见「疑问 / 下一步」）
- 相关论文：无（字幕未提论文；提及 o1、OpenRLHF、verl、Hugging Face Transformers、vLLM、PyTorch FSDP）
- 相关代码：RLite（讲者自述的开源学习框架；字幕未给仓库地址）
- 字幕原文存档：本地 `transcripts/MEETUP-RLITE.txt`（457 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（已存档）清洗提炼，提炼方式为字幕实录；凡数字均标来源为「字幕」。AI 字幕识别误差较多：RLite 在字幕中作「ALLIE / online / r line」，verl 作「VRL」，vLLM 作「VM」，OpenRLHF 作「open2HF」，`nn.Module` 作「N点module」，`.generate` 作「点GENERRATE」，PagedAttention 疑作「pay attention」。个别专名与 API 名以讲者原意校正，无法确证处在正文与文末疑问中标明，不做推测性补写。

## 一句话总结

讲者主张 LLM RL 的算法与 infra 本身并不复杂：RLite 不发明新接口，而是复用 Hugging Face Transformers、vLLM 与 PyTorch 原生的 `nn.Module` / `.generate` / `.meta` / `.cpu` / `.cuda` 语义，只在底层替换执行方式（FSDP 并行、vLLM 生成、权重同步），使一个 GRPO 训练循环能以接近单进程脚本的写法完成，并可经一个 hook 扩展到 multi-turn rollout。

## 核心

### 背景：为什么另写一个框架

讲者开场先表明定位：前面嘉宾讲得偏 high level，他只推荐自己写的一个开源学习框架 RLite，并说它与前一位博士（字幕作「傅伟博士」，用字未确证）刚才提到的东西一致——都是一个非常底层的设计，和「脚手架式的、非常全的」框架不同（字幕 00:32–00:54）。

他给了一条很短的个人时间线，用来支撑后面的核心论点——RL infra 与 RL 算法其实很简单，投入的时间并不多也能从头写出框架（字幕 02:08–02:27）：

1. 去年 10 月（按活动时间 2025-08-24 推，为 2024 年 10 月；字幕未明说年份）开始接触 o1 一类工作后，发现用已有框架写 RLVR 的 workflow 非常困难，萌生了写 RLite 的想法；
2. 当时内部先基于 OpenRLHF 探了一版；
3. 自己这一版写完之后注意到 verl，在它只有两三百个 star 时就给了 star，算是很早的关注者；
4. RLite 在 5 月初开源（字幕作「5月份、5月初」），此后一直没有空做社区维护与 PR，时间线上的灰色部分是被其他工作占用的时间。

这个背景的言下之意是：难点不在算法本身，而在已有框架把简单 workflow 写复杂了。

### 原型：GRPO 只有四步

讲者回顾最初的心路：看到 o1 出来后的第一反应是先把一个 workflow 写出来。他当时刚开始写 LLM，对 vLLM、OpenRLHF 都不熟，但写过 Hugging Face Transformers，于是直接拿 Transformers 写了一套，并断言用 Transformers 的 interface 很容易写出 RL 代码（字幕 02:33–03:02）。

他把 GRPO 的核心拆成四步（字幕 03:02–03:46）：

1. 初始化模型，与传统 `nn.Module` 完全一致；
2. 准备数据：一些 prompts，每个 prompt 有一个 answer，再定一些选项，例如每个 prompt rollout 8 次；
3. 直接调用 Hugging Face 的 `.generate`，得到很多个 rollout；
4. 拿这些 rollout 算 GRPO 的 scores，算完就能算 loss——按他的简化说法，loss 无非是做了 shift 以后的 log-prob 乘上 score。

按字幕口径把第 4 步写成式子，其中模型参数记为 ， $\theta$ ，、prompt 记为 ， $x$ ，、已生成前缀记为 ， $y_{\le t}$ ，、该位置的 score（优势类信号）记为 ， $A$ ：讲者口中的单步损失可示意为：

$$\mathcal{L}(\theta) = -\log p_\theta(y_{t+1} \mid x, y_{\le t}) \cdot A$$

需要强调：这是讲者在现场给的简化式，只表达「shift 后的 log-prob 乘 score」这层意思；GRPO 完整的 group 内归一化、importance ratio、clip 等细节在字幕中均未展开，不能把上式当作完整 GRPO 目标。

### 真正的瓶颈：慢在 generate，不在算法

讲者随即自问：为什么这么简单的东西，整个开源社区 struggle 了大半年？他的回答是：第 3 步的 `.generate` 很慢（字幕 03:49–04:17）。原因是训练侧的推理没有很多优化——例如 PagedAttention（字幕作「pay attention」）、专用推理 kernel——不加上这些，效率非常低；而训练与推理的计算 pattern 又不一样，已有框架没有把这两个功能整合在同一个进程里，使一个进程既能做 `model.generate` 又能做 `model.forward`。

他给了一句概括：如果 Hugging Face 的 `.generate` 能和 vLLM 跑得一样快，「这个世界会变得非常美好」（字幕 04:36–04:41）。这句话是全场设计的出发点：问题在执行效率，不在接口。

### RLite 的设计：不新增接口，只换底层

讲者由此给出 RLite 的设计原则（字幕 04:43–05:08）：做 RL infra 设计时不要再设计新接口，现有的 interface 已经足够好；RLite 的代码「就和 Hugging Face 的代码长得差不多」，所谓 20 行就能写一个 GRPO——他同时自承这「稍微有点标题党」，因为一些与框架无关的函数细节没有写进这 20 行。

具体做法分三块：

**训练侧：还是 `nn.Module`，只多一个钩子。** 在传统 CV / NLP 写法里，模型要么 `from_pretrained` 载入，要么自己改一点 forward 逻辑。在 RLite 里它变成一个 RLite 的 `nn.Module`，框架帮用户删掉 boilerplate，例如告诉后端 FSDP 应该怎么切分、什么叫一个 block。去掉这些后，用户只关心 forward，而 forward 直接继承 Transformers 已定义好的实现即可——社区的 kernel、装饰器、monkey patch 都能直接作用在 Transformers 的 forward 上，把效率提上去。FSDP 需要知道的信息，Transformers 里已有类似 non-splittable modules 的 tag 可以复用（字幕 05:13–06:35）。

在此之上，RLite 只增加一个 interface：一个在模型真正落到 GPU、实例化出来之后调用的钩子（字幕作「with materialized」，确切 API 拼写待查代码），主要做一件事——绑 optimizer 与 learning rate schedule（字幕 06:41–07:02）。有了它，用户就可以像写普通 `nn.Module` 一样定义任意函数：算一个 loss、做一次 backward、再做一次 optimizer step，就是一个 train step；写好之后放回前面的四步里，GRPO 就完成了（字幕 07:04–07:28）。

**推理侧：rollout 就是 vLLM 的 `.generate`。** rollout model 与 vLLM 的生成代码很像：把 inference engine 传进去，它会在后面调起 vLLM 的 model 并完成初始化，不需要很多代码；generation 直接调 `.generate`，接口与 vLLM 一模一样。讲者的原话是：RLite 没有任何新增接口，全用的是 Transformers 接口和 vLLM 接口，而 vLLM 本身也复用了 Transformers 接口，所以本质上只有一套 Hugging Face 接口（字幕 07:30–08:05）。生成之后再调一个算 reward 的函数，那是与框架无关的用户代码。

**参数搬运与分布式：逻辑 module 与执行分离。** 生成与训练之间靠一组与 PyTorch 原生语义一致的设备转移函数衔接：`.cpu` 是把参数挪到 CPU，`.cuda` 是挪回 GPU，`.meta` 则是 PyTorch 2.2 还是 2.3 新引入的 meta device（讲者现场对版本号不确定），等于把所有参数扔掉——因为下一步会把最新参数同步过来，所以可以扔；想留着算别的东西就用 `.cpu`（字幕 08:16–08:53）。

核心区别在于：`nn.Module` 在这里只是一个逻辑载体，所有计算逻辑都挂在它的某个函数里，而它本身不需要有参数。于是可以把参数初始化为 meta device 上的空参数，这样的 module 可以用 Ray 传来传去；传过去之后同样复用 `from_pretrained` 构造一个无参数 module，再把参数传进去，像 FSDP 一样用一个包装函数一包，它就变成可并行计算的 FSDP module，用户定义的每个 method 都自动变成并行计算（字幕 08:54–10:18）。用户使用时完全不用关心后面的 execute 是什么：字幕口径是后端走 FSDP2，FSDP1 将被 deprecated，后续还要接一个新后端（字幕作「TTTEN」，疑为 TorchTitan，待核）并支持 PP（pipeline parallel）；带 PP 时前端有一点额外限制，但整体上这份代码「和单进程代码没有任何区别」（字幕 10:18–10:51）。

权重同步被他快速带过：先 `.cuda`，经 IPC 做转换（字幕作「把位置拿到 CUDA 上、用 IPC 做位置转换」，疑为权重/参数的 ASR 误差），`.cpu` 卸载，再把 KV cache 在 GPU 上重新实例化，整个过程就完成了（字幕 10:51–11:16）。他的结论是：可以看到，20 行能够写一个 GRPO，「没有特别复杂」（字幕 11:16–11:23）。

### 观点：接口复用决定框架生死

后半段讲者把设计选择上升为一个明确观点（字幕 11:23–14:09）：RL infra 与算法没有特别多新东西，如果现有系统的效率能达到模型训练所需的效率，用已有 interface 就够了——我们每天用的这些 interface 完全足够表达想做的所有事情，只是下面藏了很多魔鬼细节，而用户完全感知不到。RLite 做的是同样的事：复用上面所有接口，只是下面做的事情发生了一些变化。对用户而言，心智负担会变得非常少。

他用框架史作类比：TensorFlow 以快出名，PyTorch 以好用出名，MXNet 想既快又好用，最后都没做好而死掉；他转述李牧来做 talk 时的判断，落点也是 interface design——MXNet 的接口不如 PyTorch 简洁好，是使用者越来越少的原因。增加一般用户的学习成本，会日积月累地导致用户流失（字幕 12:21–13:18）。

落到个人诉求，他说得很直白：在现在百花齐放的时代，他只是想自己用着舒服一点，用 Hugging Face 这一套写法就能跑得很快，又不需要学很多接口。因为老实说，他自己写完之后都已经忘记里面怎么实现了——细节写过一遍就会慢慢忘记，但 interface 在 PyTorch 能用、在 Hugging Face 能用，将来哪怕 RLite 死掉、PyTorch 更新一版，还是同样的 interface，这段时间就没有做额外的学习、没有浪费精力，只是 focus 在核心算法设计上（字幕 13:19–14:09）。这是本场最完整的价值主张：**接口是可迁移资产，实现细节不是。**

### 例证：multi-turn rollout 只多一个 hook

最后的例子是当天被反复提到的 multi-turn rollout（字幕 14:11–16:48）。讲者先强调 vLLM 接口本身的简洁：给 model 一个 prompts 加 sampling parameters，得到一个 list of request outputs——丢给它一些字符串，它补全以后丢回来；chat template 怎么拼用户都不用管。

要支持多轮交互时，他的做法不是改这套输入/输出，而是在同样的简洁 interface 下多加一个 hook：用户额外要做的一件事，只是告诉系统什么时候应该继续生成、或者做完什么事情之后继续生成。生成出的内容经过 hook 处理后，由 hook 判断是继续生成还是结束；hook 里可以做比较昂贵的事情，例如 reward 计算，或者调用远程服务（字幕作「jin server」，待核）等任何操作。整个系统建在 Ray 上，天生是异步的（字幕 15:49–16:01）。

于是用户侧的输入/输出几乎没有变化：从原来输入 prompts 与 sampling parameters、拿到一个 list of request outputs，变成加了一个 hook 以后，拿到一个 list of list of request outputs——每个 prompt 可能有多个轮次，用户拿到的是每个轮次的 request output。讲者的总结是：这样他就没有任何额外需要记忆的事情；他 respect 社区前人做的 interface design，以至于自己完全不用做任何 interface design（字幕 16:03–16:48）。

主持人致谢后，讲者应要求又补了两张 slide（字幕 17:20–18:23）：其一，它和单进程一样的原因是 engine 就在 main process 里，用户只有一个进程，他提到 OpenRLHF 也差不多是这样的设计，各类框架的区别其实在于这部分执行放在哪里；其二，用这套东西在 500 行以内、400 多行就能复现一个内部工作（字幕作「zero 的 RHVR」，确切所指待核），而这 400 多行里有 200 行都是 configure、log 与 IO 操作，核心可能就 100 行左右——他说这是给大家的一些 evidence。

## 关键数字总表

| 指标 | 数值 / 口径 | 来源 |
|---|---|---|
| 视频时长 | 约 1105 秒（字幕末条 18:23） | 字幕 |
| 字幕条数 | 457 条 | 字幕 |
| GRPO 核心步骤 | 4 步（初始化、备数据、generate、算 score 与 loss） | 字幕 |
| 示例 rollout 次数 | 每个 prompt 8 次 | 字幕 |
| 标题代码量 | 20 行写一个 GRPO（讲者自承略标题党，与框架无关的函数细节未计入） | 字幕 |
| verl 早期关注节点 | 两三百个 star 时讲者已 star | 字幕 |
| RLite 开源时间 | 5 月初（字幕作「5月份、5月初」，年份按活动时间推为当年） | 字幕 |
| meta device 引入版本 | PyTorch 2.2 还是 2.3（讲者现场未确定） | 字幕 |
| 复现内部工作的代码规模 | 500 行以内、400 多行 | 字幕 |
| 其中 configure / log / IO | 约 200 行 | 字幕 |
| 其中核心代码 | 约 100 行 | 字幕 |

## 可迁移

- 先原型后优化：用 Hugging Face `.generate` 把 GRPO 四步先原样写通、验证算法与 reward 逻辑，再替换执行层（vLLM 生成、FSDP 并行）。这与讲者的亲身路径一致，也避免一上来就被大框架的配置面淹没。
- 接口纪律可作为选型指标：评估一个 RL 框架时，先数它要求用户新学的接口数量。RLite 的答案是零新增接口加一个钩子；凡要求重学一套数据/模型/生成抽象的框架，都应要求它给出与之相称的收益理由。
- 逻辑与执行分离的写法值得借鉴：用 meta device 上的无参数 `nn.Module` 作为可经 Ray 传递的逻辑载体，落到执行端再注入参数并包 FSDP——用户代码与并行方式解耦，换后端（FSDP 版本、PP）时前端不动。
- multi-turn 的扩展方式：不改 rollout 的输入/输出契约，只加一个由用户控制的 hook 来决定续写或终止，并把 reward 计算、远程环境调用放进天然异步的 hook 里；单轮代码几乎不用改就能升多轮。

## 疑问 / 下一步

1. **讲者姓名用字与身份未确证**：字幕中讲者未自报姓名，结尾主持人致谢作「张涵老师」；本仓库 EP100 海报名单另有「张晗：RLite 作者、Founder of Yiven AI」。「涵 / 晗」用字与单位信息均待以活动官方材料核对，本纪要按字幕口径记为「张涵」并在此标明。
2. **多处专名与 API 名为 ASR 音译，未确证**：开场提到的博士字幕作「傅伟 / 付维」；钩子名作「with materialized」；后续要接的后端作「TTTEN」（疑为 TorchTitan）；权重同步中的「位置」疑为权重/参数；multi-turn hook 里可调用的「jin server」所指不明；结尾「zero 的 RHVR」具体指哪项工作不明。这些都需要对照 RLite 仓库代码与活动材料逐项核对。
3. **代码量口径无法审计**：20 行 GRPO 与 400 多行复现都未包含与框架无关的函数、配置与 IO，且字幕未给仓库地址；行数与实际跑通的性能（生成加速比、训练吞吐）无从在本纪要内验证。
4. **GRPO 式子的完整性**：现场只给了 shift 后 log-prob 乘 score 的简化式，group 归一化、ratio/clip、KL 项均未提及；RLite 实现里这些部分如何处理，待查代码。
5. **与 OpenRLHF / verl 的关系未展开**：讲者说内部先基于 OpenRLHF 探了一版、且 OpenRLHF 的进程设计「也差不多」，但 RLite 相对这两者具体删掉了什么、保留了什么，没有给出对照表。

## 原文金句

> 「假如说 Hugging Face 的 .generate 就和 vLLM 跑的一样快，这个世界会变得非常美好。」——字幕 04:36–04:41（干净转写）

> 「RLite 没有新增任何接口，全都用的是 Transformers 接口和 vLLM 的接口……本质上来说就只有一个 Hugging Face 的接口。」——字幕 07:46–08:05（干净转写，略去口语重复）

> 「这些细节写过一遍以后可能就会慢慢忘记，但是这些 interface 在 PyTorch 也能用、在 Hugging Face 也能用……我没有浪费额外的精力，只是 focus 在核心的算法设计上。」——字幕 13:36–14:09（干净转写，略去口语重复）
