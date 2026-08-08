# COMPACT 分享 Deck 总控大纲 v4(12 页主线 + Backup 备询)

> 定稿标题:**COMPACT:一个模型,压缩记忆、预期未来、做出决策**
> 副标题:**让智能体压缩自己的记忆**
> 叙事主线:**范式转移"从无状态预测器到有状态学习者"(页 2)→ 核心洞见"记忆即预测"(页 3)→ 三个根本问题(页 4)→ 路线之争 External vs Native(页 5)→ 核心闭环一图(页 6)→ 三个回答(页 7-9)→ 验证(页 10)→ 五层演进路线图(页 11)。**
> 设计原则:问题驱动;标题即论断;主流程零公式,全场至多 **M(记忆内容)/ P(不确定度)** 两个符号;机制细节全部进 Backup(放在 Thank You 后、同一文件内)。
> 图资源索引:
>
> | 文件 | 用途 | 页码 |
> |---|---|---|
> | `docs/compact_v2/method_overall_dataflow.drawio` | 核心闭环图(需简化至四步中文标注 + M/P) | 页 6 |
> | `docs/compact_v2/method_modules_zoom.drawio` | 三个回答页配图(简化版) | 页 7-9 |
> | `docs/compact_v2/slide3_latent_memory_compare.drawio` | 外挂 vs 原生对比 | 页 5 |
> | `docs/compact_v2/slide2_memory_taxonomy.drawio` | 记忆路线详表 | Backup F |
> | `docs/compact_v2/fig_main_results.{pdf,png}` | 主结果(已由 `scripts/make_slide_figures.py` 生成) | 页 10 |
> | `docs/compact_v2/fig_ablation.{pdf,png}` | 消融表(已生成) | Backup B |
> | `docs/compact_v2/fig1_zero_init.png` | 零初始化曲线(小图) | 页 9 |
> | `docs/compact_v2/layer_l_token_flow.drawio` | token 级详图 | Backup F |

**AI 画图统一规范(每页提示词 = STYLE + 该页主体描述;图中一律无文字,标签后期在 PPT 里加)**

```
STYLE = "Flat vector illustration, minimalist academic style for a conference talk. Color palette: navy #1B2A4A, teal #2B7A9E, coral #E8735A, green #48BB78, light gray #F0F2F5 background. No text, no letters, no numbers in the image. Clean composition, generous whitespace, 16:9 aspect ratio."
```

---

## 页 1:标题页

- 主标题:COMPACT:一个模型,压缩记忆、预期未来、做出决策
- 副标题:让智能体压缩自己的记忆
- 汇报人 / 日期
- 开场 hook(口头):"今天只想讲一句话 —— **记忆即预测**。好记忆不是把过去存下来,而是能预期未来;而且这个记忆不该挂在模型外面,就该是智能体自己。"

**转场句**:先看这件事为什么现在成了整个范式的瓶颈。

**AI 提示词**:STYLE + "A humanoid agent silhouette seen from the side; inside its head a glowing spiral compressing inward, suggesting memory folding into itself; a faint horizon line of a digital world in front of it."

---

## 页 2:大背景 —— 从无状态预测器,到有状态学习者

**文案(普适,不提 GUI)**

- 今天 AI 的成功,本质上是**无状态预测器(stateless predictor)**的成功:给定上下文,预测下一个 token、下一帧、下一个动作 —— 交互一结束,一切归零
- 但智能体要长期生存:持续探索、积累经验、越用越强 —— 它必须是**有状态学习者(stateful learner)**
- 核心矛盾:**无状态的架构 × 有状态的需求**;暴力解法(全量上下文)成本线性膨胀、注意力稀释,窗口一满性能坍塌
- 关键问题:**智能体能否在部署之后,继续从自身的交互中学习?**

**演讲提示**:"今天的模型像金鱼 —— 每次交互都是初见。这不是工程问题,是范式问题:架构是无状态的,需求是有状态的。"

**转场句**:要补上这个"状态",先得想清楚 —— 什么样的记忆才算"好"?

**AI 提示词**:STYLE + "Split scene. Left: a robot with a circular reset arrow cycling above its head, same scene repeating, suggesting amnesia. Right: the same robot walking up an ascending path, collecting glowing fragments along the way, suggesting accumulation."

---

## 页 3:核心洞见 —— 人脑不是录像机:记忆即预测

**文案**

- 学术背书(Metis):**"记忆本质上是一个预测问题"** —— 记忆存在的意义,是支撑对未来的预期
- 人脑佐证:人不会记住今天看到的每一帧画面,但能**预测**下一秒会发生什么 —— 大脑只存"意外",不存"全部"
- 常见答案"对决策有用就行"必要而不充分:模型可以靠浅层相关性投机取巧,而非真的记住历史
- 好记忆的三条判据:
  - **Faithful 忠实**:任意时刻都是全部历史的忠实压缩
  - **Predictive 前瞻**:记忆让我能预见行动后果,而非仅回溯过去
  - **Useful 够用**:记忆为当下决策服务,而非无差别堆积

**演讲提示**:"传统记忆系统在做'录像':把历史压短、存起来。我们的出发点是把记忆从'回顾'变成'预期' —— 录像机越存越满,预测者越存越省。"

**转场句**:把这三条判据落到工程上,就是三个绕不开的根本问题。

**AI 提示词**:STYLE + "A human head silhouette in profile; an eye looking at a stream of many scene frames flowing by; inside the brain only a single small spark is kept, the spark projecting a faint image of the next frame forward."

---

## 页 4:三个根本问题

**文案(每问一行,先不给答案)**

- **Q1 演化:记忆如何随行动演化?** —— 静态存储,还是随动作不断推演?
- **Q2 写入:什么信息值得被写入?** —— 全量写入,还是只写"预期之外"?
- **Q3 融合:记忆如何与推理融为一体?** —— 外挂模块,还是原生能力?

**演讲提示**:"这三个问题不是我们发明的工程需求,是任何'会记忆'的智能体都躲不开的基本问题。我们的三个设计,就是这三个问题的回答。"

**转场句**:回答之前,先看现有路线把记忆放在了哪 —— 尤其是 Q3。

**AI 提示词**:STYLE + "Three large elegant question marks arranged in a row, each standing on a small distinct pedestal (a gear, a filter funnel, two interlocking rings); equal visual weight, symmetrical layout."

---

## 页 5:路线之争 —— External Memory vs Native Memory

**文案**

- **External(记忆在模型外)**:文本记忆、RAG、外部向量库;独立 compressor 路线同样是外挂 —— 记忆由**另一个模型**生产
  - 读写与决策是两套系统,各说各话;检索噪声;prompt 膨胀;额外模块、额外计算图
- **Native(记忆是模型自己)**:记忆的读写、更新、注入,全部发生在模型**自身的前向传播**里 —— 模型本身就是记忆
- 一句话点透:**现有方法靠"压缩历史"腾空间;我们靠"预测未来"省更新**

**演讲提示**(直接回应"你们不也是 compressor 吗"):"关键区别不在压缩得好不好,而在记忆放在哪。外挂路线的记忆在模型之外 —— 另一个模块、另一条计算图。我们没有这个东西:模型本身就是记忆。"

**转场句**:那它内部怎么运转?一张图讲完。

**AI 提示词**:STYLE + "Split comparison. Left: a robot carrying an oversized external hard disk on its back, connected by tangled cables to its head, suggesting an awkward add-on. Right: the same robot with a small glowing core embedded inside its chest, self-contained, calm posture."(drawio 备选:`slide3_latent_memory_compare.drawio`)

---

## 页 6:核心机制 —— 先预测,再更新:只记"意外"

**文案(四步闭环,每步一句话)**

1. **预想(Anticipate)**:根据当前记忆 + 刚执行的动作,先推演"下一步会看到什么"
2. **对比(Compare)**:实际观测 vs 预想,得到"惊讶程度"
3. **写入(Write the Surprise)**:惊讶大 → 多写;惊讶小 → 少写 —— 容量只留给新信息
4. **决策(Act)**:基于更新后的记忆,输出下一步动作

- 闭环中心一句话:**记忆越准,需要记的越少**
- 图上只出现两个符号:记忆内容 **M**、不确定度 **P**

**演讲提示**:"就像走路:你不会记住每一步踩到哪块砖,但踩空一级台阶立刻记住 —— 这就是'只记意外'。"

**转场句**:这张图的三个角,正好回答那三个根本问题。

**AI 提示词**:STYLE + "A clean circular loop of four nodes arranged in a ring (an arrow pushing a block forward; two overlapping frames with a difference spark; a gate letting a small fragment through; a chess piece moving); in the center of the ring a small bright core; the loop suggests perpetual motion."(drawio 备选:`method_overall_dataflow.drawio` 简化重绘:四节点中文环形流程图,中心写"记忆越准,需要记的越少";公式、损失名、λ/w 全部移除)

---

## 页 7:Q1 的回答 —— 好记忆不只会回忆,更会预期

**挑战(直觉版)**:世界在变,记忆不变 —— 你刚关掉的弹窗,静态记忆还以为它开着。

**我们的方法**:让记忆随动作先"预想"一步,并用"能否预测下一帧观测"直接监督它

- 动作作为控制信号,驱动记忆状态向前推演 —— 记忆从"过去的快照"变成"对未来的预期"
- 预测质量受直接监督:想错了会被纠正 —— 步骤级监督第一次落到记忆上(传统行为监督只约束动作)

**转场句**:记忆会想了,下一个问题 —— 观测来了,写多少?

**AI 提示词**:STYLE + "A rectangular memory block being pushed forward along a timeline by an arrow labeled with a small action icon; ahead of the block a translucent ghost frame of the near future; a check mark where ghost meets reality."(drawio 备选:`method_modules_zoom.drawio` 模块①页简化版)

---

## 页 8:Q2 的回答 —— 知道"不知道",才知道"写什么"

**挑战(直觉版)**:全信观测会抖,全信记忆会漂 —— "写多少"需要一把尺子。

**我们的方法**:显式建模记忆的**不确定度 P**,由它决定"信记忆还是信观测"

- 记忆越确定 → 越信自己 → 少写;观测越意外 → 越信观测 → 多写(经典滤波思想,但强度是学出来的,不是手调的)
- 先验记忆主动从观测中检索差异信息,逐槽位写回 —— 固定容量内保住更多历史细节

**转场句**:记忆会演化、会更新了,最后一个问题 —— 怎么放回模型里,不打扰它思考?

**AI 提示词**:STYLE + "A balance scale: left pan holds a solid stable block (memory confidence), right pan holds a sparkling new fragment (observation novelty); the tilted beam controls the opening of a small gate below, through which a proportional amount of fragments falls into a container."(drawio 备选:`method_modules_zoom.drawio` 模块②页简化版,P 高 → 大门开 → 多写)

---

## 页 9:Q3 的回答 —— 记忆"长"进模型,不打扰思考

**挑战(直觉版)**:给大脑做手术,病人不能死在手术台上 —— 外挂记忆一训练就拖垮预训练基座。

**我们的方法**:让记忆注入通路**从零开始生长**

- 训练起点 = 纯基座,记忆的影响随数据需求自然生长 —— 全程零退化(右下角 fig1 小图)
- 每一层自行决定"借多少记忆":按层配额、门控学习 —— 记忆在需要的地方、以需要的强度参与决策

**转场句**:三个问题回答完,用实验说话。

**AI 提示词**:STYLE + "A tall stack of layered translucent slabs (neural network layers); between some layers tiny green sprouts growing from zero, gradually thickening upward; the stack itself remains perfectly stable and undisturbed."(drawio 备选:`method_modules_zoom.drawio` 模块③页简化版 + `fig1_zero_init.png` 小图)

---

## 页 10:实验 —— 好记忆 > 大模型

**实验设置(一句话)**:基于 **JAMEL** 基准(Tian et al., 2026)—— 86 个应用训练、**10 个从未见过的应用(test10)**评测泛化,指标 = 每应用平均覆盖率奖励;**与 Baseline 的唯一变量就是记忆**。

**主结果**(图 `fig_main_results.pdf`,占 70% 面积)

- Baseline 2B→4B 几乎不涨(15.9→15.8):**瓶颈在记忆,不在模型容量**
- COMPACT-2B(+0.29% 参数)18.0,反超 Baseline-4B —— **好记忆 > 大模型**
- COMPACT-4B 19.3,接近 JAMEL-9B 的 20.7

**演讲话术**:"左边这组对照最关键 —— 模型放大一倍,奖励原地不动。我们只加不到 0.3% 的参数,2B 反超 4B 基线。消融还有一句佐证(图在备份页):拿掉记忆监督,直接跌破基线 —— 好记忆的判据缺一不可。"

**实验硬数据(存档)**

| Model | Avg. Reward (test10) | Parameters |
|---|---|---|
| Baseline-2B (Qwen3-VL-2B) | 15.9 | 2.13B |
| Baseline-4B (Qwen3-VL-4B) | 15.8 | 4.02B |
| **COMPACT-2B** | **18.0** | 2.13B + 6.24M (+0.29%) |
| **COMPACT-4B** | **19.3** | 4.02B + 9.79M (+0.24%) |

参考基线(JAMEL paper):JAMEL-9B 20.7 / Gemini 3.1 Flash-Lite (ReAct-vision) 20.9 / MAI-UI-8B 8.4。
消融(Backup B):w/o memory writing 16.8 / w/o auxiliary losses 14.5 / full 18.0。

**转场句**:最后两分钟,说说这条线真正值钱的地方。

(本页用真实实验图,无需 AI 提示词)

---

## 页 11:愿景 —— 五层演进:从有状态,到自我进化

**文案(阶梯式路线图)**

- **L1 有状态 ✅**:模型拥有内生记忆 —— COMPACT 已实现
- **L2 自主管理记忆 ✅**:自己决定记什么、记多久 —— COMPACT,当前位置
- **L3 经验学习 🔜**:从成功与失败中提炼策略 —— 结合 JAMEL 的持续探索
- **L4 持久认知 🔭**:记忆沉淀为世界模型 —— 跨 session、跨任务
- **L5 自我进化 🔭**:探索与记忆成为预训练的原生能力 —— Agentic Pretraining

**收尾金句**:"COMPACT 不只是一个记忆模块 —— 它是让 Agent 从'工具'走向'自主进化生命体'的底层基础设施。压缩即行动,行动即压缩。谢谢,欢迎拍砖。"

**AI 提示词**:STYLE + "An ascending five-step staircase rising from lower-left to upper-right; the bottom two steps glow warmly (achieved), the middle step half-lit (near), the top two steps outlined faintly with a distant star above the last one; a small agent figure climbing."

---

## 页 12:Thank You

- 联系方式 / code 链接(如有)

**AI 提示词**:STYLE + "The glowing spiral core from the title page, now calm and steady, centered on a plain background; subtle sense of an open door or horizon behind it."

---

# Backup Slides(Thank You 之后,同一文件;被问到才翻)

## Backup A:模型整体架构

- `method_overall_dataflow.drawio` 完整版:Predict/Correct/Steer 三相 + 四个训练信号出口
- 一句话架构:基座模型 + 每层一组记忆槽(内容 M + 不确定度 P)

**AI 提示词**:STYLE + "A wide horizontal pipeline diagram of abstract geometric modules (rounded rectangles and rings) connected by arrows, three color-coded phases repeating across a layered stack; a side branch showing four small signal outlets."

## Backup B:完整消融(`fig_ablation.pdf`,已生成)

| Configuration | Avg. Reward (test10) |
|---|---|
| Baseline-2B | 15.9 |
| w/o memory writing | 16.8 |
| w/o auxiliary losses | 14.5 |
| **COMPACT-2B (full)** | **18.0** |

- 记忆只读不写 → 16.8:在线更新贡献大头
- 拿掉辅助损失 → 14.5(跌破基线):没有观测监督与不确定度校准,记忆学到投机捷径 —— **没有预测监督的记忆是有害的**

## Backup C:机制公式 —— Predict(记忆状态转移)

- M̂ = GRU-FiLM(M, a_{t-1})(动作作控制信号)
- P̂ = λ_l⊙P + Q(a) + γ_e·e_{t-1}:Q 可学习、惊讶 e 驱动膨胀;λ_l 初始化 0.70/0.85/0.95(浅→深,可学习)—— 浅层短期、深层长期
- L_obs:从 M̂ 预测下一步观测

## Backup D:机制公式 —— Correct(校准式写入)

- 创新项 ΔM = CrossAttn(Q=M̂, K,V=Z):先验记忆逐槽位检索观测差异(k=4 潜查询掩码池化)
- Kalman 增益 K = P̂/(P̂+R);M = M̂ + K⊙ΔM;P = (1−K)⊙P̂ —— 越不确定写得越多,写完收缩
- L_nll(对照真实惊讶校准 R)/ L_mem(幅度约束)
- Kalman 动力学仿真:UI 突变 → P̂ 膨胀 → K 跳到 0.93 → 覆盖写入 → 重新收缩(`COMPACT_V2_METHOD.md` §3)

## Backup E:机制公式 —— Steering(零生长注入)+ 备询 Q&A

- 注入 W↑ = 0、门控近零 → 训练起点严格等于基座前向
- 层级配额 w_l 浅 0.8/中 0.5/深 0.3,实际强度门控学习(`jamel_compact/config.py:35-40`)
- λ_l 管"记多久"、w_l 管"用多少"
- **Q:和往 prompt 里加前缀有什么本质区别?** A:① 前缀静态,注入的是随 predict-correct 滚动更新的记忆状态;② 前缀各层一视同仁,我们是层级配额 + 零初始化门控,位置与强度由数据驱动;③ 前缀不改变训练目标,我们的注入路径被 L_act 直接监督 —— 记忆是否被消费有梯度保证

## Backup F:判据推导 + 路线详表 + 其他

- 判据 ⟹ 性质:Faithful ⟹ 容量有界(append 必溢出、剪枝必删除);Predictive ⟹ 更新恒定(逐步预测)+ 按需遗忘(错先验须可覆盖);Useful ⟹ 按需查询(决策只取相关片段);"按需"的度量 ⟹ 不确定度 P
- 三类记忆路线详表:`slide2_memory_taxonomy.drawio`(文本 / 参数化 / 隐状态)
- token 级单层数据流:`layer_l_token_flow.drawio`
- v1→v2 改进对照、训练配置与超参:`COMPACT_V2_METHOD.md`

**AI 提示词**(Backup F  taxonomy 页备选):STYLE + "A three-column comparison chart rendered as abstract geometric cards: left card a scroll of paper (text), middle card a dense cube of small bricks (parameters), right card a flowing ribbon folded into a compact coil (hidden state); the right card subtly highlighted."

---

## 章节划分与 agenda 页

| Part | 章节名 | 页码 | 功能 |
|---|---|---|---|
| 1 | **范式转移** | 页 2-3 | 从无状态到有状态 + 记忆即预测 |
| 2 | **三个根本问题** | 页 4-5 | 提问 + 路线之争 |
| 3 | **我们的回答** | 页 6-9 | 一图闭环 + 三个回答 |
| 4 | **验证与愿景** | 页 10-11 | 数字 + 五层路线图 |

**Agenda 页文案(可选)**:范式转移:从无状态预测器到有状态学习者 / 洞见:记忆即预测 / 三个根本问题与我们的回答 / 实验与五层演进路线

---

## 演讲铁律(给大 boss 汇报)

1. **前 90 秒不碰架构**:只讲"所有 Agent 都面临从'无状态'到'有状态'的范式转移"和"人脑记忆的本质是预测不是录像"
2. **不念符号**:不说"卡尔曼增益",说"根据不确定度决定写多少";不说"RNN 状态转移",说"记忆必须随动作演化,所以模型先'想象'行动后果"
3. **每页不超过 30 秒**:超过就是太复杂(老板原话:"看不懂就没有交流价值")
4. **被问细节先给逻辑、再翻 Backup**:"核心逻辑是记忆按需查询,数学形式化在备份页"
5. **结尾必须拔高**:停在五层路线图的 L3-L5(经验学习 → 持久认知 → Agentic Pretraining),不停在消融数字上

## 反馈对照(8 条 → 本版落点)

| 反馈 | 对策 | 落点 |
|---|---|---|
| ① 副标题像模型不像智能体 | "让智能体压缩自己的记忆" + 开场 hook"记忆即预测" | 页 1 |
| ② 不讲 GUI,讲普适 | 升维为"无状态预测器 → 有状态学习者"范式转移,全篇以"智能体"为主语 | 页 2 |
| ③ "不也是 compressor 吗" | 升维为 External vs Native 路线之争,"记忆是模型自己" | 页 5 |
| ④ 模块页太复杂 | 拆为 Q1/Q2/Q3 三个回答页,每页一句挑战 + 两句机制 + 一图 | 页 7-9 |
| ⑤ 符号太多 | 主流程零公式,至多 M/P 两个符号 | 页 6-9 |
| ⑥ 英文标题晦涩 | 全中文标题,标题即论断 | 全局 |
| ⑦ 挑战抽象 | 挑战改为直觉场景(弹窗 / 抖动漂移 / 做手术) | 页 7-9 |
| ⑧ less is more | 12 页主线 + Backup 备询(同文件,Thank You 后) | 全局 |

**v4 相对 v3 的增量(qwen 第二版已吸收)**:页 2 升维为范式转移框架;页 3 加入 Metis 引文"记忆本质上是一个预测问题"作学术背书;页 5 从"取消压缩器"升维为 External/Native Memory 概念体系;页 11 从三条方向改为 L1-L5 五层演进路线图(收尾金句"底层基础设施");每页补齐统一风格的 AI 画图提示词(含 Backup A/F)。
