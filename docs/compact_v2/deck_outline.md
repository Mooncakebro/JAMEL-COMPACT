# COMPACT 分享 Deck 总控大纲（9 页 + 附录）

> 定稿标题：**COMPACT：一个模型，压缩记忆、预期未来、做出决策 —— 流式场景下的自压缩行动者**
> 叙事主线：**判据与评分表（页 2）→ 审判现有方法（页 3-4）→ 判据落地为架构（页 5）→ 三模块逐一兑现（页 6-8）→ 实验全绿呼应（页 9）。**
> 贯穿暗线：**不确定度 P** —— 页 2 作为"按需"的度量提出，页 5 给出生命周期，页 6 产生（Predict）、页 7 消费（Kalman 增益）、页 8 沉淀（注入的记忆带不确定度）。
> 图资源索引：
>
> | 文件 | 用途 | 页码 |
> |---|---|---|
> | `docs/compact_v2/method_overall_dataflow.drawio` | 整体结构数据流总览 | 页 5 |
> | `docs/compact_v2/method_modules_zoom.drawio` | 三模块 zoom 图（3 个 page） | 页 6-8 |
> | `docs/compact_v2/layer_l_token_flow.drawio` | token 级单层数据流（备询） | 页 5 备用 / 页 10 附录 |
> | `docs/compact_v2/fig_main_results.pdf` | 主结果柱状图（CVPR 风格） | 页 9 |
> | `docs/compact_v2/fig_ablation.pdf` | 消融实验表格（CVPR 风格） | 页 9 |
> | `docs/compact_v2/fig_per_app.pdf` | Per-app 明细柱状图（备选） | 页 9 附录 |
> | `docs/compact_v2/slide2_memory_taxonomy.drawio` | 记忆方法分类图 | 页 3 |
> | `docs/compact_v2/slide3_latent_memory_compare.drawio` | 隐状态记忆三行对比 | 页 4 |
> | `docs/compact_v2/fig1_zero_init.png`、`fig2_decision.png`、`fig3_forgetting.png` | 零初始化曲线、决策准确性、遗忘对比 | 页 8（模块③）/ 页 9（实验） |
>
> ⚠️ drawio 待同步：`method_modules_zoom.drawio` 模块①页需补 P̂ 递推（λ_l 层级 0.70/0.85/0.95）；模块③页的"λ₁<λ₂<λ₃ 层级条"需改为**双层级**（λ_l 管"记多久"浅低深高、w_l 管"用多少"浅高深低）——代码事实见 `jamel_compact/config.py:35-40`。
> 🎨 CVPR 风格图表生成脚本：`scripts/make_slide_figures.py` —— 修改其中的数据数组后重新运行即可更新所有图表。

---

## 页 1：标题页

**文案**

- 主标题：COMPACT：一个模型，压缩记忆、预期未来、做出决策
- 副标题：流式场景下的自压缩行动者（Streaming Memory Compression with a Self-Compressing Actor）
- 汇报人 / 日期
- 可选一行 hook：*一个模型，同时是记忆的压缩器和行动的决策者。*

**转场句**：要理解为什么需要"自压缩"，先看看长程交互里上下文会发生什么。

**图来源**：无图或放一张极简 hero 图（一条无限增长的 token 流被一个漏斗压成 K 个槽位）。prompt：*"Minimalist hero illustration, dark background: an endless stream of colorful tokens flowing from left, passing through a glowing funnel, compressed into a small fixed row of 4 gem-like memory slots on the right, flat vector style, teal and orange accent colors."*

---

## 页 2：Why Memory? —— 需要记忆，更需要"好"的记忆

**文案（四段递进：Why → 朴素解 → 判据 → 性质）**

**① 为什么需要记忆 —— 没有记忆，每一步都是"初见"**
- 无记忆的 Agent：不知道哪些动作已经试过、哪条路径是死胡同、上一步造成了什么后果
- 步数一涨（50+ 步长程任务），这种"金鱼式"决策便难以为继

**② 朴素的记忆：把全部历史塞进上下文**
- 现状两派：**状态塞上下文**（上下文随步数线性膨胀 → 注意力稀释、成本爆炸）；**检查点重放**（交互不连续，中间状态全丢）
- 结论：记忆必须被**压缩成固定形态** —— 但"压缩"只是手段，"压缩" ≠ "好"

**③ 什么是"好"的记忆？—— 三个判据**
- 常见答案"对决策有用即可"（L_act）**必要而不充分**：模型可依赖浅层相关性（投机特征），而非真正记住历史
- 三个正交判据 —— 问法统一是："什么情况算**违反**它？"
  - **Faithful（忠实）**：任意时刻，记忆都应是**全部历史**的忠实压缩 —— 不是只记住开头，也不是只记住最近
  - **Predictive（可预测）**：给定记忆与当前动作，应能**预期下一帧观测** —— 压缩必须保留对未来有信息量的内容
  - **Useful（可用）**：记忆必须被决策真正消费，而非摆设
- 三者的交集：**记忆即世界模型** —— 我们不为"记住过去"而设计记忆，而为"预期未来"而设计记忆

**④ 判据 ⟹ 性质（本场评分表，与页 3 四列一一对应）**
- **Faithful ⟹ 容量有界**：历史任意长，append 必溢出、剪枝必删除 —— 唯有"固定容量 + 覆盖式更新"装得下任意时刻的全部历史
- **Predictive ⟹ O(1) 在线更新**：预测每一步都要做 ⟹ 信念必须随步增量维护，不能每步重编码整段历史
- **Predictive ⟹ 按需遗忘**：预测必会出错 ⟹ 错误先验必须可被覆盖纠正，否则误差永久累积、越预测越偏
- **Useful ⟹ 按需查询**：决策只需与当前子任务相关的片段 ⟹ 记忆必须支持选择性读取，而非全文灌入
- 而"按需"需要一个度量 ⟹ **不确定度 P**：知道哪里不确定，才知道该查什么、该忘什么 —— 这是 COMPACT uncertainty-aware 设计的哲学源头（它在页 6-8 反复出现，请记住这个符号）
- （Useful 的另一半 —— 消费不能以毁基座为代价、训推必须一致 —— 是对**构建方式**的约束而非记忆本身的性质，由页 5 模块③兑现）

**转场句**：这四条性质就是我们的评分表。接下来用它审判现有方法 —— 剧透：没有一家能及格。

**图来源**：左图右柱布局（或上下两栏），两个 prompt：

- 左：困境对比图 —— *"Split comparison infographic. LEFT: 'Naive memory': a long context window stuffing growing history steps t1..t50, bar chart beside it showing context length exploding linearly, red warning icons. RIGHT: 'Bounded memory': a neat fixed-size row of memory slots updated step by step, flat green line chart, clean minimal style, soft blue and grey palette, clear Chinese labels."*
- 右：三柱图 —— *"Minimalist flat illustration: three pillars supporting a roof labeled 'Good Memory = World Model'. Pillars labeled 'Faithful', 'Predictive', 'Useful'. Foundation stone labeled 'Decision'. Below the pillars, four small checkmarks labeled 'bounded capacity / O(1) update / forget-by-need / query-by-need', plus a small gauge icon labeled 'uncertainty P'. Clean academic poster style, teal and slate palette."*

---

## 页 3：相关工作 ① —— 三类主流路线的答卷

**文案**

- 评分表（页 2 推导的四条性质）：**容量有界**、**O(1) 更新**、**按需遗忘**、**按需查询** —— 看三类主流记忆路线各自能拿几分

| 路线 | 代表 | 容量有界 | O(1) 更新 | 按需遗忘 | 按需查询 |
|---|---|---|---|---|---|
| 文本记忆 | MemoryBank / Reflexion | ✗ 无限增长 | ✓ | △ 启发式删除 | ✗ 全文检索 |
| 参数化记忆 | 微调 / LoRA 注入 | ✓ | ✗ 需训练 | ✗ 灾难性遗忘 | ✗ 不可查 |
| 隐状态/Token 记忆 | RMT / MemoryLLM / ICAC | ✓ | ✓ | △ | △ |

- 只有隐状态/Token 路线接近全部满足 → 我们站在这一条线上，继续往下挖

**转场句**：隐状态路线内部也有高下之分 —— 看下一页的两个"前辈"差在哪。

**图来源**（drawio / AI 绘图 prompt 二选一或都做）

- drawio 结构建议：三列卡片（Textual / Parametric / Hidden-State），每列顶部一个图标（文档/齿轮/芯片），下方 4 行性质用 ✓/✗/△ 着色（绿/红/黄），最右一列高亮边框。
- prompt：*"Academic comparison slide illustration, three vertical cards side by side on white background: card 1 'Textual Memory' with a document icon, card 2 'Parametric Memory' with a gear icon, card 3 'Hidden-State Memory' with a chip icon highlighted with a glowing teal border; each card has 4 rows of small check/cross marks in green and red; flat minimal style, consistent iconography."*

---

## 页 4：相关工作 ② —— 隐状态/Token 记忆内部：两个前辈，两个短板

**文案**

- **(a) 独立压缩器**（如 ICAC）：一个额外模型把历史压成 memory token
  - 短板（违反 **Useful** + 训推一致）：**压缩与决策分离** —— 压缩器不知道决策需要什么（压缩目标 ≠ 决策目标），决策模型训练时也从未见过真实分布的压缩记忆
- **(b) 压缩器 + 后压缩**（如 MemoryLLM 系的 pooling 思路）：压完再池化降 token 数
  - 短板（违反 **Faithful**）：**后压缩是无监督的信息丢失** —— 丢掉的恰恰是少数但关键的决策线索，历史不再被忠实保留
- **(c) ours — COMPACT**：**压缩器与决策器是同一个模型**（self-compressing actor）—— Useful 的架构答案：压缩目标与决策目标天然对齐；训推一致如何做到？留到页 7 揭晓

**转场句**：一句话总结 —— 别人是"压缩给决策用"，我们是"决策者自己压缩"。下面看整体结构。

**图来源**

- drawio 结构建议：三行横向流程图，每行左起"历史 token 流"→ 中间模块 → 右侧"决策模型"。行 (a) 中间是独立方框"Compressor"（与决策模型断开，虚线）；行 (b) 中间两个串联方框"Compressor → Pooling"，信息丢失处画红色 ✂；行 (c) 只有一个大圆角方框标注"Unified Model (Compress + Act)"，高亮。
- prompt：*"Three horizontal pipeline diagrams stacked vertically, academic style. Row A: history tokens flow into a separate 'Compressor' box, dashed arrow to a separate 'Policy Model' box, a crack icon between them. Row B: history tokens into 'Compressor' then a second 'Post-Pooling' box with red scissors marking information loss, then policy. Row C: history tokens flow directly into one single highlighted rounded box labeled 'Unified Compressor + Actor'. Clean flat design, teal highlight on row C, grey for A and B."*
- 注：(c) 行留给页 5 的总览图也行，本页 (a)(b) 两行画清楚即可。

---

## 页 5：整体结构 —— COMPACT 总览：压缩即行动，行动即压缩

**文案**

- **一句话架构**：COMPACT = **基座 VLM 主干** + **每层一组记忆槽（内容 M + 不确定度 P）**。记忆随时间被压缩、被校准，按层级强度注入决策
- **判据 → 性质 → 模块：一页看懂"为什么长这样"**

| 判据 | 性质（页 2 评分表） | 落地 |
|---|---|---|
| **Predictive** | O(1) 在线更新 + 按需遗忘 | 模块① Predict（动作条件状态转移）＋ 模块② Correct（校准写入）—— 同一判据的一体两面 |
| **Faithful** | 容量有界 | 记忆形态：固定 K 槽 side memory；模块② 以预测残差写入新观测信息 |
| **Useful** | 按需查询 + 不毁基座（构建约束） | 模块③ Zero-Born Steering（零初始化门控 + 层级化注入上限 w_l） |

- 三条注释（如被问）：
  - 模块①②是 Predictive 判据的**一体两面**：Predict 回答"基于记忆与动作，下一步应看到什么"，Correct 用"实际看到什么"纠偏 —— 预测-校正构成闭环
  - **不确定度 P 的生命周期**（页 2"按需"的度量如何落地）：Predict 相**产生** —— P 随转移增长，上一步越惊讶膨胀越多；Correct 相**消费** —— Kalman 增益 K = P̂/(P̂+R)，越不确定的槽位写入越多，校正后 P 收缩；Steering 相**沉淀** —— 注入决策的记忆，其不确定度由 P 显式刻画
  - 损失各有归属：**L_obs → 模块①**（预期未来），**L_nll / L_mem → 模块②**（不确定度校准 / 幅度约束），**L_act → 模块③**（服务决策）
- **数据流（一个时间步 t，分相执行）**：
  1. 上一步动作 a_{t-1} 到达 → **Predict** 相：每层记忆槽做动作条件转移，内容 M̂ 与不确定度 P̂ 同步演化
  2. 新观测 o_t 到达 → **Correct** 相：观测与 M̂ 对齐，按 Kalman 增益 K = P̂/(P̂+R) 写入槽位，P 收缩
  3. 决策时 → **Steering** 相：查询向量从槽位取回信息，经零初始化门控、按层级注入上限 w_l 注入，输出 a_t
- 训练信号：辅助损失 **L_obs**（观测预测）/ **L_nll**（不确定度校准）/ **L_mem**（幅度约束）+ 任务损失 **L_act**
- 三个模块分别对应记忆的 **转移（Predict）→ 更新（Correct）→ 注入（Steering）** —— 下三页逐一 zoom in

**转场句**：先看记忆如何"动起来" —— 模块①，状态转移。

**图来源**

- 主图：`docs/compact_v2/method_overall_dataflow.drawio`（已画好：时间步 pipeline、Predict/Calibrated-Update/Steering 三相着色、四损失出口、页 6-8 zoom 导航标注；详细版见附录页 10）
- 若需 AI 绘图风格版 prompt：*"Technical architecture diagram, dark slate background: a horizontal timeline of an agent-environment loop. Top lane: environment emits observation image + axtree each step. Middle: a stack of transformer layers, each layer group owning a small glowing memory slot bank; three phases color-coded — blue 'Predict' (action-conditioned state transition), green 'Calibrated Update' (observation write-back), orange 'Steering' (top-down injection into action head). Arrows show one full step cycle. Minimal, precise, paper-figure style."*
- 备注：本页图也是后续三页的"底图" —— 每页 zoom 图左上角放该图的缩略版 + 高亮当前模块位置（见 `method_modules_zoom.drawio` 的 locator 设计）。

---

## 页 6：模块① —— 记忆的状态转移：Action-Conditioned Predict（记忆如何"动起来"）

**挑战**：静态记忆不随动作演化，与世界状态脱节；预测缺乏不确定度刻画，写入强度无据可依；传统行为监督仅覆盖动作输出，记忆学什么无从约束。

**我们的方法**：**动作条件驱动的演化式记忆状态转移设计（Action-Conditioned Predict）**

1. **动作条件转移**【Predictive · O(1) 在线更新】：将动作编码为控制信号，经 FiLM-GRU 驱动记忆逐步转移，实现记忆与环境状态的同步演化。
2. **不确定度共演化**【Predictive · 按需遗忘的度量】：显式建模记忆的不确定度，与记忆内容双轨转移，实现浅层短期、深层长期的多时间尺度记忆。
3. **可监督预期**【Predictive 判据落地】：以观测预测损失监督转移后的记忆，使"预期未来"成为可优化目标，为记忆学习提供步骤级监督信号。

**一句话**：先预期、后校正 —— 预测-校正架构把"学记忆"从单一行为监督拆解为"预测世界 + 纠正先验"两个更容易的梯度问题（Kalman 式归纳偏置）；而 P 就是连接预测与校正的桥梁。

**转场句**：记忆带着内容和不确定度做出了预测 —— 下一页看世界如何用真实观测回应它。

**图来源**：`method_modules_zoom.drawio` 的 page "模块① Predict"（已画好：观测前后双相时间线、L_obs 监督、表示隔离图注；左上角带整体结构 locator）。⚠️ 待补：P̂ 递推公式与 λ_l 层级（0.70/0.85/0.95）图注。

---

## 页 7：模块② —— 观测驱动的记忆更新：Calibrated Correct（记忆如何"写进去"）

**挑战**：观测写入缺乏"信记忆还是信观测"的原则性权衡，写入强度由固定规则给定；写入内容在各槽位间无区分度，历史细节被均质化丢失。

**我们的方法**：**不确定度驱动的校准式观测写入设计（Kalman-Gated Calibrated Correct）**

1. **Kalman 增益写入**【Predictive · 按需遗忘】：以先验不确定度与观测噪声之比标定写入强度，不确定度越高写入越多、写入后收缩，实现可学习的按需遗忘。
2. **逐槽位创新**【Faithful · 忠实写入】：以先验记忆为查询对观测做交叉注意力，各槽位按需检索差异信息生成创新项并写回先验，实现观测信息向记忆的忠实沉淀。

**一句话**：Predict 负责"我以为会看到什么、有多大把握"，Correct 负责"实际看到了什么" —— 按不确定度加权的残差写入记忆，信念滚动更新。

**转场句**：记忆存好了、不确定度也校准了，决策时怎么用回去？下一页，注入。

**图来源**：`method_modules_zoom.drawio` 的 page "模块② Calibrated-Update"（已画好：μ̂ 对齐虚线、L_nll/L_mem 出口；⚠️ 图中"噪声注入"元素需删除 —— v2 无此机制）。⚠️ 待补：K = P̂/(P̂+R) 增益门图示与 P 收缩标注。

---

## 页 8：模块③ —— 记忆注入：Zero-Born Steering（记忆如何"用起来"）

**挑战**：外挂记忆模块在训练初期即冲击预训练表示，基座能力退化；各层记忆需求不均，一刀切注入失衡；静态 prefix 与滚动演化的历史脱节。

**我们的方法**：**零初始化生长的层级化记忆注入设计（Zero-Born Steering）**

1. **零初始化注入**【Useful · 不毁基座】：注入投影与门控从零初始化，训练起点严格等价于基座前向，记忆贡献按数据需求生长，消除基座退化。
2. **层级化注入配额**【Useful · 按需查询】：按层设定注入上限并由门控学习实际强度，实现注入位置与强度的按需分配。
3. **演化记忆注入**【Useful · 决策真正消费】：注入随预测-校正滚动更新的记忆状态（替代静态 prefix），注入路径受任务损失直接监督，保证记忆被决策消费。

**一句话**：以不伤害基座为硬约束，让记忆在需要的地方、以需要的强度参与决策。

**备询 Q&A（Q：这样做和直接往 prompt 里加前缀有什么本质区别？）**
A：三点。① 前缀是静态的，steering 注入的是随每步 predict-correct 更新的记忆状态；② 前缀对所有层一视同仁，我们是层级化配额 w_l + 零初始化门控，强度和位置都由数据驱动；③ 前缀没有改变训练目标，我们的注入路径被 L_act 直接监督 —— 记忆是否被决策消费，是有梯度保证的。

**转场句**：三个模块合起来，每条性质都有人兑现 —— 最后用实验说话。

**图来源**：`method_modules_zoom.drawio` 的 page "模块③ Steering"（⚠️ 待改：原"λ₁<λ₂<λ₃ 层级条"改为**双层级条** —— λ_l 0.70/0.85/0.95（记多久）+ w_l 0.8/0.5/0.3（用多少），α=0 零初始化门控）；可叠加 `fig1_zero_init.png`（零初始化 vs 直接注入曲线）作右下角小图。

---

## 页 9：实验 —— 评分表，逐条兑现

**实验设置（开场 30 秒，讲清 test10 是什么）**：评测基于 **JAMEL**（Tian et al., 2026）—— 用代码覆盖率等确定性新颖信号联合训练 agent 记忆与探索策略的 GUI 基准：**86 个 web 应用训练、10 个 held-out 应用（test10）评测泛化**，指标 = 每应用平均覆盖率奖励。COMPACT 继承其 coverage-weighted SFT 训练范式，仅替换记忆机制 —— 与 Baseline 的唯一变量就是记忆。

### 幻灯片结构（本页拆为 3 张 slide）

#### Slide 9a：主结果（Main Results）—— 占 70% 面积

**上方标题**：COMPACT 在未见应用上显著超越同等规模基线

**主图（居中大面积）**：`fig_main_results.pdf`（已生成，同目录附 .png）
- 横轴按基座分两组（Qwen3-VL-2B / Qwen3-VL-4B，组下标注参数量 2.13B / 4.02B），组内两根柱：Baseline（SFT，灰色斜线）vs COMPACT（teal 实色）
- 纵轴：test10 平均每应用奖励
- 柱顶数值标签；COMPACT 柱内标注参数开销（+6.24M / +9.79M，均 <0.3%）
- 组内橙色箭头标注增益：+2.1 (+13%) / +3.5 (+22%)

**图右侧或下方关键洞察 bullet**：
- Baseline 2B→4B 几乎不涨（15.9→15.8）：瓶颈在记忆，不在模型容量
- COMPACT-2B（+0.29% 参数）18.0，反超 Baseline-4B —— 好记忆 > 大模型
- COMPACT-4B 达到 19.3，接近 JAMEL-9B 的 20.7
- 推理时 KV-cache 长度不变，无额外推理延迟

**演示话术（9a）**："先看左边这组对照 —— Baseline 从 2B 放大到 4B，奖励几乎原地不动（15.9 → 15.8），说明瓶颈根本不在模型容量，而在记忆机制。COMPACT 只加不到 0.3% 的参数：2B 就拿到 18.0，直接反超 4B 基线；4B 到 19.3，已经摸到 JAMEL-9B（20.7）的肩膀。这就是评分表上 Useful 一条的兑现 —— 记忆被决策真正消费了。"

**来源**：`scripts/make_slide_figures.py` → `docs/compact_v2/fig_main_results.{pdf,png}`

#### Slide 9b：消融实验（Ablation Study）

**上方标题**：每个组件都在起作用

**居中表格**：`fig_ablation.pdf`（已生成，同目录附 .png）

| Configuration | Avg. Reward (test10) |
|---|---|
| Baseline-2B (Qwen3-VL-2B SFT) | 15.9 |
| COMPACT-2B w/o memory writing | 16.8 |
| COMPACT-2B w/o auxiliary losses | 14.5 |
| **COMPACT-2B (full)** | **18.0** |

**表格下方解读 bullet（口语化，讲解时展开）**：
- **去掉记忆写入**：16.8（+0.9）—— 记忆只读不写，退化为"带偏置的 prompt"，仍有微弱提升
- **去掉辅助损失**：14.5（低于 Baseline）—— 无 L_obs/L_nll 约束，记忆学到投机捷径而非忠实压缩，污染基座
- **完整 COMPACT-2B**：18.0（+2.1）—— 三判据需同时满足

**演示话术（9b）**："消融回答两个问题。第一，记忆只读不写，18.0 掉到 16.8 —— 在线更新（O(1)、按需遗忘）贡献大头；剩下的 +0.9 说明注入路径本身也学到一点东西，但只是'带偏置的 prompt'。第二，拿掉辅助损失，直接跌破基线到 14.5 —— 没有 L_obs 的观测监督与 L_nll 的不确定度校准，记忆学到投机捷径，反而污染基座。三个判据缺一不可 —— 这不是锦上添花，是生死线。"

**来源**：`scripts/make_slide_figures.py` → `docs/compact_v2/fig_ablation.{pdf,png}`

#### Slide 9c（附录定位）：与 JAMEL 的对比

**一句话**：JAMEL（9B 独立记忆模型）→ 20.7；COMPACT-2B（2.13B 自压缩）→ 18.0
- JAMEL 用独立 9B 模型做记忆压缩，COMPACT 把压缩器嵌入决策模型自身
- 额外参数：COMPACT +6.24M（0.29%）vs JAMEL 完整 9B
- 训推一致：压缩器 = 决策器，无分布失配

> **JAMEL 简介（1 句过渡）**：JAMEL（Tian et al., 2026）首次提出用代码覆盖率等确定性新颖信号训练 agent memory —— 探索行为触发覆盖增长，覆盖增长提供记忆监督。COMPACT 继承此训练范式（coverage-weighted SFT），但将外挂式压缩器替换为内置 self-compressing actor。

---

### 实验数据（硬数据）

**主结果**：
| Model | Avg. Reward (test10) | Parameters |
|---|---|---|
| Baseline-2B (Qwen3-VL-2B) | 15.9 | 2.13B |
| Baseline-4B (Qwen3-VL-4B) | 15.8 | 4.02B |
| **COMPACT-2B** | **18.0** | 2.13B + 6.24M (+0.29%) |
| **COMPACT-4B** | **19.3** | 4.02B + 9.79M (+0.24%) |

**消融实验（2B）**：
| Configuration | Avg. Reward |
|---|---|
| Baseline-2B | 15.9 |
| COMPACT-2B w/o memory writing | 16.8 |
| COMPACT-2B w/o auxiliary losses | 14.5 |
| COMPACT-2B (full) | 18.0 |

**参考基线（JAMEL paper）**：
| Method | Model | Avg. Reward |
|---|---|---|
| JAMEL | JAMEL-9B | 20.7 |
| ReAct-vision | Gemini 3.1 Flash-Lite | 20.9 |
| MAI-UI | MAI-UI-8B | 8.4 |

---

### 一句话结论

COMPACT 是唯一在评分表上全绿的方法：Faithful（bounded）、Predictive（O(1) 在线 Kalman）、Useful（零初始化不伤基座），以 < 0.3% 参数开销换 +2.1~3.5 绝对增益。

**转场句**：记忆即世界模型 —— 当压缩器与决策者合为一体，压缩即行动，行动即压缩。谢谢。

**图来源**：
- 主结果柱状图：`docs/compact_v2/fig_main_results.{pdf,png}`（已由 `scripts/make_slide_figures.py` 生成）
- 消融表格：`docs/compact_v2/fig_ablation.{pdf,png}`（同上脚本，已生成）
- 可选 per-app 明细：`docs/compact_v2/fig_per_app.pdf`（待生成 —— 需要 per-app 数据后在同一脚本中补一个函数）

---

## 页 10：附录（备询，不主动讲）

- token 级单层数据流详图（`layer_l_token_flow.drawio`）
- 不确定度 P 的完整机制：λ_l / w_l 双层级（`jamel_compact/config.py:35-40`）、Kalman 增益动力学（`COMPACT_V2_METHOD.md` §3 的 UI 突变仿真：e  spike → P̂ 膨胀 → K 跳到 0.93 → 覆盖写入 → 重新收缩）
- v1→v2 改进对照表（`COMPACT_V2_METHOD.md` 开头对照表）
- 损失权重、训练配置、超参表

---

## 章节划分与 agenda 页

**叙事节奏（为什么这样分）**：页 2 一次性给出"为什么需要记忆 + 什么是好记忆 + 评分表"，页 3-4 用评分表审判现有方法，页 5-8 用同一张评分表兑现我们的设计，页 9 回收。全场的逻辑骨架就是**一张评分表的"立 → 破 → 立"**，暗线是**不确定度 P 从哲学度量到机制实现**的落地过程。

**章节结构（4 个 Part）**

| Part | 章节名 | 页码 | 功能 |
|---|---|---|---|
| 1 | **为什么需要记忆，什么是好记忆？** | 页 2 | 动机 + 设计哲学 + 评分表 |
| 2 | **现有方法差在哪** | 页 3-4 | 用评分表审判三类路线 + 隐状态路线内部 |
| 3 | **我们的答案：COMPACT** | 页 5-8 | 总览 + 三模块（转移 → 更新 → 注入） |
| 4 | **效果与验证** | 页 9 | 评分表全绿回收 |

**Agenda 页（放在标题页后，可选）文案**

- 为什么需要记忆，什么是好记忆？—— 一张评分表
- 现有方法差在哪 —— 用评分表审判
- COMPACT：压缩即行动 —— 总览与三个模块
- 实验：评分表全绿

**agenda 图来源**：纯文字版式即可；若要视觉化，用一条横向进度条分 4 段，当前章节高亮（每个 Part 开头可复用同一版式做 section divider）。
