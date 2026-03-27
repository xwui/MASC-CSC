# MASC-CSC 项目分析与 SCI 改造计划

---

## 一、 核心立意：我们到底解决了什么问题？

仔细对照您之前的 [focused_plan.md](file:///C:/Users/zhangxw/.gemini/antigravity/brain/79a28ef6-9980-4baf-a2b4-1d6af12e0634/focused_plan.md) 与本项目现有的文档，我们可以提取出一个非常明确的、直接可以放在 SCI 论文 Abstract 和 Intro 里的核心立意（Thesis）：

### 🚨 发现的痛点 (The Observation)
近年来，将大语言模型（LLM）应用于中文拼写纠错（CSC）引发了极大的关注，但**LLM存在严重的“过度纠正（Over-correction）”问题**。
具体来说，LLM 强大的语义推理能力让它可以改写句子、替换近义词，却**缺乏对中文错别字产生机制的感知**（LLM在分词和推理时，不知道“平果”和“苹果”是拼音相似，也不知道“己”和“已”是字形相似）。因此，当LLM面对一个可能有错的句子时，它常常把它当成“开放式润色”任务，从而把原本只是微小拼写错误的句子，改写成了另一种语义表达。

### 💡 我们的解法 (The Solution)
**让 LLM 获得“错误机制感知（Error Mechanism Awareness）”能力。**
我们不是简单地把 LLM 和传统模型（如BERT）的输出做概率混合（像现有的 MSLLM 或 Corrector-Verifier 那样）。我们是**把 NamBert 作为一个“多模态特征感官”外挂给 LLM**。
因为 NamBert 拥有特化的 `PinyinEncoder` 和 `GlyphEncoder`，它能敏锐地察觉到“这里是一个**音近/形近**错误”。我们将这个机制约束（Mechanism Constraint）以“动态混淆集”或“机制类型特征”的形式注入到 LLM 的生成过程中（即 *Multimodal Confusion-Constrained Decoding, MCCD*）。

**一句话总结我们的故事**：
> 现有的 LLM 纠错方法忽略了中文字符错误的物理机制（音形）。我们通过引入多模态传统模型（NamBert），**首次让 LLM 具备了错误机制感知能力**。在此指导下，LLM 的纠错范围被严格限制在符合发生机制的候选集中，从而从根本上解决了大模型的“过度纠正”问题。

---

## 二、 当前代码现状与 SCI 论文目标的 Gap 分析

我完整阅读了目前 `F:\MASC_CSC\docs\` 下的文档和项目源码。目前的工程处于“想法已具雏形，但实现仍是启发式规则”的阶段。具体体现在以下三个核心模块的差距：

### Gap 1: 错误机制的推断（Mechanism Inference）
*   **当前代码 ([masc_csc/mechanism.py](file:///F:/MASC_CSC/masc_csc/mechanism.py))**：使用纯逻辑规则计算。判断是否同音就直接对比 pypinyin 的输出，判断形近就用提前截取的图像算 cosine similarity。
*   **SCI 标准**：这种规则组合不能算作“模型能力”，容易被审稿人质疑。
*   **改造方案**：引入**对比学习增强的多模态检测（CMED）**或隐式的**特征映射**。即使保留 [mechanism.py](file:///F:/MASC_CSC/masc_csc/mechanism.py) 作为快速过滤，也需要在 NamBert 内部（[multimodal_frontend.py](file:///F:/MASC_CSC/models/multimodal_frontend.py)）增加一个显式的“机制分类器（Mechanism Head）”或在 LLM 端做一个“特征注入层（Feature Injection）”。

### Gap 2: 候选约束与 LLM 协同（Constraint & LLM Verifier）
*   **当前代码 ([candidate_generator.py](file:///F:/MASC_CSC/masc_csc/candidate_generator.py) & [llm_verifier.py](file:///F:/MASC_CSC/masc_csc/llm_verifier.py))**：在 Pipeline 里生成了几句话，用 A/B/C/D 的选项 Prompt 喂给 LLM 做多选题。
*   **SCI 标准**：多选 Prompt 是比较浅层的应用（工程上Work，学术上不够新）。如果我们要提 *Multimodal Confusion-Constrained Decoding (MCCD)*，我们需要深入到 Logits 层面。
*   **改造方案**：抛弃 A/B/C/D 选择题。我们将机制约束深入到 LLM 的解码层，通过动态 Mask 掉不在混淆集（由NamBert圈定）里的词，强迫 LLM 在“语义合理并且符合错误机制”的范围内生成（或者改写为更深度的 Prompt 引导）。

### Gap 3: 风险路由与自适应门控（Risk Routing / ACG）
*   **当前代码 ([router.py](file:///F:/MASC_CSC/masc_csc/router.py))**：用了四个硬编码的阈值（如 detection_score > 0.35, uncertainty_score > 1.5），做启发式打分。
*   **SCI 标准**：硬编码阈值在学术上被认为“缺乏泛化性（data-specific heuristic）”。
*   **改造方案**：将其升级为论文中规划的**基于多层感知机的自适应置信度门控（Adaptive Confidence-Gated Fusion, ACG）** $\alpha$ 网络。可以作为一个轻量级小模型，通过训练集上的误差回传进行联合优化或者简单的强化学习。

---

## 三、 推荐的实施开发路线（Proposed Changes）

基于以上分析，我为接下来的开发制定了以下实施方案：

### Phase 1: 夯实底座 (NamBert 端升级)
*   **[MODIFY] [models/multimodal_frontend.py](file:///F:/MASC_CSC/models/multimodal_frontend.py)**
    *   移除/优化目前极度依赖显式拼音图像提取的外部逻辑，强化 [predict_with_metadata](file:///F:/MASC_CSC/models/multimodal_frontend.py#259-298) 接口。
    *   **新增**：给 NamBert 增加一个 `DetectionHead`（二分类：是否有错）和一个 `MechanismHead`（多分类：音近/形近/其他），利用标注数据提前训练这两个 Head。
    *   （可选）加入 `ContrastiveLoss`，让音近/形近字的表征距离更远，便于分类。

### Phase 2: 开发可学习的自适应门控 (ACG 模块)
*   **[MODIFY] [masc_csc/router.py](file:///F:/MASC_CSC/masc_csc/router.py)**
    *   重构启发式规则，实现一个名为 `AdaptiveConfidenceGate` 的 PyTorch Module。
    *   输入：NamBert 输出的 `detection_logits`, [entropy](file:///F:/MASC_CSC/models/multimodal_frontend.py#255-258), `top1_prob`。
    *   输出：$\alpha \in (0, 1)$。
*   **[NEW] `masc_csc/train_router.py`**
    *   使用类似 DPO 或强化学习/交叉熵的思路，冷启动训练这个门控。如果 NamBert+LLM 的联合结果是对的，就鼓励当前 $\alpha$。

### Phase 3: 实现多模态约束解码 (MCCD)
*   **[MODIFY] [masc_csc/llm_verifier.py](file:///F:/MASC_CSC/masc_csc/llm_verifier.py)** 
    *   将其改名为 `llm_collaborator.py` 或 `llm_generator.py`。
    *   如果是使用本地 HuggingFace 模型（或者类似 vLLM 接口），添加一个 `LogitsProcessor`，在生成下一个字时，利用 NamBert 传来的混淆集（包含高置信度的音近/形近候选），将不在集合内的字 Logits 惩罚至负无穷，彻底封死过纠。
    *   如果仍受限于黑盒 API（只能发文本），则要彻底重构 Prompt 范式，使其看起来是一种“Type-Aware Constraint Programming”。

### Phase 4: 实验验证链条 (Verification Plan)
1.  **自动化测试基准建设**：在 [scripts/run_masc_csc.py](file:///F:/MASC_CSC/scripts/run_masc_csc.py) 基础上扩展 `scripts/evaluate_pipeline.py`，支持 SIGHAN 13/14/15 的全量批量验证计算 F1。
2.  **过纠率专项统计**：我们在 [utils/metrics.py](file:///F:/MASC_CSC/utils/metrics.py) 中增加“干净句子误伤率（False Positive Rate on Clean Sentences）”和“机制分类型 F1（分类统计音近提升、形近提升）”这在论文中是证明“解决大模型过纠”的最核心证据。

## 下一步行动请示

如果您认同这个分析和立意，我们可以立刻从 **Phase 1: 夯实底座 (NamBert 预测头的增加与重构)** 或者 **Phase 4: 建立能直接跑出 SIGHAN 批量分数的流水线脚本（方便跑消融）** 开始动手修改代码。您觉得我们先切入哪一步？
