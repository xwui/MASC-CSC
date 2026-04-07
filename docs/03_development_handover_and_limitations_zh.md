# MASC-CSC 开发接续与已知不足

## 1. 当前项目适合做什么

MASC-CSC 当前最适合做的是：

- 验证协同思路
- 跑单句或小规模样本实验
- 继续完善机制推断和候选生成
- 开始搭建论文实验脚本

## 2. 当前项目还不适合直接做什么

当前项目还不太适合直接拿去做以下事情：

- 大规模自动实验
- verifier 批量评测
- 完整论文最终实验复现
- 一键式部署

原因不是结构有问题，而是很多模块还停留在“研究骨架”阶段。

## 3. 距离学术顶会/SCI的严重缺陷（已知不足）

### 3.1 机制推断全是硬编码规则，不是端到端特征 (缺少 CMED)
当前 `MechanismInferencer` 使用 `pypinyin` 的字符匹配和简单的字体图像余弦距离来区分音近/形近。这种非参数化的外部规则不能算作模型本身的“感知能力”，且对罕见字或复杂上下文的抗干扰力极差。**必须进行重构，在模型内部增加联合优化的分类头。**

### 3.2 路由器是人类写死的阈值 (缺少 ACG)
`RiskAwareRouter` 的触发是“当检测＞0.35 且 熵＞1.5”此类拍脑袋决定。这彻底丧失了作为论文核心贡献“自适应置信度门控”的学术价值。**必须引入通过监督或强化学习更新的独立评分网络 $\alpha$。**

### 3.3 Verifier 只是做表面 Prompt 选择题 (缺少 MCCD)
当前的 `LocalLLMVerifier` 让大模型做文字选择题（A, B, C）。这种 Prompt 限制了大模型的长处，且极易导致幻觉。真正的“混淆集约束解码”要求我们深入到底层 API，通过 Logits Mask 的方式介入大语言模型的每一步解码。

### 3.4 严重匮乏对过纠问题证明的实验脚本
目前只能评测单句或是算出最基础的 Frontend 指标。为了证明大模型“不瞎改了”，急需一整套能够验证 Clean Dataset 原句保持率、按“机制切片”提供细粒度 F1 的脚本。

## 4. SCI 级别改造的下一步开发路线

### Phase 1：夯实底层特征网 (CMED 增强)
- 任务 1：改造 `multimodal_frontend.py` 的 Forward，引入单独两层 MLP 作为 `DetectionHead` (有无错) 与 `MechanismHead` (音近/形近/其他)。
- 任务 2：收集正负样本，加入 Contrastive Loss。

### Phase 2：开发可学习的自适应门控 (ACG)
- 任务 1：废除 `router.py` 中的人工阈值，代之以 `AdaptiveConfidenceGate(nn.Module)`。
- 任务 2：增加一个 `train_router.py` 离线蒸馏或在验证集上监督学习这个门控的分数。

### Phase 3：植入隐式约束解码 (MCCD)
- 任务 1：重写 LLM 调用模块。弃用自然语言选择题。
- 任务 2：若对接 HuggingFace/vLLM 本地大模型，截断并植入 `LogitsProcessor`；若受限只能查 API，则重写强化版 System Prompt。

### Phase 4：彻底拉通批量验证链
- 任务 1：写出 `scripts/evaluate_masc.py`，必须覆盖 SIGHAN 13/14/15 测试集的端到端测评。
- 任务 2：扩展 `metrics.py`，支持 FPR (误改率) 和 TPR (正确召回率) 对比。

## 5. 如果是别人来接手，建议先看什么

### 第一步

读：

- `README.md`
- `docs/01_implementation_thinking_zh.md`

### 第二步

读：

- `docs/02_architecture_and_interfaces_zh.md`

### 第三步

看代码：

- `models/multimodal_frontend.py`
- `masc_csc/pipeline.py`
- `masc_csc/mechanism.py`
- `masc_csc/candidate_generator.py`

### 第四步

最后再去动：

- `masc_csc/llm_verifier.py`
- `train.py`

## 6. 交付给别人的最低要求

如果你要把项目交给别人继续开发，至少应保证下面几项是清楚的：

- 项目主线是什么
- 哪个文件是前端入口
- 哪个文件是协同入口
- 数据格式是什么
- 当前哪些功能已经能跑
- 当前哪些功能只是骨架

目前这套文档的目标，就是满足这几个最低要求。
