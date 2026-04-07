# MASC-CSC 架构与接口说明

## 1. 目录结构

当前项目的核心目录如下：

```text
MASC_CSC/
├── models/
│   ├── multimodal_frontend.py
│   └── common.py
├── masc_csc/
│   ├── __init__.py
│   ├── types.py
│   ├── mechanism.py
│   ├── candidate_generator.py
│   ├── router.py
│   ├── llm_verifier.py
│   └── pipeline.py
├── scripts/
│   ├── data_process.py
│   └── run_masc_csc.py
├── utils/
│   ├── dataset.py
│   ├── dataloader.py
│   ├── metrics.py
│   └── utils.py
├── train.py
└── train_frontend.sh
```

## 2. 主要模块说明

### 2.1 `models/multimodal_frontend.py`

这是前端小模型主实现。

核心类：

- `MultimodalCSCFrontend`

主要职责：

- 训练字符级 CSC 前端
- 输出字符级 logits
- 输出 top-k 候选
- 暴露检测与不确定性信息

关键接口：

#### `predict(sentence: str) -> str`

输入一句中文字符串，输出前端模型自己的 top-1 纠错结果。

#### `predict_with_metadata(sentence: str, top_k: int = 5) -> Dict[str, List]`

这是 MASC-CSC 协同层最依赖的接口。

当前返回的关键字段包括：

- `source_text`
- `source_tokens`
- `predicted_text`
- `predicted_tokens`
- `topk_ids`
- `topk_probs`
- `topk_tokens`
- `copy_probs`
- `detection_scores`
- `uncertainty_scores`

这是后续 `pipeline.py` 的输入源。

### 2.2 `masc_csc/types.py`

这里定义协同层的数据结构。

#### `ErrorMechanism`

枚举类型：

- `PHONOLOGICAL`
- `VISUAL`
- `UNCERTAIN`

#### `TokenAlternative`

表示某个位置上的候选字：

- `token`
- `token_id`
- `score`

#### `PositionPrediction`

表示某一个字符位置的综合预测信息：

- 原字
- 预测字
- 检测分数
- 不确定度
- 错误机制
- 候选列表

#### `SentencePrediction`

表示整句的中间状态，包括：

- 原句
- 前端预测句
- 所有位置预测

#### `CandidateSentence`

表示候选句：

- `text`
- `edited_indices`
- `score`
- `source`

#### `RouterDecision`

表示路由器的决策：

- `invoke_llm`
- `risk_score`
- `reasons`

#### `VerificationResult`

表示最终验证结果：

- `text`
- `selected_source`
- `reason`
- `candidates`

### 2.3 `masc_csc/mechanism.py`

核心类：

- `MechanismInferencer`

作用：

- 利用拼音和字形信息，为候选位置推断错误机制

关键接口：

#### `is_phonological_match(source_token, candidate_token) -> bool`

判断两个字是否满足音近匹配。

#### `glyph_similarity(source_token, candidate_token) -> float`

计算字形图像余弦相似度。

#### `is_visual_match(source_token, candidate_token) -> bool`

根据图像相似度阈值判断形近。

#### `infer_from_alternatives(source_token, alternatives) -> ErrorMechanism`

根据当前位置候选集投票出机制类型。

### 2.4 `masc_csc/candidate_generator.py`

核心类：

- `MechanismAwareCandidateGenerator`

作用：

- 在机制约束下生成少量等长候选句

关键配置：

- `max_positions`
- `max_candidates`
- `max_alternatives_per_position`

关键接口：

#### `generate(prediction: SentencePrediction) -> List[CandidateSentence]`

输入整句预测，输出候选句列表。

当前规则：

- 原句保留
- 选择高风险位置展开
- 按机制过滤候选
- 加入前端模型 top-1 结果

### 2.5 `masc_csc/router.py`

核心类：

- `RiskAwareRouter`

作用：

- 判断一句话是否应交给 LLM verifier

关键配置：

- `detection_threshold`
- `uncertainty_threshold`
- `margin_threshold`
- `multi_edit_threshold`

关键接口：

#### `decide(prediction: SentencePrediction) -> RouterDecision`

当前为启发式打分。

### 2.6 `masc_csc/llm_verifier.py`

核心类：

- `LocalLLMVerifier`
- `NoOpVerifier`

作用：

- 调用本地 OpenAI-compatible LLM 服务
- 在候选中做选择

关键接口：

#### `build_prompt(prediction, candidates) -> List[dict]`

组装 chat completion 用的消息。

#### `verify(prediction, candidates) -> VerificationResult`

返回最终验证结果。

当前默认输出格式约束为：

- `Choice: <label>`
- `Reason: <one sentence>`

### 2.7 `masc_csc/pipeline.py`

核心类：

- `MASCCSCPipeline`

作用：

- 串联前端、机制推断、候选生成、路由和 verifier

关键接口：

#### `analyze(sentence: str, top_k: int = 5) -> SentencePrediction`

返回中间分析结果。

#### `correct(sentence: str, top_k: int = 5) -> VerificationResult`

返回最终纠错结果。

## 3. 训练入口

文件：

- `train.py`

当前支持的模型名：

- `multimodal_frontend`
- `frontend`
- `masc_frontend`

主要功能：

- 训练前端模型
- 验证前端模型
- 测试前端模型

注意：

- 当前 `train.py` 还不直接训练 verifier 或整套协同系统
- 它目前只负责前端小模型

## 4. 数据接口

文件：

- `utils/dataset.py`
- `utils/dataloader.py`
- `scripts/data_process.py`

### 4.1 训练数据格式

当前项目使用 CSV：

```text
src,tgt
我喜换吃平果,我喜欢吃苹果
...
```

要求：

- `src` 和 `tgt` 等长
- 不能包含逗号

### 4.2 数据预处理

`scripts/data_process.py` 支持把原始 pickle 数据转成 CSV。

默认会生成：

- `sighan_2013_test.csv`
- `sighan_2014_test.csv`
- `sighan_2015_test.csv`
- `train.csv`

## 5. 当前接口层面的已知不足

### 5.1 `predict_with_metadata()` 还比较原型化

虽然字段已经够协同层使用，但后续仍建议：

- 增加句级风险分数
- 增加显式机制标签
- 增加更规范的返回对象

### 5.2 verifier 只支持单轮单句调用

当前 `LocalLLMVerifier` 适合原型验证，不适合大规模实验直接跑。

### 5.3 路由器还是启发式

后续可以把 `RiskAwareRouter` 升级成可学习模块。

## 6. 给接手开发者的建议

如果别人接手开发，建议从下面 4 个文件开始：

- `models/multimodal_frontend.py`
- `masc_csc/mechanism.py`
- `masc_csc/candidate_generator.py`
- `masc_csc/pipeline.py`

这四个文件基本覆盖了当前项目最核心的数据流。
