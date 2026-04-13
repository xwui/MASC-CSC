"""
MASC-CSC Local LLM Verifier
============================
纯本地部署的大语言模型验证器，支持：
- Baichuan-7B / Qwen 等 HuggingFace 兼容模型本地推理
- LoRA adapter 动态加载
- 两种工作模式：
  (1) 选择题模式（ConstrainedChoiceLogitsProcessor）—— 旧版，保留用于对比
  (2) 定点纠正模式（ChineseCharLogitsProcessor）—— 新版，LLM 从全词表纠正
- 机制感知的增强 Prompt
"""

import logging
import string
from typing import Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from masc_csc.types import (
    CandidateSentence,
    ErrorMechanism,
    SentencePrediction,
    PositionPrediction,
    VerificationResult,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# 1. 约束解码器：选择题模式（旧版，保留用于 Baseline 对比）
# ──────────────────────────────────────────────────────────

class ConstrainedChoiceLogitsProcessor(LogitsProcessor):
    """
    在 model.generate() 的每一步拦截 logits，
    仅保留有效选项 token (A/B/C/D/E) 的概率通道，其余置为 -inf。
    这从根本上阻止 LLM 输出 "我选B因为..." 这类非格式化回复。
    """

    def __init__(self, valid_token_ids: List[int]):
        self.valid_ids = set(valid_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, float('-inf'))
        for token_id in self.valid_ids:
            mask[:, token_id] = 0.0
        return scores + mask


# ──────────────────────────────────────────────────────────
# 2. 约束解码器：定点纠正模式（新版）
# ──────────────────────────────────────────────────────────

class ChineseCharLogitsProcessor(LogitsProcessor):
    """
    定点纠正模式的约束解码器。

    按生成步骤交替约束：
    - 奇数步（第1、3、5...个 token）：只允许汉字 token
    - 偶数步（第2、4...个 token）：只允许逗号 token
    - 最后一步：只允许汉字 token

    这样 LLM 输出格式为："欢,苹" （汉字,汉字,汉字...）
    """

    def __init__(
            self,
            chinese_token_ids: List[int],
            comma_token_ids: List[int],
            total_positions: int,
    ):
        self.chinese_ids = set(chinese_token_ids)
        self.comma_ids = set(comma_token_ids)
        self.total_positions = total_positions
        self.step = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, float('-inf'))

        if self.step % 2 == 0:
            # 汉字步
            for tid in self.chinese_ids:
                mask[:, tid] = 0.0
        else:
            # 逗号步
            for tid in self.comma_ids:
                mask[:, tid] = 0.0

        self.step += 1
        return scores + mask


# ──────────────────────────────────────────────────────────
# 3. NoOp 验证器 (无 LLM 时的降级方案，保持不变)
# ──────────────────────────────────────────────────────────

class NoOpVerifier:
    """不调用 LLM，直接选择得分最高的候选"""

    def verify(self, prediction: SentencePrediction, candidates: Sequence[CandidateSentence]) -> VerificationResult:
        selected = max(candidates, key=lambda candidate: candidate.score)
        return VerificationResult(
            text=selected.text,
            selected_source=selected.source,
            reason="NoOp verifier fallback selected the highest-scoring candidate.",
            candidates=list(candidates),
        )


# ──────────────────────────────────────────────────────────
# 4. 机制名称映射（中文，用于 Prompt 展示）
# ──────────────────────────────────────────────────────────

MECHANISM_LABEL_ZH: Dict[ErrorMechanism, str] = {
    ErrorMechanism.PHONOLOGICAL: "音近错误（读音相似）",
    ErrorMechanism.VISUAL: "形近错误（字形相似）",
    ErrorMechanism.UNCERTAIN: "不确定",
}


# ──────────────────────────────────────────────────────────
# 5. 本地验证器（核心，支持两种模式）
# ──────────────────────────────────────────────────────────

class BaichuanLocalVerifier:
    """
    纯本地部署的 LLM 验证器。

    支持两种工作模式（通过 mode 参数切换）：

    mode="choice"（旧版，用于 Baseline 对比）：
        - LLM 从候选 A/B/C 中选择一个
        - 约束解码只允许选项字母
        - max_new_tokens = 1

    mode="targeted"（新版，默认）：
        - LLM 对每个可疑位置独立给出纠正字
        - 约束解码只允许汉字和逗号
        - max_new_tokens = 2N-1（N 为可疑位置数）
        - 突破候选上限：LLM 从全词表中选择
    """

    def __init__(
            self,
            model_path: str = "baichuan-inc/Baichuan-7B",
            adapter_path: Optional[str] = None,
            device: str = "cuda",
            torch_dtype=torch.float16,
            max_new_tokens: int = 1,
            mode: str = "targeted",
    ):
        """
        Args:
            model_path: 模型的本地路径或 HuggingFace Hub ID
            adapter_path: LoRA adapter 的路径（可选，微调后使用）
            device: 推理设备 ('cuda' / 'cpu')
            torch_dtype: 模型精度 (torch.float16 / torch.bfloat16)
            max_new_tokens: 生成的最大 token 数（choice 模式下为 1）
            mode: 工作模式，"choice"（选择题）或 "targeted"（定点纠正）
        """
        assert mode in ("choice", "targeted"), f"mode 必须是 'choice' 或 'targeted'，收到: {mode}"

        logger.info("Loading LLM model from: %s (mode=%s)", model_path, mode)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.mode = mode

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )

        # 加载基座模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )

        # 挂载 LoRA adapter（如果提供）
        if adapter_path is not None:
            logger.info("Loading LoRA adapter from: %s", adapter_path)
            try:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
                logger.info("LoRA adapter loaded successfully.")
            except ImportError:
                logger.warning("peft not installed, skipping LoRA adapter loading.")
            except Exception as e:
                logger.warning("Failed to load LoRA adapter: %s", e)

        self.model.eval()

        # 预计算 token 集合
        self._option_token_ids = self._resolve_option_token_ids()
        self._chinese_token_ids = self._resolve_chinese_token_ids()
        self._comma_token_ids = self._resolve_comma_token_ids()

        logger.info("Valid option token IDs: %s", self._option_token_ids)
        logger.info("Chinese token count: %d, Comma token count: %d",
                     len(self._chinese_token_ids), len(self._comma_token_ids))

    def _resolve_option_token_ids(self) -> List[int]:
        """获取选项字母 A-E 对应的 token id"""
        option_ids = []
        for letter in ['A', 'B', 'C', 'D', 'E']:
            ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if ids:
                option_ids.append(ids[0])
        return option_ids

    def _resolve_chinese_token_ids(self) -> List[int]:
        """获取词表中所有汉字对应的 token id"""
        chinese_ids = []
        for token, token_id in self.tokenizer.get_vocab().items():
            # 检查 token 是否为单个汉字（Unicode 范围 \u4e00-\u9fff）
            clean = token.replace("▁", "").replace("Ġ", "").strip()
            if len(clean) == 1 and '\u4e00' <= clean <= '\u9fff':
                chinese_ids.append(token_id)
        logger.info("Found %d Chinese character tokens in vocabulary.", len(chinese_ids))
        return chinese_ids

    def _resolve_comma_token_ids(self) -> List[int]:
        """获取逗号（中文逗号和英文逗号）对应的 token id"""
        comma_ids = []
        for comma_char in [',', '，']:
            ids = self.tokenizer.encode(comma_char, add_special_tokens=False)
            if ids:
                comma_ids.append(ids[0])
        return comma_ids

    @staticmethod
    def _label(index: int) -> str:
        return string.ascii_uppercase[index]

    # ──────────────────────────────────────────────────────
    # 选择题模式（旧版 Prompt + 解码，保留用于 Baseline）
    # ──────────────────────────────────────────────────────

    def _build_choice_prompt(
            self,
            prediction: SentencePrediction,
            candidates: Sequence[CandidateSentence],
    ) -> str:
        """构建选择题模式的 Prompt（旧版逻辑，保持不变）"""
        source_text = prediction.source_text
        source_chars = list(source_text)

        error_lines = []
        for position in prediction.edited_positions:
            idx = position.index
            char = source_chars[idx] if idx < len(source_chars) else "?"
            mech_label = MECHANISM_LABEL_ZH.get(position.mechanism, "不确定")
            error_lines.append(f"  - 位置{idx + 1} (\"{char}\")：{mech_label}")

        error_section = ""
        if error_lines:
            error_section = "检测到的潜在错误：\n" + "\n".join(error_lines) + "\n\n"

        candidate_lines = []
        for index, candidate in enumerate(candidates):
            label = self._label(index)
            if not candidate.edited_indices:
                annotation = "（保持原句不变）"
            else:
                edited_pos_str = ", ".join(str(i + 1) for i in candidate.edited_indices)
                annotation = f" [修改位置: {edited_pos_str}]"
            candidate_lines.append(f"{label}. {candidate.text}{annotation}")

        prompt = (
            f"源句子：\"{source_text}\"\n"
            f"{error_section}"
            f"候选修正方案：\n"
            f"{chr(10).join(candidate_lines)}\n\n"
            f"请从以上候选方案中选择最合适的修正结果。\n"
            f"严格只输出一个字母选项（如 A、B、C），不要输出其他任何内容。\n"
            f"选项："
        )
        return prompt

    def _parse_choice(self, generated_token_id: int, candidates: Sequence[CandidateSentence]) -> int:
        """将生成的 token id 解码为候选索引"""
        generated_text = self.tokenizer.decode([generated_token_id], skip_special_tokens=True).strip()
        if generated_text and generated_text[0] in string.ascii_uppercase:
            index = string.ascii_uppercase.index(generated_text[0])
            if index < len(candidates):
                return index
        logger.warning("Failed to parse LLM choice '%s', falling back to highest score.", generated_text)
        return max(range(len(candidates)), key=lambda i: candidates[i].score)

    @torch.no_grad()
    def _verify_choice(
            self,
            prediction: SentencePrediction,
            candidates: Sequence[CandidateSentence],
    ) -> VerificationResult:
        """选择题模式的验证逻辑（旧版）"""
        prompt = self._build_choice_prompt(prediction, candidates)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)

        num_candidates = len(candidates)
        valid_ids = self._option_token_ids[:num_candidates]
        logits_processor = LogitsProcessorList([
            ConstrainedChoiceLogitsProcessor(valid_ids),
        ])

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            logits_processor=logits_processor,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = outputs[0][input_ids.shape[1]:]
        if len(generated_ids) == 0:
            logger.warning("LLM generated empty output, falling back.")
            choice_index = max(range(len(candidates)), key=lambda i: candidates[i].score)
        else:
            choice_index = self._parse_choice(generated_ids[0].item(), candidates)

        selected = candidates[choice_index]
        choice_label = self._label(choice_index)

        return VerificationResult(
            text=selected.text,
            selected_source=selected.source,
            reason=f"LLM verifier (choice mode) selected option {choice_label} via constrained decoding.",
            candidates=list(candidates),
        )

    # ──────────────────────────────────────────────────────
    # 定点纠正模式（新版 Prompt + 解码）
    # ──────────────────────────────────────────────────────

    def _get_suspicious_positions(self, prediction: SentencePrediction) -> List[PositionPrediction]:
        """
        获取需要 LLM 定点纠正的可疑位置。
        选取被编辑过或检测分较高的位置，按风险排序，最多取 3 个。
        """
        candidates_pos = []
        for pos in prediction.positions:
            if pos.is_edited or pos.detection_score >= 0.08 or pos.uncertainty >= 0.80:
                candidates_pos.append(pos)

        # 按检测分 + 不确定度降序排列
        candidates_pos.sort(
            key=lambda p: (p.detection_score + p.uncertainty * 0.5),
            reverse=True,
        )
        return candidates_pos[:3]

    def _build_targeted_prompt(
            self,
            prediction: SentencePrediction,
            suspicious_positions: List[PositionPrediction],
    ) -> str:
        """
        构建定点纠正模式的 Prompt。

        示例输出：
        ────────────────────
        以下句子中标记的位置可能存在拼写错误。
        请根据上下文语义，判断每个标记位置的正确汉字。

        句子：我 喜 [换] 吃 [平] 果

        可疑位置：
        [1] 第3字"换" — 音近错误（读音相似），前端建议"欢"
        [2] 第5字"平" — 形近错误（字形相似），前端建议"苹"

        请按顺序输出每个可疑位置的正确汉字，用逗号分隔。
        如果前端建议正确就采纳，如果不正确请给出你认为正确的字，如果原字无误请保留原字。

        输出：
        ────────────────────
        """
        source_text = prediction.source_text
        source_chars = list(source_text)

        # 构建带标记的句子展示
        suspicious_indices = set(p.index for p in suspicious_positions)
        display_chars = []
        for i, char in enumerate(source_chars):
            if i in suspicious_indices:
                display_chars.append(f"[{char}]")
            else:
                display_chars.append(char)
        display_sentence = " ".join(display_chars)

        # 构建可疑位置详情
        position_lines = []
        for rank, pos in enumerate(suspicious_positions, 1):
            idx = pos.index
            char = source_chars[idx] if idx < len(source_chars) else "?"
            mech_label = MECHANISM_LABEL_ZH.get(pos.mechanism, "不确定")
            suggestion = pos.predicted_token
            position_lines.append(
                f"[{rank}] 第{idx + 1}字\"{char}\" — {mech_label}，前端建议\"{suggestion}\""
            )

        prompt = (
            f"以下句子中标记的位置可能存在拼写错误。\n"
            f"请根据上下文语义，判断每个标记位置的正确汉字。\n\n"
            f"句子：{display_sentence}\n\n"
            f"可疑位置：\n"
            f"{chr(10).join(position_lines)}\n\n"
            f"请按顺序输出每个可疑位置的正确汉字，用逗号分隔。\n"
            f"如果前端建议正确就采纳，如果不正确请给出你认为正确的字，如果原字无误请保留原字。\n\n"
            f"输出："
        )
        return prompt

    def _parse_targeted_output(
            self,
            generated_text: str,
            suspicious_positions: List[PositionPrediction],
    ) -> List[str]:
        """
        解析定点纠正模式的 LLM 输出。

        输入：如 "欢,苹" 或 "欢，苹"
        输出：["欢", "苹"]

        如果解析失败，降级使用前端建议。
        """
        # 统一逗号格式
        text = generated_text.replace("，", ",").strip()
        chars = [c.strip() for c in text.split(",") if c.strip()]

        # 补齐或截断到目标位置数
        n = len(suspicious_positions)
        result = []
        for i in range(n):
            if i < len(chars) and len(chars[i]) == 1 and '\u4e00' <= chars[i] <= '\u9fff':
                result.append(chars[i])
            else:
                # 降级：使用前端建议
                result.append(suspicious_positions[i].predicted_token)
                logger.warning(
                    "Position %d: failed to parse LLM output, using frontend suggestion '%s'.",
                    i, suspicious_positions[i].predicted_token,
                )
        return result

    @torch.no_grad()
    def _verify_targeted(
            self,
            prediction: SentencePrediction,
            candidates: Sequence[CandidateSentence],
    ) -> VerificationResult:
        """
        定点纠正模式的验证逻辑（新版）。

        LLM 对每个可疑位置独立给出纠正字，不受 BERT 候选限制。
        """
        suspicious_positions = self._get_suspicious_positions(prediction)

        if not suspicious_positions:
            # 没有可疑位置，直接返回原句
            return VerificationResult(
                text=prediction.source_text,
                selected_source="original",
                reason="No suspicious positions for targeted correction.",
                candidates=list(candidates),
            )

        prompt = self._build_targeted_prompt(prediction, suspicious_positions)
        n_positions = len(suspicious_positions)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)

        # 构建约束解码器
        logits_processor = LogitsProcessorList([
            ChineseCharLogitsProcessor(
                chinese_token_ids=self._chinese_token_ids,
                comma_token_ids=self._comma_token_ids,
                total_positions=n_positions,
            ),
        ])

        # 生成：N 个汉字 + (N-1) 个逗号 = 2N-1 个 token
        max_tokens = 2 * n_positions - 1
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max(max_tokens, 1),
            logits_processor=logits_processor,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # 提取并解析生成的 token
        generated_ids = outputs[0][input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        logger.info("Targeted correction LLM output: '%s'", generated_text)

        corrections = self._parse_targeted_output(generated_text, suspicious_positions)

        # 将纠正字回填到原句
        result_chars = list(prediction.source_text)
        corrected_indices = []
        for pos, correction in zip(suspicious_positions, corrections):
            if correction != result_chars[pos.index]:
                corrected_indices.append(pos.index)
            result_chars[pos.index] = correction

        corrected_text = "".join(result_chars)

        # 构造原因说明
        correction_details = []
        for pos, correction in zip(suspicious_positions, corrections):
            src_char = prediction.source_text[pos.index]
            if correction != src_char:
                correction_details.append(f"位置{pos.index + 1}: '{src_char}'→'{correction}'")
            else:
                correction_details.append(f"位置{pos.index + 1}: '{src_char}'保持不变")

        reason = (
            f"LLM verifier (targeted mode) corrected {len(corrected_indices)} positions. "
            f"Details: {'; '.join(correction_details)}"
        )

        return VerificationResult(
            text=corrected_text,
            selected_source="llm-targeted-correction",
            reason=reason,
            candidates=list(candidates),
        )

    # ──────────────────────────────────────────────────────
    # 统一入口
    # ──────────────────────────────────────────────────────

    def verify(
            self,
            prediction: SentencePrediction,
            candidates: Sequence[CandidateSentence],
    ) -> VerificationResult:
        """
        统一验证入口，根据 self.mode 调度到对应模式。
        """
        if self.mode == "choice":
            return self._verify_choice(prediction, candidates)
        else:
            return self._verify_targeted(prediction, candidates)

    # 向后兼容：保留 build_prompt 方法
    def build_prompt(self, prediction, candidates):
        if self.mode == "choice":
            return self._build_choice_prompt(prediction, candidates)
        suspicious = self._get_suspicious_positions(prediction)
        return self._build_targeted_prompt(prediction, suspicious)


# ──────────────────────────────────────────────────────────
# 6. 向后兼容：保留别名
# ──────────────────────────────────────────────────────────

LocalLLMVerifier = BaichuanLocalVerifier
