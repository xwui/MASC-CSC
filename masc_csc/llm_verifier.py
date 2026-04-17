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
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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

    输出格式被约束为：汉字,汉字,汉字...
    - 第 1,3,5... 步：只允许单个汉字 token
    - 第 2,4,6... 步：只允许逗号 token
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
            valid_ids = self.chinese_ids
        else:
            valid_ids = self.comma_ids

        for tid in valid_ids:
            mask[:, tid] = 0.0

        self.step += 1
        return scores + mask


class ChoiceSequenceLogitsProcessor(LogitsProcessor):
    """
    定点 closed-choice 模式的约束解码器。

    输出格式被约束为：字母,字母,字母...
    - 第 1,3,5... 步：只允许选项字母 token
    - 第 2,4,6... 步：只允许逗号 token
    """

    def __init__(
            self,
            option_token_ids: List[int],
            comma_token_ids: List[int],
            total_positions: int,
    ):
        self.option_ids = set(option_token_ids)
        self.comma_ids = set(comma_token_ids)
        self.total_positions = total_positions
        self.step = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, float('-inf'))
        valid_ids = self.option_ids if self.step % 2 == 0 else self.comma_ids
        for token_id in valid_ids:
            mask[:, token_id] = 0.0
        self.step += 1
        return scores + mask


@dataclass
class StageOneDecision:
    position: PositionPrediction
    options: List[str]
    label: str
    selected_token: Optional[str]
    abstained: bool


@dataclass
class StageTwoProposal:
    position: PositionPrediction
    proposed_char: Optional[str]
    kept: bool
    raw_output: str


# ──────────────────────────────────────────────────────────
# 3. NoOp 验证器 (无 LLM 时的降级方案，保持不变)
# ──────────────────────────────────────────────────────────

class NoOpVerifier:
    """不调用 LLM，直接选择得分最高的候选"""

    def verify(
            self,
            prediction: SentencePrediction,
            candidates: Sequence[CandidateSentence],
            selected_positions: Optional[List[int]] = None,
    ) -> VerificationResult:
        del prediction, selected_positions
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
        """获取选项字母 A-Z 对应的 token id"""
        option_ids = []
        for letter in string.ascii_uppercase:
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

    def _prepare_prompt_inputs(self, prompt: str) -> Dict[str, torch.Tensor]:
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {
                    'role': 'system',
                    'content': '你是中文拼写纠错助手。严格按要求只输出最终答案，不要解释，不要复述题目。',
                },
                {
                    'role': 'user',
                    'content': prompt,
                },
            ]
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = self.tokenizer(rendered, return_tensors='pt')
        else:
            model_inputs = self.tokenizer(prompt, return_tensors='pt')
        return {name: tensor.to(self.model.device) for name, tensor in model_inputs.items()}

    def _generate_text(
            self,
            inputs: Dict[str, torch.Tensor],
            max_new_tokens: int,
            logits_processor: Optional[LogitsProcessorList] = None,
    ):
        generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.001,
            top_p=1.0,
            top_k=50,
            repetition_penalty=1.0,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if logits_processor is not None:
            generate_kwargs['logits_processor'] = logits_processor

        outputs = self.model.generate(**inputs, **generate_kwargs)
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_ids, generated_text

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

        inputs = self._prepare_prompt_inputs(prompt)
        input_ids = inputs["input_ids"]

        num_candidates = len(candidates)
        valid_ids = self._option_token_ids[:num_candidates]
        logits_processor = LogitsProcessorList([
            ConstrainedChoiceLogitsProcessor(valid_ids),
        ])

        generated_ids, _ = self._generate_text(
            inputs,
            max_new_tokens=self.max_new_tokens,
            logits_processor=logits_processor,
        )
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

    def _get_suspicious_positions(
            self,
            prediction: SentencePrediction,
            selected_positions: Optional[List[int]] = None,
    ) -> List[PositionPrediction]:
        """
        Resolve the positions for targeted correction.

        If the router provides explicit positions, prefer them. Otherwise fall back
        to the verifier's own high-recall proposal heuristic.
        """
        if selected_positions:
            selected = []
            seen = set()
            for index in selected_positions:
                if index in seen:
                    continue
                if 0 <= index < len(prediction.positions):
                    selected.append(prediction.positions[index])
                    seen.add(index)
            if selected:
                return selected

        candidates_pos = []
        for pos in prediction.positions:
            if pos.is_edited or pos.detection_score >= 0.08 or pos.uncertainty >= 0.80:
                candidates_pos.append(pos)

        candidates_pos.sort(
            key=lambda p: (p.detection_score + p.uncertainty * 0.5),
            reverse=True,
        )
        return candidates_pos[:3]

    def _build_targeted_options(
            self,
            position: PositionPrediction,
            max_options: int = 4,
    ) -> List[str]:
        options: List[str] = []
        seen = set()

        def add(token: str):
            if not self._is_single_chinese_char(token):
                return
            if token in seen:
                return
            seen.add(token)
            options.append(token)

        add(position.source_token)
        if position.predicted_token != position.source_token:
            add(position.predicted_token)
        for alternative in position.alternatives:
            add(alternative.token)
            if len(options) >= max_options:
                break

        return options or [position.source_token]

    def _build_targeted_prompt(
            self,
            prediction: SentencePrediction,
            suspicious_positions: List[PositionPrediction],
            option_lists: Optional[List[List[str]]] = None,
            allow_abstain: bool = False,
            stage_name: str = "第一阶段",
    ) -> str:
        """Build a targeted closed-choice prompt with optional abstention."""
        source_text = prediction.source_text
        source_chars = list(source_text)

        suspicious_indices = set(p.index for p in suspicious_positions)
        display_chars = []
        for i, char in enumerate(source_chars):
            if i in suspicious_indices:
                display_chars.append(f"[{char}]")
            else:
                display_chars.append(char)
        display_sentence = " ".join(display_chars)

        position_blocks = []
        for rank, pos in enumerate(suspicious_positions, 1):
            idx = pos.index
            char = source_chars[idx] if idx < len(source_chars) else "?"
            mech_label = MECHANISM_LABEL_ZH.get(pos.mechanism, "不确定")
            suggestion = pos.predicted_token
            options = option_lists[rank - 1] if option_lists is not None else self._build_targeted_options(pos)
            option_lines = []
            for option_index, token in enumerate(options):
                label = self._label(option_index)
                if option_index == 0:
                    option_lines.append(f"    {label}. 保持原字：{token}")
                else:
                    option_lines.append(f"    {label}. 改为：{token}")
            if allow_abstain:
                option_lines.append("    N. 候选均不合适")
            position_blocks.append(
                f"[{rank}] 第{idx + 1}字\"{char}\" — {mech_label}，前端建议\"{suggestion}\"\n"
                f"{chr(10).join(option_lines)}"
            )

        abstain_instructions = (
            "如果所有候选都明显不合适，并且该位置确实存在拼写错误，才输出 N。\n"
            "N 不是默认答案，只有在候选空间不足时才能使用。\n"
            if allow_abstain else
            "你必须在给定候选中做出选择，不能输出候选外答案。\n"
        )
        prompt = (
            f"{stage_name}：以下句子中只有标记位置可能存在拼写错误。\n"
            "你是中文拼写纠错验证器，不是改写器。\n"
            "禁止同义替换、风格润色、专名改写、代词性别替换、异体字/规范字替换。\n"
            "除标记位置外，不要改动任何其他字符。\n\n"
            f"句子：{display_sentence}\n\n"
            "可疑位置与选项：\n"
            f"{chr(10).join(position_blocks)}\n\n"
            f"{abstain_instructions}"
            "请按顺序输出每个可疑位置的选项字母，用逗号分隔。\n"
            "严格只输出字母和逗号，不要解释，不要输出序号、空格或其他任何文字。\n"
            "输出示例：A,B,C\n\n"
            "输出："
        )
        return prompt

    def _build_stage2_prompt(
            self,
            prediction: SentencePrediction,
            position: PositionPrediction,
            rejected_options: List[str],
    ) -> str:
        source_chars = list(prediction.source_text)
        display_chars = []
        for i, char in enumerate(source_chars):
            display_chars.append(f"[{char}]" if i == position.index else char)
        display_sentence = " ".join(display_chars)
        options_text = " / ".join(rejected_options)
        return (
            "第二阶段：前一步已经判定当前候选都不合适。\n"
            "你是中文拼写纠错补充器，不是改写器。\n"
            "只允许针对指定位置输出一个单个汉字；如果没有明确更优的补充修正，请直接输出原字本身。\n"
            "禁止同义替换、风格润色、专名改写、代词性别替换、异体字/规范字替换。\n\n"
            f"句子：{display_sentence}\n"
            f"检查位置：第{position.index + 1}字“{position.source_token}”\n"
            f"已判定不合适的候选：{options_text}\n\n"
            "要求：\n"
            "1. 只能输出一个汉字\n"
            "2. 如果没有把握，请输出原字本身\n"
            "3. 不要输出解释、标点或其他内容\n\n"
            "输出："
        )

    @staticmethod
    def _first_chinese_char(text: str) -> Optional[str]:
        for ch in text:
            if '\u4e00' <= ch <= '\u9fff':
                return ch
        return None

    @staticmethod
    def _contains_keep_marker(text: str) -> bool:
        normalized = text.strip().upper()
        keep_markers = ["KEEP", "保留", "原字", "不改", "保持"]
        return any(marker in normalized or marker in text for marker in keep_markers)

    @staticmethod
    def _is_single_chinese_char(text: str) -> bool:
        return len(text) == 1 and '\u4e00' <= text <= '\u9fff'

    def _parse_choice_sequence_labels(
            self,
            generated_text: str,
            expected_count: int,
    ) -> List[Optional[str]]:
        """Parse comma-separated choice labels."""
        text = generated_text.replace("，", ",").replace(" ", "").strip().upper()
        segments = [seg.strip() for seg in text.split(",") if seg.strip()]
        labels: List[Optional[str]] = []
        seg_idx = 0
        for _ in range(expected_count):
            chosen_label = None
            while seg_idx < len(segments):
                segment = segments[seg_idx]
                seg_idx += 1
                label = next((ch for ch in segment if ch in string.ascii_uppercase), None)
                if label is not None:
                    chosen_label = label
                    break
            labels.append(chosen_label)
        return labels

    def _run_stage1(
            self,
            prediction: SentencePrediction,
            active_positions: List[PositionPrediction],
            option_lists: List[List[str]],
    ) -> List[StageOneDecision]:
        prompt = self._build_targeted_prompt(
            prediction,
            active_positions,
            option_lists,
            allow_abstain=True,
            stage_name="第一阶段",
        )
        inputs = self._prepare_prompt_inputs(prompt)
        n_positions = len(active_positions)
        logits_processor = None
        if self._option_token_ids and (n_positions == 1 or self._comma_token_ids):
            logits_processor = LogitsProcessorList([
                ChoiceSequenceLogitsProcessor(
                    self._option_token_ids,
                    self._comma_token_ids,
                    n_positions,
                )
            ])

        max_tokens = max(1, 2 * n_positions - 1)
        _, generated_text = self._generate_text(
            inputs,
            max_new_tokens=max_tokens,
            logits_processor=logits_processor,
        )
        logger.info("Stage-1 closed-choice output: %s", generated_text)

        labels = self._parse_choice_sequence_labels(generated_text, n_positions)
        decisions: List[StageOneDecision] = []
        for pos, options, label in zip(active_positions, option_lists, labels):
            if label == "N":
                decisions.append(
                    StageOneDecision(
                        position=pos,
                        options=options,
                        label="N",
                        selected_token=None,
                        abstained=True,
                    )
                )
                continue

            selected_token = None
            if label is not None:
                option_index = string.ascii_uppercase.index(label)
                if option_index < len(options):
                    selected_token = options[option_index]

            if selected_token is None:
                selected_token = options[0] if options else pos.source_token
                logger.warning(
                    "Failed to parse stage-1 output '%s' for position %d, using fallback '%s'.",
                    generated_text,
                    pos.index,
                    selected_token,
                )
                label = "A"

            decisions.append(
                StageOneDecision(
                    position=pos,
                    options=options,
                    label=label,
                    selected_token=selected_token,
                    abstained=False,
                )
            )
        return decisions

    def _should_allow_open_repair(self, position: PositionPrediction) -> bool:
        if position.detection_score >= 0.25:
            return True
        if position.uncertainty >= 1.10:
            return True
        if position.margin <= 0.12:
            return True
        if position.mechanism == ErrorMechanism.UNCERTAIN:
            return True
        return position.is_edited

    def _parse_stage2_output(self, generated_text: str, source_char: str) -> str:
        proposed_char = self._first_chinese_char(generated_text)
        if proposed_char is not None:
            return proposed_char
        if self._contains_keep_marker(generated_text):
            return source_char
        return source_char

    def _run_stage2(
            self,
            prediction: SentencePrediction,
            stage1_decisions: List[StageOneDecision],
    ) -> List[StageTwoProposal]:
        proposals: List[StageTwoProposal] = []
        for decision in stage1_decisions:
            if not decision.abstained:
                continue
            if not self._should_allow_open_repair(decision.position):
                proposals.append(
                    StageTwoProposal(
                        position=decision.position,
                        proposed_char=decision.position.source_token,
                        kept=True,
                        raw_output="",
                    )
                )
                continue

            prompt = self._build_stage2_prompt(
                prediction,
                decision.position,
                decision.options,
            )
            inputs = self._prepare_prompt_inputs(prompt)
            logits_processor = None
            if self._chinese_token_ids:
                logits_processor = LogitsProcessorList([
                    ChineseCharLogitsProcessor(
                        self._chinese_token_ids,
                        self._comma_token_ids,
                        1,
                    )
                ])

            _, generated_text = self._generate_text(
                inputs,
                max_new_tokens=1,
                logits_processor=logits_processor,
            )
            logger.info(
                "Stage-2 restricted repair output for position %d: %s",
                decision.position.index,
                generated_text,
            )
            proposed_char = self._parse_stage2_output(generated_text, decision.position.source_token)
            proposals.append(
                StageTwoProposal(
                    position=decision.position,
                    proposed_char=proposed_char,
                    kept=(proposed_char == decision.position.source_token),
                    raw_output=generated_text,
                )
            )
        return proposals

    def _run_stage3(
            self,
            prediction: SentencePrediction,
            stage1_decisions: List[StageOneDecision],
            proposals: List[StageTwoProposal],
    ) -> Dict[int, Tuple[str, str]]:
        proposal_map = {proposal.position.index: proposal for proposal in proposals}
        active_positions: List[PositionPrediction] = []
        option_lists: List[List[str]] = []
        fallback_tokens: Dict[int, str] = {}

        for decision in stage1_decisions:
            proposal = proposal_map.get(decision.position.index)
            if proposal is None or proposal.kept or proposal.proposed_char is None:
                continue
            if proposal.proposed_char == decision.position.source_token:
                continue

            options = list(decision.options)
            if proposal.proposed_char not in options:
                options.append(proposal.proposed_char)

            if len(options) <= 1:
                continue

            active_positions.append(decision.position)
            option_lists.append(options)
            fallback_tokens[decision.position.index] = decision.position.source_token

        if not active_positions:
            return {}

        prompt = self._build_targeted_prompt(
            prediction,
            active_positions,
            option_lists,
            allow_abstain=False,
            stage_name="第三阶段",
        )
        inputs = self._prepare_prompt_inputs(prompt)
        n_positions = len(active_positions)
        logits_processor = None
        if self._option_token_ids and (n_positions == 1 or self._comma_token_ids):
            logits_processor = LogitsProcessorList([
                ChoiceSequenceLogitsProcessor(
                    self._option_token_ids,
                    self._comma_token_ids,
                    n_positions,
                )
            ])

        max_tokens = max(1, 2 * n_positions - 1)
        _, generated_text = self._generate_text(
            inputs,
            max_new_tokens=max_tokens,
            logits_processor=logits_processor,
        )
        logger.info("Stage-3 recheck output: %s", generated_text)

        labels = self._parse_choice_sequence_labels(generated_text, n_positions)
        final_choices: Dict[int, Tuple[str, str]] = {}
        for pos, options, label in zip(active_positions, option_lists, labels):
            selected_token = None
            if label is not None:
                option_index = string.ascii_uppercase.index(label)
                if option_index < len(options):
                    selected_token = options[option_index]
            if selected_token is None:
                selected_token = fallback_tokens.get(pos.index, pos.source_token)
                final_choices[pos.index] = (selected_token, "stage3_fallback")
            elif selected_token == fallback_tokens.get(pos.index, pos.source_token):
                final_choices[pos.index] = (selected_token, "stage3_reject_open")
            else:
                final_choices[pos.index] = (selected_token, "accepted_stage3_recheck")
        return final_choices

    def _safe_merge_targeted_corrections(
            self,
            prediction: SentencePrediction,
            candidates: Sequence[CandidateSentence],
            suspicious_positions: List[PositionPrediction],
            corrections: List[str],
            decision_reasons: Optional[Dict[int, str]] = None,
    ) -> VerificationResult:
        result_chars = list(prediction.source_text)
        corrected_indices = []
        correction_details = []

        for pos, correction in zip(suspicious_positions, corrections):
            src_char = prediction.source_text[pos.index]
            accepted_char = src_char
            decision_reason = (decision_reasons or {}).get(pos.index, 'keep')

            if correction == 'KEEP':
                accepted_char = src_char
                decision_reason = 'llm_keep'
            elif not self._is_single_chinese_char(correction):
                accepted_char = src_char
                decision_reason = 'invalid_output_fallback'
            elif correction == src_char:
                accepted_char = src_char
                if decision_reason == 'keep':
                    decision_reason = 'llm_keep'
            else:
                high_confidence_frontend_keep = (
                    (not pos.is_edited)
                    and pos.detection_score < 0.10
                    and pos.uncertainty < 0.90
                    and pos.margin >= 0.35
                )
                if high_confidence_frontend_keep:
                    accepted_char = src_char
                    decision_reason = 'frontend_veto'
                else:
                    accepted_char = correction
                    decision_reason = 'accepted_closed_choice'

            result_chars[pos.index] = accepted_char
            if accepted_char != src_char:
                corrected_indices.append(pos.index)
                correction_details.append(
                    f"位置{pos.index + 1}: '{src_char}'→'{accepted_char}' ({decision_reason})"
                )
            else:
                correction_details.append(
                    f"位置{pos.index + 1}: '{src_char}'保持不变 ({decision_reason})"
                )

        corrected_text = "".join(result_chars)
        reason = (
            f"LLM verifier (targeted mode) processed {len(suspicious_positions)} positions and "
            f"accepted {len(corrected_indices)} edits. "
            f"Details: {'; '.join(correction_details)}"
        )

        return VerificationResult(
            text=corrected_text,
            selected_source='llm-targeted-correction',
            reason=reason,
            candidates=list(candidates),
        )

    @torch.no_grad()
    def _verify_targeted(
            self,
            prediction: SentencePrediction,
            candidates: Sequence[CandidateSentence],
            selected_positions: Optional[List[int]] = None,
    ) -> VerificationResult:
        """Hierarchical targeted verifier: stage1 choice+N, stage2 open repair, stage3 recheck."""
        suspicious_positions = self._get_suspicious_positions(
            prediction,
            selected_positions=selected_positions,
        )

        if not suspicious_positions:
            return VerificationResult(
                text=prediction.predicted_text,
                selected_source='front-end-top1',
                reason='No suspicious positions for targeted correction.',
                candidates=list(candidates),
            )

        llm_positions = [
            pos for pos in suspicious_positions
            if self._is_single_chinese_char(prediction.source_text[pos.index])
        ]
        correction_map = {}
        decision_reasons: Dict[int, str] = {}
        for pos in suspicious_positions:
            if pos not in llm_positions:
                correction_map[pos.index] = pos.predicted_token if pos.is_edited else pos.source_token
                decision_reasons[pos.index] = 'non_chinese_passthrough'

        active_positions = []
        active_options = []
        for pos in llm_positions:
            options = self._build_targeted_options(pos)
            if len(options) <= 1:
                correction_map[pos.index] = options[0] if options else pos.source_token
                decision_reasons[pos.index] = 'single_option_passthrough'
            else:
                active_positions.append(pos)
                active_options.append(options)

        if active_positions:
            stage1_decisions = self._run_stage1(prediction, active_positions, active_options)
            for decision in stage1_decisions:
                if decision.abstained:
                    correction_map[decision.position.index] = decision.position.source_token
                    decision_reasons[decision.position.index] = 'stage1_abstain_keep'
                else:
                    correction_map[decision.position.index] = decision.selected_token or decision.position.source_token
                    decision_reasons[decision.position.index] = 'accepted_stage1_choice'

            stage2_proposals = self._run_stage2(prediction, stage1_decisions)
            for proposal in stage2_proposals:
                if proposal.kept:
                    correction_map[proposal.position.index] = proposal.position.source_token
                    decision_reasons[proposal.position.index] = 'stage2_keep_source'

            stage3_results = self._run_stage3(prediction, stage1_decisions, stage2_proposals)
            for index, (final_token, reason) in stage3_results.items():
                correction_map[index] = final_token
                decision_reasons[index] = reason

        corrections = [
            correction_map.get(pos.index, pos.predicted_token if pos.is_edited else pos.source_token)
            for pos in suspicious_positions
        ]
        result = self._safe_merge_targeted_corrections(
            prediction,
            candidates,
            suspicious_positions,
            corrections,
            decision_reasons=decision_reasons,
        )
        stage1_abstain = sum(1 for reason in decision_reasons.values() if reason == 'stage1_abstain_keep')
        stage2_proposed = sum(1 for reason in decision_reasons.values() if reason == 'accepted_stage3_recheck')
        result.reason = (
            f"Hierarchical verifier: stage1_abstain={stage1_abstain}, "
            f"stage3_accepted={stage2_proposed}. {result.reason}"
        )
        result.selected_source = 'llm-targeted-hierarchical'
        return result

    # ──────────────────────────────────────────────────────
    # 统一入口
    # ──────────────────────────────────────────────────────

    def verify(
            self,
            prediction: SentencePrediction,
            candidates: Sequence[CandidateSentence],
            selected_positions: Optional[List[int]] = None,
    ) -> VerificationResult:
        """
        统一验证入口，根据 self.mode 调度到对应模式。
        """
        if self.mode == "choice":
            return self._verify_choice(prediction, candidates)
        else:
            return self._verify_targeted(
                prediction,
                candidates,
                selected_positions=selected_positions,
            )

    # 向后兼容：保留 build_prompt 方法
    def build_prompt(self, prediction, candidates):
        if self.mode == "choice":
            return self._build_choice_prompt(prediction, candidates)
        suspicious = self._get_suspicious_positions(prediction)
        return self._build_targeted_prompt(
            prediction,
            suspicious,
            allow_abstain=True,
            stage_name="第一阶段",
        )


# ──────────────────────────────────────────────────────────
# 6. 向后兼容：保留别名
# ──────────────────────────────────────────────────────────

LocalLLMVerifier = BaichuanLocalVerifier
