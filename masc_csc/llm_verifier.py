"""
MASC-CSC Local LLM Verifier
============================
纯本地部署的大语言模型验证器，支持：
- Baichuan-7B (或其他 HuggingFace 兼容模型) 本地推理
- LoRA adapter 动态加载
- ConstrainedChoiceLogitsProcessor 约束解码（强制只输出选项字母）
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
    VerificationResult,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# 1. 约束解码器：强制 LLM 只能输出有效选项字母
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
# 2. NoOp 验证器 (无 LLM 时的降级方案，保持不变)
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
# 3. 机制名称映射（中文，用于 Prompt 展示）
# ──────────────────────────────────────────────────────────

MECHANISM_LABEL_ZH: Dict[ErrorMechanism, str] = {
    ErrorMechanism.PHONOLOGICAL: "音近错误（读音相似）",
    ErrorMechanism.VISUAL: "形近错误（字形相似）",
    ErrorMechanism.UNCERTAIN: "不确定",
}


# ──────────────────────────────────────────────────────────
# 4. Baichuan 本地验证器（核心）
# ──────────────────────────────────────────────────────────

class BaichuanLocalVerifier:
    """
    纯本地部署的 Baichuan-7B 验证器。

    特性：
    - 使用 HuggingFace Transformers 原生加载模型，不走 API
    - 支持 LoRA adapter 热加载（用 peft 库）
    - 通过 ConstrainedChoiceLogitsProcessor 实现约束解码
    - Prompt 包含错误机制标注和变更位置信息
    """

    def __init__(
            self,
            model_path: str = "baichuan-inc/Baichuan-7B",
            adapter_path: Optional[str] = None,
            device: str = "cuda",
            torch_dtype=torch.float16,
            max_new_tokens: int = 1,
    ):
        """
        Args:
            model_path: Baichuan-7B 模型的本地路径或 HuggingFace Hub ID
            adapter_path: LoRA adapter 的路径（可选，微调后使用）
            device: 推理设备 ('cuda' / 'cpu')
            torch_dtype: 模型精度 (torch.float16 / torch.bfloat16)
            max_new_tokens: 生成的最大 token 数（选择题只需 1）
        """
        logger.info("Loading Baichuan model from: %s", model_path)
        self.device = device
        self.max_new_tokens = max_new_tokens

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

        # 预计算有效选项的 token ids (A, B, C, D, E)
        self._option_token_ids = self._resolve_option_token_ids()
        logger.info("Valid option token IDs: %s", self._option_token_ids)

    def _resolve_option_token_ids(self) -> List[int]:
        """获取选项字母 A-E 对应的 token id（兼容不同 tokenizer）"""
        option_ids = []
        for letter in ['A', 'B', 'C', 'D', 'E']:
            ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if ids:
                option_ids.append(ids[0])
        return option_ids

    @staticmethod
    def _label(index: int) -> str:
        return string.ascii_uppercase[index]

    def build_prompt(
            self,
            prediction: SentencePrediction,
            candidates: Sequence[CandidateSentence],
    ) -> str:
        """
        构建带有错误机制标注的增强型 Prompt。

        示例输出:
        ────────────────────
        源句子："我喜换吃平果"
        检测到的潜在错误：
          - 位置3 ("换")：音近错误（读音相似）
          - 位置5 ("平")：形近错误（字形相似）

        候选修正方案：
        A. 我喜换吃平果（保持原句不变）
        B. 我喜欢吃平果 [修改位置: 3]
        C. 我喜欢吃苹果 [修改位置: 3, 5]

        请从以上候选方案中选择最合适的修正结果。
        严格只输出一个字母选项（如 A、B、C），不要输出其他任何内容。
        ────────────────────
        """
        source_text = prediction.source_text
        source_chars = list(source_text)

        # ---- 错误位置分析 ----
        error_lines = []
        for position in prediction.edited_positions:
            idx = position.index
            char = source_chars[idx] if idx < len(source_chars) else "?"
            mech_label = MECHANISM_LABEL_ZH.get(position.mechanism, "不确定")
            error_lines.append(f"  - 位置{idx + 1} (\"{char}\")：{mech_label}")

        error_section = ""
        if error_lines:
            error_section = "检测到的潜在错误：\n" + "\n".join(error_lines) + "\n\n"

        # ---- 候选列表 ----
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
        # fallback: 选分数最高的
        logger.warning("Failed to parse LLM choice '%s', falling back to highest score.", generated_text)
        return max(range(len(candidates)), key=lambda i: candidates[i].score)

    @torch.no_grad()
    def verify(
            self,
            prediction: SentencePrediction,
            candidates: Sequence[CandidateSentence],
    ) -> VerificationResult:
        """
        调用本地 Baichuan-7B 进行约束解码验证。
        通过 ConstrainedChoiceLogitsProcessor 确保输出只可能是 A/B/C/D/E。
        """
        prompt = self.build_prompt(prediction, candidates)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)

        # 构建约束解码器 —— 限制有效选项数量
        num_candidates = len(candidates)
        valid_ids = self._option_token_ids[:num_candidates]
        logits_processor = LogitsProcessorList([
            ConstrainedChoiceLogitsProcessor(valid_ids),
        ])

        # 生成（只需 1 个 token）
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            logits_processor=logits_processor,
            do_sample=False,  # 贪心解码
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # 提取生成的 token（去掉 prompt 部分）
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
            reason=f"Baichuan local verifier selected option {choice_label} via constrained decoding.",
            candidates=list(candidates),
        )


# ──────────────────────────────────────────────────────────
# 5. 向后兼容：保留 LocalLLMVerifier 别名
# ──────────────────────────────────────────────────────────

# 为了不破坏已有的 import 代码，保留别名
LocalLLMVerifier = BaichuanLocalVerifier
