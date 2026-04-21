"""
Generate pipeline-aligned staged LLM LoRA training data.

Supported stages:
- stage1: closed-choice with abstention (A/B/C/.../N)
- unified: single-stage candidate-first decision (A/B/C/... or GEN:字)
- stage2: restricted open repair (single-char proposal)
- stage3: closed-choice re-check after a proposed new char is introduced
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import string
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from masc_csc.frontend_adapter import HFNamBertFrontendAdapter  # noqa: E402
from masc_csc.mechanism import MechanismInferencer  # noqa: E402
from masc_csc.selective_escalation import SelectiveEscalationRouter  # noqa: E402
from masc_csc.types import (  # noqa: E402
    ErrorMechanism,
    PositionPrediction,
    SentencePrediction,
    TokenAlternative,
)
from models.multimodal_frontend import MultimodalCSCFrontend  # noqa: E402
from utils.str_utils import is_chinese  # noqa: E402


MECHANISM_LABEL_ZH = {
    ErrorMechanism.PHONOLOGICAL: "音近错误（读音相似）",
    ErrorMechanism.VISUAL: "形近错误（字形相似）",
    ErrorMechanism.UNCERTAIN: "不确定",
}

NOISY_SUBSTITUTION_PAIRS = {
    ("的", "得"),
    ("得", "的"),
    ("余", "馀"),
    ("馀", "余"),
    ("作", "做"),
    ("做", "作"),
    ("惟", "唯"),
    ("唯", "惟"),
    ("他", "她"),
    ("她", "他"),
    ("妳", "你"),
    ("你", "妳"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate targeted closed-choice LoRA training data.")
    parser.add_argument("--frontend-path", type=str, required=True,
                        help="HF NamBERT 目录或旧版 frontend checkpoint 路径")
    parser.add_argument("--data-path", type=str, required=True,
                        help="输入数据路径；支持单个 jsonl、目录、或逗号分隔多个路径")
    parser.add_argument("--output", type=str, required=True, help="输出 JSONL 路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--task-stage",
        type=str,
        default="stage1",
        choices=["stage1", "unified", "stage2", "stage3"],
        help="生成哪一阶段的训练数据",
    )
    parser.add_argument("--router-ckpt", type=str, default=None, help="Selective router checkpoint，可选")
    parser.add_argument("--router-threshold", type=float, default=0.60)
    parser.add_argument("--proposal-budget", type=int, default=3, help="router top-r 特征参数")
    parser.add_argument("--max-options-per-position", type=int, default=4)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--include-skip-samples", action="store_true",
                        help="默认只保留 router 决定 invoke_llm 的样本；开启后也保留 skip 样本")
    parser.add_argument("--allow-missing-gold", action="store_true",
                        help="兼容旧版参数；staged 生成中 stage1 会保留为 N，stage2/3 会转成后续阶段样本")
    parser.add_argument("--stage1-skip-core-ratio", type=float, default=0.25,
                        help="stage1-v2 中保留 skip core-choice 样本的比例上限，相对于 invoke core-choice 样本数")
    parser.add_argument("--stage1-target-n-sample-ratio", type=float, default=0.15,
                        help="stage1-v2 目标中，包含 N 的样本占比")
    parser.add_argument("--stage1-easy-to-hard-ratio", type=float, default=2.0,
                        help="stage1-v2 中 synthetic easy-N 与 hard-N 的目标比例")
    parser.add_argument("--stage1-target-rare-ratio", type=float, default=0.06,
                        help="stage1-v2 中 rare-choice 样本相对 core-choice 的目标比例")
    parser.add_argument("--stage1-build-mode", type=str, default="v2", choices=["v2", "v3"],
                        help="stage1 数据构建模式：v2 为 easy-N/rare 版本，v3 为 router-invoke 四桶版本")
    parser.add_argument("--stage1-rebuild-from", type=str, default=None,
                        help="从已有 stage1 JSONL 直接重构 stage1 数据，跳过前端/router 重跑")
    parser.add_argument("--stage1-v3-keep-hard-ratio", type=float, default=0.35,
                        help="stage1-v3 中 keep-hard 桶的目标比例")
    parser.add_argument("--stage1-v3-choice-hard-ratio", type=float, default=0.40,
                        help="stage1-v3 中 choice-hard 桶的目标比例")
    parser.add_argument("--stage1-v3-abstain-hard-ratio", type=float, default=0.15,
                        help="stage1-v3 中 abstain-hard 桶的目标比例")
    parser.add_argument("--stage1-v3-noisy-negative-ratio", type=float, default=0.10,
                        help="stage1-v3 中 noisy-negative 桶的目标比例")
    parser.add_argument("--seed", type=int, default=42, help="stage1-v2 采样随机种子")
    return parser.parse_args()


def _split_input_paths(data_path: str) -> List[str]:
    parts = [item.strip() for item in data_path.split(",") if item.strip()]
    if not parts:
        raise ValueError("data_path 为空")
    return parts


def resolve_data_files(data_path: str) -> List[Path]:
    resolved: List[Path] = []
    for raw_path in _split_input_paths(data_path):
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"找不到数据路径：{raw_path}")

        if path.is_file():
            if path.suffix != ".jsonl":
                raise ValueError(f"当前仅支持 .jsonl 文件，收到：{path}")
            resolved.append(path.resolve())
            continue

        if path.is_dir():
            all_jsonl = sorted(p.resolve() for p in path.rglob("*.jsonl"))
            if not all_jsonl:
                raise FileNotFoundError(f"目录下没有找到任何 .jsonl 文件：{path}")
            train_like = [p for p in all_jsonl if "train" in p.stem.lower()]
            resolved.extend(train_like or all_jsonl)
            continue

        raise ValueError(f"不支持的数据路径：{raw_path}")

    deduped: List[Path] = []
    seen = set()
    for item in resolved:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def iter_records(data_files: Sequence[Path], limit: int = -1) -> Iterable[Tuple[str, str]]:
    count = 0
    for file_path in data_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                src = str(obj.get("source", obj.get("src", ""))).rstrip("\n")
                tgt = str(obj.get("target", obj.get("tgt", ""))).rstrip("\n")
                if not src or not tgt:
                    continue
                yield src, tgt
                count += 1
                if limit > 0 and count >= limit:
                    return


def load_frontend_model(frontend_path: str, device: str):
    path = Path(frontend_path)
    if path.is_dir():
        return HFNamBertFrontendAdapter(str(path), device)

    args = SimpleNamespace(device=device, hyper_params={})
    model = MultimodalCSCFrontend(args)
    state = torch.load(frontend_path, map_location="cpu")
    state_dict = state.get("state_dict", state)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def build_sentence_prediction(metadata: dict, mechanism_inferencer: MechanismInferencer) -> SentencePrediction:
    positions: List[PositionPrediction] = []
    for index, source_token in enumerate(metadata["source_tokens"]):
        alternative_tokens = metadata["topk_tokens"][index]
        alternative_ids = metadata["topk_ids"][index]
        alternative_scores = metadata["topk_probs"][index]
        alternatives = [
            TokenAlternative(token=token, token_id=token_id, score=float(score))
            for token, token_id, score in zip(alternative_tokens, alternative_ids, alternative_scores)
        ]
        mechanism = mechanism_inferencer.infer_from_alternatives(source_token, alternatives)
        positions.append(
            PositionPrediction(
                index=index,
                source_token=source_token,
                predicted_token=metadata["predicted_tokens"][index],
                detection_score=float(metadata["detection_scores"][index]),
                uncertainty=float(metadata["uncertainty_scores"][index]),
                mechanism=mechanism if isinstance(mechanism, ErrorMechanism) else ErrorMechanism.UNCERTAIN,
                alternatives=alternatives,
            )
        )

    return SentencePrediction(
        source_text=metadata["source_text"],
        predicted_text=metadata["predicted_text"],
        positions=positions,
    )


def get_suspicious_positions(prediction: SentencePrediction, selected_positions: Optional[List[int]] = None) -> List[PositionPrediction]:
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


def is_single_chinese_char(text: str) -> bool:
    return len(text) == 1 and is_chinese(text)


def build_targeted_options(position: PositionPrediction, max_options: int = 4) -> List[str]:
    options: List[str] = []
    seen = set()

    def add(token: str):
        if not is_single_chinese_char(token):
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


def build_targeted_prompt(
    prediction: SentencePrediction,
    suspicious_positions: List[PositionPrediction],
    option_lists: Optional[List[List[str]]] = None,
    allow_abstain: bool = False,
    stage_name: str = "第一阶段",
) -> str:
    source_text = prediction.source_text
    source_chars = list(source_text)

    suspicious_indices = set(p.index for p in suspicious_positions)
    display_chars = []
    for i, char in enumerate(source_chars):
        display_chars.append(f"[{char}]" if i in suspicious_indices else char)
    display_sentence = " ".join(display_chars)

    position_blocks = []
    for rank, pos in enumerate(suspicious_positions, 1):
        idx = pos.index
        char = source_chars[idx] if idx < len(source_chars) else "?"
        mech_label = MECHANISM_LABEL_ZH.get(pos.mechanism, "不确定")
        suggestion = pos.predicted_token
        options = option_lists[rank - 1] if option_lists is not None else build_targeted_options(pos)
        option_lines = []
        for option_index, token in enumerate(options):
            label = string.ascii_uppercase[option_index]
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
    return (
        f"{stage_name}：以下句子中只有标记位置可能存在拼写错误。\n"
        "你是审校器，不是改写器。\n"
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


def build_unified_prompt_from_sample(
    source_text: str,
    suspicious_indices: List[int],
    option_lists: List[List[str]],
) -> str:
    source_chars = list(source_text)
    suspicious_set = set(suspicious_indices)
    display_chars = []
    for i, char in enumerate(source_chars):
        display_chars.append(f"[{char}]" if i in suspicious_set else char)
    display_sentence = " ".join(display_chars)

    position_blocks = []
    for rank, (index, options) in enumerate(zip(suspicious_indices, option_lists), 1):
        char = source_chars[index] if 0 <= index < len(source_chars) else "?"
        option_lines = []
        for option_index, token in enumerate(options):
            label = string.ascii_uppercase[option_index]
            if token == char:
                option_lines.append(f"    {label}. 保持原字：{token}")
            else:
                option_lines.append(f"    {label}. 改为：{token}")
        position_blocks.append(
            f"[{rank}] 第{index + 1}字\"{char}\"\n"
            f"{chr(10).join(option_lines)}"
        )

    return (
        "统一决策阶段：以下句子中只有标记位置可能存在拼写错误。\n"
        "你是中文拼写纠错验证器，不是改写器。\n"
        "你的目标是对每个位置一次性做出最终决定。\n\n"
        "动作规则：\n"
        "1. 如果给定候选中已经有正确答案，必须直接输出对应选项字母。\n"
        "2. 只有当所有候选都不合适，并且该位置确实存在拼写错误时，才能输出 GEN:汉字。\n"
        "3. 禁止同义替换、风格润色、专名改写、代词性别替换、异体字/规范字替换。\n"
        "4. 除标记位置外，不要改动任何其他字符。\n"
        "5. 如果原字就是正确答案，请选择“保持原字”的那个候选字母，不要输出 KEEP。\n\n"
        f"句子：{display_sentence}\n\n"
        "可疑位置与候选：\n"
        f"{chr(10).join(position_blocks)}\n\n"
        "输出格式：按顺序输出每个位置的最终决策，用逗号分隔。\n"
        "合法格式只有两种：\n"
        "- 直接输出候选字母，例如 A 或 B\n"
        "- 输出 GEN:汉字，例如 GEN:座\n\n"
        "输出示例：A,B,GEN:座\n"
        "严格只输出最终结果，不要解释，不要输出序号、空格或其他任何文字。\n\n"
        "输出："
    )


def build_stage2_prompt(
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


def _sanitize_for_json(value):
    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        return value.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
    if hasattr(value, 'item'):
        try:
            return _sanitize_for_json(value.item())
        except Exception:
            pass
    return str(value).encode('utf-8', errors='replace').decode('utf-8', errors='replace')


def build_stage1_sample(
    source_text: str,
    target_text: str,
    prediction: SentencePrediction,
    suspicious_positions: List[PositionPrediction],
    max_options_per_position: int,
) -> Optional[dict]:
    if len(source_text) != len(target_text):
        return None

    active_positions: List[PositionPrediction] = []
    option_lists: List[List[str]] = []
    labels: List[str] = []

    for pos in suspicious_positions:
        if not is_single_chinese_char(source_text[pos.index]):
            continue

        options = build_targeted_options(pos, max_options=max_options_per_position)
        if len(options) <= 1:
            continue

        gold_char = target_text[pos.index]
        if gold_char not in options:
            label = "N"
        else:
            label = string.ascii_uppercase[options.index(gold_char)]

        active_positions.append(pos)
        option_lists.append(options)
        labels.append(label)

    if not active_positions:
        return None

    prompt = build_targeted_prompt(
        prediction,
        active_positions,
        option_lists,
        allow_abstain=True,
        stage_name="第一阶段",
    )
    num_non_keep = sum(1 for label in labels if label != "A")
    num_abstain = sum(1 for label in labels if label == "N")

    return {
        "task_type": "targeted_choice_abstain",
        "instruction": "请完成中文拼写纠错的定点闭集验证，只输出逗号分隔的字母序列；必要时可输出 N。",
        "input": prompt,
        "output": ",".join(labels),
        "source": source_text,
        "target": target_text,
        "frontend_prediction": prediction.predicted_text,
        "num_positions": len(active_positions),
        "num_non_keep": num_non_keep,
        "num_abstain": num_abstain,
        "all_keep": num_non_keep == 0,
        "sample_label": 0 if num_non_keep == 0 else 1,
        "suspicious_indices": [pos.index for pos in active_positions],
        "option_lists": option_lists,
    }


def build_stage1_instance(
    source_text: str,
    target_text: str,
    prediction: SentencePrediction,
    suspicious_positions: List[PositionPrediction],
    max_options_per_position: int,
    routing,
) -> Optional[dict]:
    if len(source_text) != len(target_text):
        return None

    active_positions: List[SimpleNamespace] = []
    option_lists: List[List[str]] = []
    labels: List[str] = []

    for pos in suspicious_positions:
        if not is_single_chinese_char(source_text[pos.index]):
            continue

        options = build_targeted_options(pos, max_options=max_options_per_position)
        if len(options) <= 1:
            continue

        gold_char = target_text[pos.index]
        if gold_char not in options:
            label = "N"
        else:
            label = string.ascii_uppercase[options.index(gold_char)]

        active_positions.append(
            SimpleNamespace(
                index=pos.index,
                source_token=pos.source_token,
                predicted_token=pos.predicted_token,
                mechanism=pos.mechanism,
            )
        )
        option_lists.append(list(options))
        labels.append(label)

    if not active_positions:
        return None

    return {
        "source_text": source_text,
        "target_text": target_text,
        "predicted_text": prediction.predicted_text,
        "active_positions": active_positions,
        "option_lists": option_lists,
        "labels": labels,
        "router_invoke": bool(routing.invoke_llm),
        "router_risk": float(routing.risk_score),
        "router_reasons": list(routing.reasons),
        "suspicious_indices": [pos.index for pos in active_positions],
    }


def classify_stage1_instance(instance: dict) -> str:
    labels = instance["labels"]
    if "N" in labels:
        return "hard-N"
    if any(label in {"C", "D"} for label in labels):
        return "rare-choice"
    return "core-choice"


def _stage1_prompt_from_instance(instance: dict, option_lists: List[List[str]]) -> str:
    prediction_stub = SimpleNamespace(source_text=instance["source_text"])
    return build_targeted_prompt(
        prediction_stub,
        instance["active_positions"],
        option_lists,
        allow_abstain=True,
        stage_name="第一阶段",
    )


def materialize_stage1_sample(instance: dict, sample_type: str, synthetic_n: bool = False) -> dict:
    option_lists = copy.deepcopy(instance["option_lists"])
    labels = list(instance["labels"])
    prompt = _stage1_prompt_from_instance(instance, option_lists)
    num_non_keep = sum(1 for label in labels if label != "A")
    num_abstain = sum(1 for label in labels if label == "N")
    return {
        "task_type": "targeted_choice_abstain",
        "instruction": "请完成中文拼写纠错的定点闭集验证，只输出逗号分隔的字母序列；必要时可输出 N。",
        "input": prompt,
        "output": ",".join(labels),
        "source": instance["source_text"],
        "target": instance["target_text"],
        "frontend_prediction": instance["predicted_text"],
        "num_positions": len(labels),
        "num_non_keep": num_non_keep,
        "num_abstain": num_abstain,
        "all_keep": num_non_keep == 0,
        "sample_label": 0 if num_non_keep == 0 else 1,
        "suspicious_indices": list(instance["suspicious_indices"]),
        "option_lists": option_lists,
        "sample_type": sample_type,
        "synthetic_n": synthetic_n,
        "contains_N": num_abstain > 0,
        "task_stage": "stage1",
        "router_invoke": instance["router_invoke"],
        "router_risk": instance["router_risk"],
        "router_reasons": list(instance["router_reasons"]),
    }


def build_stage1_easy_n_variants(instance: dict) -> List[dict]:
    variants: List[dict] = []
    for pos_idx, label in enumerate(instance["labels"]):
        if label not in {"B", "C", "D"}:
            continue
        gold_option_index = string.ascii_uppercase.index(label)
        options = instance["option_lists"][pos_idx]
        if gold_option_index >= len(options):
            continue

        mutated = {
            "source_text": instance["source_text"],
            "target_text": instance["target_text"],
            "predicted_text": instance["predicted_text"],
            "active_positions": instance["active_positions"],
            "option_lists": copy.deepcopy(instance["option_lists"]),
            "labels": list(instance["labels"]),
            "router_invoke": instance["router_invoke"],
            "router_risk": instance["router_risk"],
            "router_reasons": list(instance["router_reasons"]),
            "suspicious_indices": list(instance["suspicious_indices"]),
        }
        mutated["option_lists"][pos_idx].pop(gold_option_index)
        mutated["labels"][pos_idx] = "N"
        variants.append(mutated)
    return variants


def _sample_with_optional_replacement(items: List[dict], target_count: int, rng: random.Random) -> List[dict]:
    if target_count <= 0 or not items:
        return []
    if target_count <= len(items):
        return rng.sample(items, target_count)
    selected = list(items)
    selected.extend(rng.choice(items) for _ in range(target_count - len(items)))
    return selected


def _sample_without_replacement(items: List[dict], target_count: int, rng: random.Random) -> List[dict]:
    if target_count <= 0 or not items:
        return []
    if target_count >= len(items):
        return list(items)
    return rng.sample(items, target_count)


def summarize_stage1_samples(samples: List[dict], raw_counts: dict) -> dict:
    label_counter = {label: 0 for label in ["A", "B", "C", "D", "N"]}
    sample_type_counter = {}
    num_positions_counter = {}
    contains_n_counter = {"true": 0, "false": 0}
    router_invoke_counter = {"true": 0, "false": 0}

    total_positions = 0
    for sample in samples:
        sample_type = sample.get("sample_type", "unknown")
        sample_type_counter[sample_type] = sample_type_counter.get(sample_type, 0) + 1
        num_positions = int(sample.get("num_positions", 0))
        num_positions_counter[str(num_positions)] = num_positions_counter.get(str(num_positions), 0) + 1
        contains_n_counter["true" if sample.get("contains_N") else "false"] += 1
        router_invoke_counter["true" if sample.get("router_invoke") else "false"] += 1
        labels = [lb for lb in str(sample.get("output", "")).split(",") if lb]
        total_positions += len(labels)
        for lb in labels:
            label_counter[lb] = label_counter.get(lb, 0) + 1

    label_ratio = {
        label: (count / total_positions if total_positions else 0.0)
        for label, count in label_counter.items()
    }
    return {
        "raw_counts": raw_counts,
        "final_num_samples": len(samples),
        "final_total_positions": total_positions,
        "sample_type_counter": sample_type_counter,
        "num_positions_counter": num_positions_counter,
        "contains_n_counter": contains_n_counter,
        "router_invoke_counter": router_invoke_counter,
        "label_counter": label_counter,
        "label_ratio": label_ratio,
    }


def _split_labels(sample: dict) -> List[str]:
    return [label for label in str(sample.get("output", "")).split(",") if label]


def _active_position_tuples(sample: dict) -> List[Tuple[int, str, str, str]]:
    source_text = sample["source"]
    target_text = sample["target"]
    suspicious_indices = list(sample.get("suspicious_indices", []))
    labels = _split_labels(sample)
    tuples: List[Tuple[int, str, str, str]] = []
    for local_idx, index in enumerate(suspicious_indices):
        if not (0 <= index < len(source_text) and 0 <= index < len(target_text)):
            continue
        label = labels[local_idx] if local_idx < len(labels) else ""
        tuples.append((index, source_text[index], target_text[index], label))
    return tuples


def _noisy_label_positions(sample: dict) -> List[int]:
    positions: List[int] = []
    for local_idx, _, source_char, target_char, label in [
        (i, idx, src, tgt, lb) for i, (idx, src, tgt, lb) in enumerate(_active_position_tuples(sample))
    ]:
        if label in {"B", "C", "D"} and (source_char, target_char) in NOISY_SUBSTITUTION_PAIRS:
            positions.append(local_idx)
    return positions


def materialize_stage1_v3_sample(sample: dict, sample_type: str) -> dict:
    result = copy.deepcopy(sample)
    labels = _split_labels(result)
    if sample_type == "noisy-negative":
        for local_idx in _noisy_label_positions(result):
            if local_idx < len(labels):
                labels[local_idx] = "A"
    result["output"] = ",".join(labels)
    result["num_positions"] = len(labels)
    result["num_non_keep"] = sum(1 for label in labels if label != "A")
    result["num_abstain"] = sum(1 for label in labels if label == "N")
    result["all_keep"] = result["num_non_keep"] == 0
    result["sample_label"] = 0 if result["all_keep"] else 1
    result["contains_N"] = result["num_abstain"] > 0
    result["sample_type"] = sample_type
    result["synthetic_n"] = False
    result["task_stage"] = "stage1"
    result["stage1_build_mode"] = "v3"
    result["router_invoke"] = bool(result.get("router_invoke", False))
    return result


def _label_to_unified_action(
    label: str,
    option_list: List[str],
    target_text: str,
    index: int,
) -> Optional[str]:
    label = str(label).strip().upper()
    if not label:
        return None
    if label == "N":
        if not (0 <= index < len(target_text)):
            return None
        gold_char = target_text[index]
        if not is_single_chinese_char(gold_char):
            return None
        return f"GEN:{gold_char}"
    if label in string.ascii_uppercase:
        option_index = string.ascii_uppercase.index(label)
        if option_index < len(option_list):
            return label
    return None


def materialize_unified_sample(sample: dict) -> Optional[dict]:
    source_text = sample["source"]
    target_text = sample["target"]
    suspicious_indices = list(sample.get("suspicious_indices", []))
    option_lists = copy.deepcopy(sample.get("option_lists", []))
    labels = _split_labels(sample)
    if not suspicious_indices or not option_lists or len(suspicious_indices) != len(option_lists):
        return None

    actions: List[str] = []
    num_generate = 0
    for idx, option_list, label in zip(suspicious_indices, option_lists, labels):
        action = _label_to_unified_action(label, option_list, target_text, idx)
        if action is None:
            return None
        if action.startswith("GEN:"):
            num_generate += 1
        actions.append(action)

    prompt = build_unified_prompt_from_sample(
        source_text=source_text,
        suspicious_indices=suspicious_indices,
        option_lists=option_lists,
    )
    num_non_keep = sum(1 for action in actions if action != "A")
    return {
        "task_type": "targeted_unified",
        "instruction": "请完成中文拼写纠错的统一单阶段验证；优先选择候选，候选都不合适时才输出 GEN:字。",
        "input": prompt,
        "output": ",".join(actions),
        "source": source_text,
        "target": target_text,
        "frontend_prediction": sample.get("frontend_prediction", sample.get("predicted_text", "")),
        "num_positions": len(actions),
        "num_non_keep": num_non_keep,
        "num_generate": num_generate,
        "all_keep": num_non_keep == 0,
        "sample_label": 0 if num_non_keep == 0 else 1,
        "suspicious_indices": suspicious_indices,
        "option_lists": option_lists,
        "sample_type": sample.get("sample_type", "unknown"),
        "contains_gen": num_generate > 0,
        "task_stage": "unified",
        "router_invoke": bool(sample.get("router_invoke", False)),
        "router_risk": sample.get("router_risk"),
        "router_reasons": list(sample.get("router_reasons", [])),
    }


def summarize_unified_samples(samples: List[dict], raw_counts: dict) -> dict:
    action_counter = {"A": 0, "B": 0, "C": 0, "D": 0, "GEN": 0}
    sample_type_counter = {}
    num_positions_counter = {}
    contains_gen_counter = {"true": 0, "false": 0}
    router_invoke_counter = {"true": 0, "false": 0}

    total_positions = 0
    for sample in samples:
        sample_type = sample.get("sample_type", "unknown")
        sample_type_counter[sample_type] = sample_type_counter.get(sample_type, 0) + 1
        num_positions = int(sample.get("num_positions", 0))
        num_positions_counter[str(num_positions)] = num_positions_counter.get(str(num_positions), 0) + 1
        contains_gen_counter["true" if sample.get("contains_gen") else "false"] += 1
        router_invoke_counter["true" if sample.get("router_invoke") else "false"] += 1
        actions = [act for act in str(sample.get("output", "")).split(",") if act]
        total_positions += len(actions)
        for act in actions:
            if act.startswith("GEN:"):
                action_counter["GEN"] += 1
            elif act in action_counter:
                action_counter[act] += 1
            else:
                action_counter.setdefault(act, 0)
                action_counter[act] += 1

    action_ratio = {
        action: (count / total_positions if total_positions else 0.0)
        for action, count in action_counter.items()
    }
    return {
        "raw_counts": raw_counts,
        "final_num_samples": len(samples),
        "final_total_positions": total_positions,
        "sample_type_counter": sample_type_counter,
        "num_positions_counter": num_positions_counter,
        "contains_gen_counter": contains_gen_counter,
        "router_invoke_counter": router_invoke_counter,
        "action_counter": action_counter,
        "action_ratio": action_ratio,
    }


def classify_stage1_v3_sample(sample: dict) -> Optional[str]:
    if not bool(sample.get("router_invoke", False)):
        return None
    labels = _split_labels(sample)
    if not labels:
        return None
    if _noisy_label_positions(sample):
        return "noisy-negative"
    if "N" in labels:
        return "abstain-hard"
    if all(label == "A" for label in labels):
        return "keep-hard"
    return "choice-hard"


def summarize_stage1_bucket_profile(pools: dict[str, List[dict]]) -> dict:
    profile = {}
    for bucket, items in pools.items():
        label_counter = {label: 0 for label in ["A", "B", "C", "D", "N"]}
        total_positions = 0
        for item in items:
            labels = _split_labels(item)
            total_positions += len(labels)
            for label in labels:
                label_counter[label] = label_counter.get(label, 0) + 1
        profile[bucket] = {
            "samples": len(items),
            "positions": total_positions,
            "label_counter": label_counter,
        }
    return profile


def build_stage1_v3_dataset_from_samples(samples: List[dict], args) -> Tuple[List[dict], dict]:
    rng = random.Random(args.seed)

    pools = {
        "keep-hard": [],
        "choice-hard": [],
        "abstain-hard": [],
        "noisy-negative": [],
    }
    skipped_non_invoke = 0
    for sample in samples:
        bucket = classify_stage1_v3_sample(sample)
        if bucket is None:
            skipped_non_invoke += 1
            continue
        pools[bucket].append(sample)

    invoke_total = sum(len(items) for items in pools.values())
    ratio_config = {
        "keep-hard": max(args.stage1_v3_keep_hard_ratio, 0.0),
        "choice-hard": max(args.stage1_v3_choice_hard_ratio, 0.0),
        "abstain-hard": max(args.stage1_v3_abstain_hard_ratio, 0.0),
        "noisy-negative": max(args.stage1_v3_noisy_negative_ratio, 0.0),
    }
    ratio_sum = sum(ratio_config.values()) or 1.0
    normalized = {bucket: value / ratio_sum for bucket, value in ratio_config.items()}

    targets = {
        bucket: int(round(invoke_total * normalized[bucket]))
        for bucket in normalized
    }
    assigned = sum(targets.values())
    if assigned != invoke_total:
        targets["choice-hard"] += invoke_total - assigned

    final_samples: List[dict] = []
    final_bucket_counter = {}
    for bucket, target_count in targets.items():
        selected = _sample_with_optional_replacement(pools[bucket], target_count, rng)
        final_bucket_counter[bucket] = len(selected)
        for item in selected:
            final_samples.append(materialize_stage1_v3_sample(item, sample_type=bucket))

    rng.shuffle(final_samples)

    raw_counts = {
        "raw_total": len(samples),
        "raw_invoke_total": invoke_total,
        "raw_non_invoke_skipped": skipped_non_invoke,
        "target_bucket_counts": targets,
        "final_bucket_counts": final_bucket_counter,
    }
    summary = summarize_stage1_samples(final_samples, raw_counts)
    summary["raw_bucket_profile"] = summarize_stage1_bucket_profile(pools)
    summary["bucket_ratio_config"] = normalized
    return final_samples, summary


def build_stage1_v2_dataset(instances: List[dict], args) -> Tuple[List[dict], dict]:
    rng = random.Random(args.seed)

    core_all: List[dict] = []
    rare_all: List[dict] = []
    hard_n_all: List[dict] = []
    for instance in instances:
        bucket = classify_stage1_instance(instance)
        if bucket == "hard-N":
            hard_n_all.append(instance)
        elif bucket == "rare-choice":
            rare_all.append(instance)
        else:
            core_all.append(instance)

    core_invoke = [inst for inst in core_all if inst["router_invoke"]]
    core_skip = [inst for inst in core_all if not inst["router_invoke"]]
    max_skip_core = int(round(len(core_invoke) * max(args.stage1_skip_core_ratio, 0.0)))
    retained_skip = _sample_without_replacement(core_skip, max_skip_core, rng)
    core_selected = list(core_invoke) + retained_skip

    rare_target = max(len(rare_all), int(round(len(core_selected) * max(args.stage1_target_rare_ratio, 0.0))))
    rare_selected = _sample_with_optional_replacement(rare_all, rare_target, rng)
    hard_selected = list(hard_n_all)

    easy_candidates: List[dict] = []
    for instance in core_selected + rare_selected:
        easy_candidates.extend(build_stage1_easy_n_variants(instance))

    base_non_n_count = len(core_selected) + len(rare_selected)
    hard_n_count = len(hard_selected)
    target_n_ratio = min(max(args.stage1_target_n_sample_ratio, 0.0), 0.95)
    easy_target_from_ratio = 0
    if target_n_ratio > 0:
        numerator = target_n_ratio * (base_non_n_count + hard_n_count) - hard_n_count
        easy_target_from_ratio = max(0, math.ceil(numerator / max(1e-6, 1.0 - target_n_ratio)))
    easy_target_from_hard = int(round(hard_n_count * max(args.stage1_easy_to_hard_ratio, 0.0)))
    easy_target = max(easy_target_from_ratio, easy_target_from_hard)
    easy_selected = _sample_with_optional_replacement(easy_candidates, easy_target, rng)

    samples: List[dict] = []
    samples.extend(materialize_stage1_sample(instance, sample_type="core-choice") for instance in core_selected)
    samples.extend(materialize_stage1_sample(instance, sample_type="rare-choice") for instance in rare_selected)
    samples.extend(materialize_stage1_sample(instance, sample_type="hard-N") for instance in hard_selected)
    samples.extend(materialize_stage1_sample(instance, sample_type="easy-N", synthetic_n=True) for instance in easy_selected)
    rng.shuffle(samples)

    raw_counts = {
        "raw_total": len(instances),
        "raw_core_choice": len(core_all),
        "raw_rare_choice": len(rare_all),
        "raw_hard_n": len(hard_n_all),
        "core_invoke": len(core_invoke),
        "core_skip": len(core_skip),
        "core_skip_retained": len(retained_skip),
        "rare_selected": len(rare_selected),
        "hard_selected": len(hard_selected),
        "easy_candidates": len(easy_candidates),
        "easy_selected": len(easy_selected),
        "easy_target_from_ratio": easy_target_from_ratio,
        "easy_target_from_hard": easy_target_from_hard,
        "easy_target_final": easy_target,
    }
    return samples, summarize_stage1_samples(samples, raw_counts)


def build_stage2_samples(
    source_text: str,
    target_text: str,
    prediction: SentencePrediction,
    suspicious_positions: List[PositionPrediction],
    max_options_per_position: int,
) -> List[dict]:
    if len(source_text) != len(target_text):
        return []

    samples: List[dict] = []
    for pos in suspicious_positions:
        if not is_single_chinese_char(source_text[pos.index]):
            continue
        options = build_targeted_options(pos, max_options=max_options_per_position)
        if len(options) <= 1:
            continue

        gold_char = target_text[pos.index]
        if gold_char == source_text[pos.index]:
            continue
        if gold_char in options:
            continue

        samples.append({
            "task_type": "targeted_open_repair",
            "instruction": "请完成中文拼写纠错的候选外补充修复，只输出一个汉字。",
            "input": build_stage2_prompt(prediction, pos, options),
            "output": gold_char,
            "source": source_text,
            "target": target_text,
            "position_index": pos.index,
            "source_char": source_text[pos.index],
            "target_char": gold_char,
            "option_list": options,
            "sample_label": 1 if gold_char != source_text[pos.index] else 0,
        })
    return samples


def build_stage3_samples(
    source_text: str,
    target_text: str,
    prediction: SentencePrediction,
    suspicious_positions: List[PositionPrediction],
    max_options_per_position: int,
) -> List[dict]:
    if len(source_text) != len(target_text):
        return []

    samples: List[dict] = []
    for pos in suspicious_positions:
        if not is_single_chinese_char(source_text[pos.index]):
            continue
        options = build_targeted_options(pos, max_options=max_options_per_position)
        if len(options) <= 1:
            continue

        gold_char = target_text[pos.index]
        if gold_char == source_text[pos.index]:
            continue
        if gold_char in options:
            continue

        recheck_options = list(options)
        recheck_options.append(gold_char)
        prompt = build_targeted_prompt(
            prediction,
            [pos],
            [recheck_options],
            allow_abstain=False,
            stage_name="第三阶段",
        )
        samples.append({
            "task_type": "targeted_recheck",
            "instruction": "请完成中文拼写纠错的补充候选复核，只输出字母选项。",
            "input": prompt,
            "output": string.ascii_uppercase[len(recheck_options) - 1],
            "source": source_text,
            "target": target_text,
            "position_index": pos.index,
            "source_char": source_text[pos.index],
            "target_char": gold_char,
            "proposal_char": gold_char,
            "option_list": recheck_options,
            "sample_label": 1,
        })

        distractor_char = None
        for alternative in pos.alternatives:
            token = alternative.token
            if not is_single_chinese_char(token):
                continue
            if token in options or token == gold_char:
                continue
            distractor_char = token
            break

        if distractor_char is not None:
            reject_options = list(options)
            reject_options.append(distractor_char)
            reject_prompt = build_targeted_prompt(
                prediction,
                [pos],
                [reject_options],
                allow_abstain=False,
                stage_name="第三阶段",
            )
            samples.append({
                "task_type": "targeted_recheck",
                "instruction": "请完成中文拼写纠错的补充候选复核，只输出字母选项。",
                "input": reject_prompt,
                "output": "A",
                "source": source_text,
                "target": target_text,
                "position_index": pos.index,
                "source_char": source_text[pos.index],
                "target_char": gold_char,
                "proposal_char": distractor_char,
                "option_list": reject_options,
                "sample_label": 0,
            })
    return samples


def main():
    args = parse_args()

    total = 0
    saved = 0
    skipped_skip = 0
    skipped_len = 0
    skipped_no_positions = 0
    skipped_stage_empty = 0
    skipped_serialize_error = 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.task_stage in {"stage1", "unified"}:
        if args.stage1_rebuild_from:
            rebuild_path = Path(args.stage1_rebuild_from)
            if not rebuild_path.exists():
                raise FileNotFoundError(f"找不到 stage1 重构输入文件：{rebuild_path}")
            with open(rebuild_path, "r", encoding="utf-8") as fin:
                rebuild_samples = [json.loads(line) for line in fin if line.strip()]
            total = len(rebuild_samples)
            if args.task_stage == "stage1":
                if args.stage1_build_mode != "v3":
                    raise ValueError("当前从已有 stage1 JSONL 重构仅支持 --stage1-build-mode v3")
                final_samples, final_summary = build_stage1_v3_dataset_from_samples(rebuild_samples, args)
                generation_mode = "stage1-v3"
            else:
                final_samples = []
                skipped_unified_invalid = 0
                for sample in rebuild_samples:
                    unified_sample = materialize_unified_sample(sample)
                    if unified_sample is None:
                        skipped_unified_invalid += 1
                        continue
                    final_samples.append(unified_sample)
                final_summary = summarize_unified_samples(
                    final_samples,
                    raw_counts={
                        "raw_total": len(rebuild_samples),
                        "raw_invalid_skipped": skipped_unified_invalid,
                    },
                )
                generation_mode = "unified-rebuild"
            with open(output_path, "w", encoding="utf-8") as fout:
                for sample in final_samples:
                    fout.write(json.dumps(_sanitize_for_json(sample), ensure_ascii=False) + "\n")
            summary_path = Path(str(output_path) + ".summary.json")
            full_summary = {
                "task_stage": args.task_stage,
                "generation_mode": generation_mode,
                "rebuild_from": str(rebuild_path),
                "total_records_seen": total,
                "saved": len(final_samples),
                "summary": final_summary,
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(_sanitize_for_json(full_summary), f, ensure_ascii=False, indent=2)

            print(f"\n[*] {args.task_stage} rebuild completed:")
            print(f"  Source samples: {total}")
            print(f"  Saved: {len(final_samples)}")
            print(f"  Summary: {summary_path}")
            return

        print("[*] Loading frontend...")
        frontend_model = load_frontend_model(args.frontend_path, args.device)
        mechanism_inferencer = MechanismInferencer()
        router = SelectiveEscalationRouter(
            checkpoint_path=args.router_ckpt,
            llm_invoke_threshold=args.router_threshold,
            proposal_budget=args.proposal_budget,
            device=args.device,
        )
        data_files = resolve_data_files(args.data_path)
        stage1_instances: List[dict] = []
        for src, tgt in tqdm(iter_records(data_files, limit=args.limit), desc="Collecting stage1 instance pool"):
            total += 1
            if len(src) != len(tgt):
                skipped_len += 1
                continue

            metadata = frontend_model.predict_with_metadata(src, top_k=args.top_k)
            prediction = build_sentence_prediction(metadata, mechanism_inferencer)
            routing = router.decide(prediction)

            if (not args.include_skip_samples) and (not routing.invoke_llm):
                skipped_skip += 1
                continue

            suspicious_positions = get_suspicious_positions(prediction, selected_positions=routing.supplement_positions)
            if not suspicious_positions:
                skipped_no_positions += 1
                continue

            instance = build_stage1_instance(
                source_text=src,
                target_text=tgt,
                prediction=prediction,
                suspicious_positions=suspicious_positions,
                max_options_per_position=args.max_options_per_position,
                routing=routing,
            )
            if instance is None:
                skipped_stage_empty += 1
                continue
            stage1_instances.append(instance)

        if args.stage1_build_mode == "v2":
            stage1_samples, stage1_summary = build_stage1_v2_dataset(stage1_instances, args)
            base_generation_mode = "stage1-v2"
        else:
            base_stage1_samples = [
                materialize_stage1_sample(instance, sample_type="raw-stage1")
                for instance in stage1_instances
            ]
            stage1_samples, stage1_summary = build_stage1_v3_dataset_from_samples(base_stage1_samples, args)
            base_generation_mode = "stage1-v3"

        if args.task_stage == "stage1":
            final_samples = stage1_samples
            final_summary = stage1_summary
            generation_mode = base_generation_mode
        else:
            unified_samples: List[dict] = []
            skipped_unified_invalid = 0
            for sample in stage1_samples:
                unified_sample = materialize_unified_sample(sample)
                if unified_sample is None:
                    skipped_unified_invalid += 1
                    continue
                unified_samples.append(unified_sample)
            final_samples = unified_samples
            final_summary = summarize_unified_samples(
                unified_samples,
                raw_counts={
                    "base_generation_mode": base_generation_mode,
                    "base_saved": len(stage1_samples),
                    "base_skipped_unified_invalid": skipped_unified_invalid,
                },
            )
            generation_mode = f"unified-from-{base_generation_mode}"
        with open(output_path, "w", encoding="utf-8") as fout:
            for sample in final_samples:
                safe_sample = _sanitize_for_json(sample)
                fout.write(json.dumps(safe_sample, ensure_ascii=False) + "\n")
        summary_path = Path(str(output_path) + ".summary.json")
        full_summary = {
            "task_stage": args.task_stage,
            "generation_mode": generation_mode,
            "total_records_seen": total,
            "saved": len(final_samples),
            "skipped_len": skipped_len,
            "skipped_skip": skipped_skip,
            "skipped_no_positions": skipped_no_positions,
            "skipped_stage_empty": skipped_stage_empty,
            "summary": final_summary,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(_sanitize_for_json(full_summary), f, ensure_ascii=False, indent=2)

        print(f"\n[*] {generation_mode} generation completed:")
        print(f"  Total processed: {total}")
        print(f"  Saved: {len(final_samples)}")
        print(f"  Summary: {summary_path}")
        return

    print("[*] Loading frontend...")
    frontend_model = load_frontend_model(args.frontend_path, args.device)
    mechanism_inferencer = MechanismInferencer()
    router = SelectiveEscalationRouter(
        checkpoint_path=args.router_ckpt,
        llm_invoke_threshold=args.router_threshold,
        proposal_budget=args.proposal_budget,
        device=args.device,
    )
    data_files = resolve_data_files(args.data_path)

    with open(output_path, "w", encoding="utf-8") as fout:
        for src, tgt in tqdm(iter_records(data_files, limit=args.limit), desc="Generating targeted-choice data"):
            total += 1
            if len(src) != len(tgt):
                skipped_len += 1
                continue

            metadata = frontend_model.predict_with_metadata(src, top_k=args.top_k)
            prediction = build_sentence_prediction(metadata, mechanism_inferencer)
            routing = router.decide(prediction)

            if (not args.include_skip_samples) and (not routing.invoke_llm):
                skipped_skip += 1
                continue

            suspicious_positions = get_suspicious_positions(prediction, selected_positions=routing.supplement_positions)
            if not suspicious_positions:
                skipped_no_positions += 1
                continue

            samples: List[dict] = []
            if args.task_stage == "stage1":
                sample = build_stage1_sample(
                    source_text=src,
                    target_text=tgt,
                    prediction=prediction,
                    suspicious_positions=suspicious_positions,
                    max_options_per_position=args.max_options_per_position,
                )
                if sample is None:
                    skipped_stage_empty += 1
                    continue
                samples = [sample]
            elif args.task_stage == "stage2":
                samples = build_stage2_samples(
                    source_text=src,
                    target_text=tgt,
                    prediction=prediction,
                    suspicious_positions=suspicious_positions,
                    max_options_per_position=args.max_options_per_position,
                )
                if not samples:
                    skipped_stage_empty += 1
                    continue
            else:
                samples = build_stage3_samples(
                    source_text=src,
                    target_text=tgt,
                    prediction=prediction,
                    suspicious_positions=suspicious_positions,
                    max_options_per_position=args.max_options_per_position,
                )
                if not samples:
                    skipped_stage_empty += 1
                    continue

            for sample in samples:
                sample["task_stage"] = args.task_stage
                sample["router_invoke"] = bool(routing.invoke_llm)
                sample["router_risk"] = float(routing.risk_score)
                sample["router_reasons"] = list(routing.reasons)

                try:
                    safe_sample = _sanitize_for_json(sample)
                    fout.write(json.dumps(safe_sample, ensure_ascii=False) + "\n")
                    saved += 1
                except Exception as exc:
                    skipped_serialize_error += 1
                    print(f"[SerializeError] sample #{total}: {type(exc).__name__}: {exc}")
                    continue

    print(f"\n=== Staged LoRA Data Generation Complete ({args.task_stage}) ===")
    print(f"Total processed:              {total}")
    print(f"Saved:                        {saved}")
    print(f"Skipped by router skip:       {skipped_skip}")
    print(f"Skipped length mismatch:      {skipped_len}")
    print(f"Skipped no active positions:  {skipped_no_positions}")
    print(f"Skipped stage-empty cases:    {skipped_stage_empty}")
    print(f"Skipped serialize errors:     {skipped_serialize_error}")
    print(f"Output file:                  {output_path}")


if __name__ == "__main__":
    main()
