"""
Generate pipeline-aligned staged LLM LoRA training data.

Supported stages:
- stage1: closed-choice with abstention (A/B/C/.../N)
- stage2: restricted open repair (single-char proposal)
- stage3: closed-choice re-check after a proposed new char is introduced
"""

from __future__ import annotations

import argparse
import json
import math
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
        choices=["stage1", "stage2", "stage3"],
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
    total = 0
    saved = 0
    skipped_skip = 0
    skipped_len = 0
    skipped_no_positions = 0
    skipped_stage_empty = 0
    skipped_serialize_error = 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
