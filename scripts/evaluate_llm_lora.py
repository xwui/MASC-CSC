import argparse
import json
import os
import string
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _ensure_cuda_runtime_env():
    env = os.environ.copy()
    updated = False

    required_lib_dirs = [
        "/home/bkai/anaconda3/envs/cwq_masc_csc/lib",
        "/home/bkai/anaconda3/envs/cwq_masc_csc/lib/python3.10/site-packages/nvidia/cusparse/lib",
        "/home/bkai/anaconda3/envs/cwq_masc_csc/lib/python3.10/site-packages/nvidia/cuda_runtime/lib",
    ]
    current_ld = env.get("LD_LIBRARY_PATH", "")
    ld_parts = [part for part in current_ld.split(":") if part]
    for lib_dir in reversed(required_lib_dirs):
        if lib_dir and lib_dir not in ld_parts:
            ld_parts.insert(0, lib_dir)
            updated = True
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)

    current_pythonpath = env.get("PYTHONPATH", "")
    py_parts = [part for part in current_pythonpath.split(":") if part]
    if "/home/bkai/cwq/packages" not in py_parts:
        py_parts.insert(0, "/home/bkai/cwq/packages")
        env["PYTHONPATH"] = ":".join(py_parts)
        updated = True

    if env.get("TOKENIZERS_PARALLELISM") != "false":
        env["TOKENIZERS_PARALLELISM"] = "false"
        updated = True

    if updated and env.get("MASC_EVAL_ENV_READY") != "1":
        env["MASC_EVAL_ENV_READY"] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)

    os.environ.update(
        {
            "LD_LIBRARY_PATH": env["LD_LIBRARY_PATH"],
            "PYTHONPATH": env.get("PYTHONPATH", ""),
            "TOKENIZERS_PARALLELISM": env["TOKENIZERS_PARALLELISM"],
            "MASC_EVAL_ENV_READY": env.get("MASC_EVAL_ENV_READY", "0"),
        }
    )


_ensure_cuda_runtime_env()

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from transformers import modeling_utils as _transformers_modeling_utils
except Exception:
    _transformers_modeling_utils = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.metrics import SighanCSCMetrics  # noqa: E402

DEFAULT_CORRECTION_PROMPT = (
    "请纠正下面句子中的中文拼写错误；如果原句没有错误，保持原句不变。\n"
    "只输出纠正后的句子，不要添加解释。"
)

CORRECTION_PREFIXES = (
    "以下为纠正后的句子：",
    "以下是纠正后的句子：",
    "以下为修正后的句子：",
    "以下是修正后的句子：",
    "纠正后的句子：",
    "修正后的句子：",
    "改正后的句子：",
    "改正后：",
    "修正后：",
    "正确句子：",
    "答案：",
    "输出：",
)
CORRECTION_SUFFIXES = (
    "(Corrected)",
    "（Corrected）",
    "(corrected)",
    "（corrected）",
)
CHAT_MARKERS = ("User:", "Human:", "Assistant:", "User", "Human", "Assistant")


def _bnb_getitem(self, key):
    return getattr(self, key)


def _bnb_contains(self, key):
    return hasattr(self, key)


BitsAndBytesConfig.__getitem__ = _bnb_getitem
BitsAndBytesConfig.__contains__ = _bnb_contains
BitsAndBytesConfig.get = lambda self, key, default=None: getattr(self, key, default)


def _patch_dispatch_model_for_4bit():
    if _transformers_modeling_utils is None:
        return

    original_dispatch_model = getattr(_transformers_modeling_utils, "dispatch_model", None)
    if original_dispatch_model is None or getattr(original_dispatch_model, "_masc_safe_4bit_patch", False):
        return

    def safe_dispatch_model(model, device_map=None, *args, **kwargs):
        if isinstance(device_map, dict) and device_map and len(set(device_map.values())) == 1:
            # 4-bit 单卡加载时，避免 accelerate 再次调用 .to()。
            model.hf_device_map = dict(device_map)
            return model
        return original_dispatch_model(model, device_map=device_map, *args, **kwargs)

    safe_dispatch_model._masc_safe_4bit_patch = True
    _transformers_modeling_utils.dispatch_model = safe_dispatch_model


def _patch_qwen2_rotary_embedding_device():
    try:
        from transformers.models.qwen2 import modeling_qwen2 as qwen2_modeling
    except Exception:
        return

    rotary_cls = getattr(qwen2_modeling, "Qwen2RotaryEmbedding", None)
    if rotary_cls is None:
        return

    original_forward = getattr(rotary_cls, "forward", None)
    if original_forward is None or getattr(original_forward, "_masc_rotary_device_patch", False):
        return

    # 修改为支持动态参数 (*args, **kwargs) 的兼容写法
    def safe_forward(self, x, *args, **kwargs):
        target_device = x.device

        if hasattr(self, "inv_freq") and self.inv_freq.device != target_device:
            self.inv_freq = self.inv_freq.to(target_device)

        # 尝试提取真实的 seq_len
        actual_seq_len = x.shape[-2]
        
        # 判断传进来的位置参数是旧版(int)还是新版(Tensor)
        if len(args) > 0:
            if isinstance(args[0], torch.Tensor):
                actual_seq_len = torch.max(args[0]).item() + 1
            elif isinstance(args[0], int):
                actual_seq_len = args[0]
        
        # 处理可能的关键字传参
        if "position_ids" in kwargs and isinstance(kwargs["position_ids"], torch.Tensor):
            actual_seq_len = torch.max(kwargs["position_ids"]).item() + 1
        elif "seq_len" in kwargs and isinstance(kwargs["seq_len"], int):
            actual_seq_len = kwargs["seq_len"]

        if (
            actual_seq_len > self.max_seq_len_cached
            or self.cos_cached.device != target_device
            or self.sin_cached.device != target_device
        ):
            self._set_cos_sin_cache(
                seq_len=max(actual_seq_len, self.max_seq_len_cached),
                device=target_device,
                dtype=x.dtype,
            )

        return original_forward(self, x, *args, **kwargs)

    safe_forward._masc_rotary_device_patch = True
    rotary_cls.forward = safe_forward

def _get_model_device(model):
    for param in model.parameters():
        return param.device
    for buffer in model.buffers():
        return buffer.device
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="评测 base model + LoRA adapter 的 LLM 权重")
    parser.add_argument("--base_model", required=True, help="基座模型目录或 HuggingFace ID")
    parser.add_argument("--adapter_path", default=None, help="LoRA adapter 目录；不传则只测基座模型")
    parser.add_argument("--data_path", required=True, help="测试数据路径；支持单个 jsonl、目录、或逗号分隔多个路径")
    parser.add_argument("--output_dir", default="./outputs/llm_eval", help="评测输出目录")
    parser.add_argument(
        "--task_type",
        default="auto",
        choices=[
            "auto",
            "correction",
            "choice",
            "targeted_choice",
            "targeted_choice_abstain",
            "targeted_open_repair",
            "targeted_recheck",
        ],
        help="auto 自动识别；其余分别对应不同 verifier 训练任务",
    )
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--limit", type=int, default=-1, help="只评测前 N 条，-1 表示全量")
    parser.add_argument("--print_every", type=int, default=50, help="每多少条打印一次进度样例")
    return parser.parse_args()


def resolve_data_files(data_path: str) -> List[Path]:
    parts = [item.strip() for item in data_path.split(",") if item.strip()]
    if not parts:
        raise ValueError("data_path 为空")

    resolved = []
    for raw_path in parts:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"找不到数据路径：{raw_path}")

        if path.is_file():
            if path.suffix != ".jsonl":
                raise ValueError(f"当前仅支持 .jsonl 文件，收到：{path}")
            resolved.append(path)
            continue

        if path.is_dir():
            all_jsonl = sorted(path.rglob("*.jsonl"))
            if not all_jsonl:
                raise FileNotFoundError(f"目录下没有找到任何 .jsonl 文件：{path}")
            resolved.extend(all_jsonl)
            continue

        raise ValueError(f"不支持的数据路径：{raw_path}")

    deduped = []
    seen = set()
    for item in resolved:
        item = item.resolve()
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def load_jsonl_records(data_path: str, limit: int = -1) -> List[dict]:
    files = resolve_data_files(data_path)
    records = []
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
                if limit > 0 and len(records) >= limit:
                    return records
    return records


def detect_task_type(sample: dict, user_task_type: str) -> str:
    if user_task_type != "auto":
        return user_task_type

    task_type = str(sample.get("task_type", "")).strip().lower()
    if task_type == "targeted_choice":
        return "targeted_choice"
    if task_type == "targeted_choice_abstain":
        return "targeted_choice_abstain"
    if task_type == "targeted_open_repair":
        return "targeted_open_repair"
    if task_type == "targeted_recheck":
        return "targeted_recheck"

    if "source" in sample and "target" in sample:
        return "correction"
    if "instruction" in sample and "output" in sample:
        return "choice"

    raise ValueError(
        "无法自动识别任务类型。样本需要包含 {source,target}、{instruction,output}，"
        "或显式包含 task_type=targeted_choice；或者手动传 --task_type correction|choice|targeted_choice。"
    )


def build_correction_prompt(source: str) -> str:
    return (
        f"User:\n"
        f"{DEFAULT_CORRECTION_PROMPT}\n"
        f"原句：{source.strip()}\n\n"
        f"Assistant:"
    )


def build_choice_prompt(example: dict) -> str:
    instruction = str(example.get("instruction", "")).strip()
    input_text = str(example.get("input", "")).strip()

    if input_text:
        return f"User:\n{instruction}\n\n{input_text}\n\nAssistant:"
    return f"User:\n{instruction}\n\nAssistant:"


def build_targeted_choice_prompt(example: dict) -> str:
    return build_choice_prompt(example)


def _truncate_at_chat_marker(text: str) -> str:
    cut_positions = []
    for marker in CHAT_MARKERS:
        start = 0
        while True:
            pos = text.find(marker, start)
            if pos < 0:
                break
            if pos > 0 and not text[pos - 1].isalpha():
                cut_positions.append(pos)
                break
            start = pos + len(marker)

    if cut_positions:
        text = text[:min(cut_positions)]
    return text.strip()


def normalize_generation(text: str) -> str:
    text = text.strip()
    if text.startswith("Assistant:"):
        text = text[len("Assistant:"):].strip()
    text = _truncate_at_chat_marker(text)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    normalized = lines[0] if lines else ""
    normalized = _truncate_at_chat_marker(normalized)

    for prefix in CORRECTION_PREFIXES:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):].strip()
            break

    for suffix in CORRECTION_SUFFIXES:
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)].rstrip()
            break

    return normalized.strip().strip("\"'“”")


def parse_choice_prediction(text: str) -> Tuple[Optional[str], str]:
    normalized = unicodedata.normalize("NFKC", normalize_generation(text)).upper()
    for ch in normalized:
        if ch in string.ascii_uppercase:
            return ch, normalized
    return None, normalized


def parse_targeted_choice_prediction(text: str) -> Tuple[Optional[str], str]:
    normalized = unicodedata.normalize("NFKC", normalize_generation(text)).upper()
    normalized = normalized.replace("，", ",").replace(" ", "")
    segments = [seg for seg in normalized.split(",") if seg]
    labels = []
    for seg in segments:
        label = next((ch for ch in seg if ch in string.ascii_uppercase), None)
        if label is not None:
            labels.append(label)
    if not labels:
        return None, normalized
    return ",".join(labels), normalized


def parse_single_char_prediction(text: str) -> Tuple[Optional[str], str]:
    normalized = unicodedata.normalize("NFKC", normalize_generation(text)).strip()
    for ch in normalized:
        if '\u4e00' <= ch <= '\u9fff':
            return ch, normalized
    return None, normalized


def load_model_and_tokenizer(base_model: str, adapter_path: Optional[str], use_4bit: bool):
    _patch_dispatch_model_for_4bit()
    _patch_qwen2_rotary_embedding_device()

    resolved_adapter_path = None
    if adapter_path:
        adapter_dir = Path(adapter_path).expanduser().resolve()
        resolved_adapter_path = str(adapter_dir) if adapter_dir.exists() else adapter_path

    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer_source = base_model
    if resolved_adapter_path:
        adapter_dir = Path(resolved_adapter_path)
        if adapter_dir.exists() and any((adapter_dir / name).exists() for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json")):
            tokenizer_source = resolved_adapter_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    if resolved_adapter_path:
        model = PeftModel.from_pretrained(model, resolved_adapter_path)

    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    model_device = _get_model_device(model)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def evaluate_choice(records: List[dict], model, tokenizer, max_new_tokens: int, print_every: int):
    predictions = []
    errors = []

    correct = 0
    invalid = 0

    for idx, example in enumerate(tqdm(records, desc="Evaluating choice"), start=1):
        prompt = build_choice_prompt(example)
        raw_output = generate_text(model, tokenizer, prompt, max_new_tokens=min(max_new_tokens, 8))
        pred_choice, normalized_output = parse_choice_prediction(raw_output)
        gold_choice = str(example["output"]).strip().upper()
        is_correct = pred_choice == gold_choice

        if pred_choice is None:
            invalid += 1
        if is_correct:
            correct += 1

        item = {
            "instruction": example.get("instruction", ""),
            "input": example.get("input", ""),
            "gold": gold_choice,
            "prediction": pred_choice,
            "raw_output": raw_output,
            "normalized_output": normalized_output,
            "correct": is_correct,
        }
        predictions.append(item)
        if not is_correct:
            errors.append(item)

        if print_every > 0 and idx % print_every == 0:
            print(f"[{idx}] gold={gold_choice} pred={pred_choice} raw={raw_output}")

    total = len(records)
    metrics = {
        "task_type": "choice",
        "num_samples": total,
        "correct": correct,
        "invalid_predictions": invalid,
        "accuracy": correct / total if total > 0 else 0.0,
    }
    return metrics, predictions, errors


def evaluate_targeted_choice(records: List[dict], model, tokenizer, max_new_tokens: int, print_every: int):
    predictions = []
    errors = []
    exact_correct = 0
    invalid = 0
    total_positions = 0
    correct_positions = 0

    for idx, example in enumerate(tqdm(records, desc="Evaluating targeted_choice"), start=1):
        prompt = build_targeted_choice_prompt(example)
        raw_output = generate_text(model, tokenizer, prompt, max_new_tokens=min(max_new_tokens, 16))
        pred_seq, normalized_output = parse_targeted_choice_prediction(raw_output)
        gold_seq = str(example["output"]).replace("，", ",").replace(" ", "").strip().upper()
        is_correct = pred_seq == gold_seq

        gold_labels = [x for x in gold_seq.split(",") if x]
        pred_labels = [x for x in (pred_seq.split(",") if pred_seq else []) if x]
        total_positions += len(gold_labels)
        correct_positions += sum(1 for g, p in zip(gold_labels, pred_labels) if g == p)

        if pred_seq is None:
            invalid += 1
        if is_correct:
            exact_correct += 1

        item = {
            "instruction": example.get("instruction", ""),
            "input": example.get("input", ""),
            "gold": gold_seq,
            "prediction": pred_seq,
            "raw_output": raw_output,
            "normalized_output": normalized_output,
            "correct": is_correct,
            "num_positions": example.get("num_positions"),
            "num_non_keep": example.get("num_non_keep"),
            "all_keep": example.get("all_keep"),
        }
        predictions.append(item)
        if not is_correct:
            errors.append(item)

        if print_every > 0 and idx % print_every == 0:
            print(f"[{idx}] gold={gold_seq} pred={pred_seq} raw={raw_output}")

    total = len(records)
    metrics = {
        "task_type": "targeted_choice",
        "num_samples": total,
        "exact_correct": exact_correct,
        "invalid_predictions": invalid,
        "sequence_accuracy": exact_correct / total if total > 0 else 0.0,
        "position_accuracy": correct_positions / total_positions if total_positions > 0 else 0.0,
    }
    return metrics, predictions, errors


def evaluate_targeted_choice_abstain(records: List[dict], model, tokenizer, max_new_tokens: int, print_every: int):
    predictions = []
    errors = []
    exact_correct = 0
    invalid = 0
    total_positions = 0
    correct_positions = 0
    total_abstain = 0
    correct_abstain = 0
    predicted_abstain = 0

    for idx, example in enumerate(tqdm(records, desc="Evaluating targeted_choice_abstain"), start=1):
        prompt = build_targeted_choice_prompt(example)
        raw_output = generate_text(model, tokenizer, prompt, max_new_tokens=min(max_new_tokens, 16))
        pred_seq, normalized_output = parse_targeted_choice_prediction(raw_output)
        gold_seq = str(example["output"]).replace("，", ",").replace(" ", "").strip().upper()
        is_correct = pred_seq == gold_seq

        gold_labels = [x for x in gold_seq.split(",") if x]
        pred_labels = [x for x in (pred_seq.split(",") if pred_seq else []) if x]
        total_positions += len(gold_labels)
        correct_positions += sum(1 for g, p in zip(gold_labels, pred_labels) if g == p)
        total_abstain += sum(1 for label in gold_labels if label == "N")
        correct_abstain += sum(1 for g, p in zip(gold_labels, pred_labels) if g == "N" and p == "N")
        predicted_abstain += sum(1 for label in pred_labels if label == "N")

        if pred_seq is None:
            invalid += 1
        if is_correct:
            exact_correct += 1

        item = {
            "instruction": example.get("instruction", ""),
            "input": example.get("input", ""),
            "gold": gold_seq,
            "prediction": pred_seq,
            "raw_output": raw_output,
            "normalized_output": normalized_output,
            "correct": is_correct,
            "num_positions": example.get("num_positions"),
            "num_non_keep": example.get("num_non_keep"),
            "num_abstain": example.get("num_abstain"),
            "all_keep": example.get("all_keep"),
        }
        predictions.append(item)
        if not is_correct:
            errors.append(item)

        if print_every > 0 and idx % print_every == 0:
            print(f"[{idx}] gold={gold_seq} pred={pred_seq} raw={raw_output}")

    total = len(records)
    abstain_precision = correct_abstain / predicted_abstain if predicted_abstain > 0 else 0.0
    abstain_recall = correct_abstain / total_abstain if total_abstain > 0 else 0.0
    metrics = {
        "task_type": "targeted_choice_abstain",
        "num_samples": total,
        "exact_correct": exact_correct,
        "invalid_predictions": invalid,
        "sequence_accuracy": exact_correct / total if total > 0 else 0.0,
        "position_accuracy": correct_positions / total_positions if total_positions > 0 else 0.0,
        "abstain_precision": abstain_precision,
        "abstain_recall": abstain_recall,
    }
    return metrics, predictions, errors


def evaluate_targeted_open_repair(records: List[dict], model, tokenizer, max_new_tokens: int, print_every: int):
    predictions = []
    errors = []
    correct = 0
    invalid = 0

    for idx, example in enumerate(tqdm(records, desc="Evaluating targeted_open_repair"), start=1):
        prompt = build_targeted_choice_prompt(example)
        raw_output = generate_text(model, tokenizer, prompt, max_new_tokens=min(max_new_tokens, 4))
        pred_char, normalized_output = parse_single_char_prediction(raw_output)
        gold_char = str(example["output"]).strip()
        is_correct = pred_char == gold_char

        if pred_char is None:
            invalid += 1
        if is_correct:
            correct += 1

        item = {
            "instruction": example.get("instruction", ""),
            "input": example.get("input", ""),
            "gold": gold_char,
            "prediction": pred_char,
            "raw_output": raw_output,
            "normalized_output": normalized_output,
            "correct": is_correct,
            "position_index": example.get("position_index"),
            "source_char": example.get("source_char"),
            "target_char": example.get("target_char"),
        }
        predictions.append(item)
        if not is_correct:
            errors.append(item)

        if print_every > 0 and idx % print_every == 0:
            print(f"[{idx}] gold={gold_char} pred={pred_char} raw={raw_output}")

    total = len(records)
    metrics = {
        "task_type": "targeted_open_repair",
        "num_samples": total,
        "correct": correct,
        "invalid_predictions": invalid,
        "accuracy": correct / total if total > 0 else 0.0,
    }
    return metrics, predictions, errors


def evaluate_targeted_recheck(records: List[dict], model, tokenizer, max_new_tokens: int, print_every: int):
    metrics, predictions, errors = evaluate_choice(
        records=records,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        print_every=print_every,
    )
    metrics["task_type"] = "targeted_recheck"
    return metrics, predictions, errors


def evaluate_correction(records: List[dict], model, tokenizer, max_new_tokens: int, print_every: int):
    predictions = []
    errors = []

    exact_match = 0
    clean_total = 0
    clean_correct = 0
    error_total = 0
    error_correct = 0
    over_correction = 0
    under_correction = 0

    sighan_metrics = SighanCSCMetrics()

    for idx, example in enumerate(tqdm(records, desc="Evaluating correction"), start=1):
        source = str(example["source"]).strip()
        target = str(example["target"]).strip()
        prompt = build_correction_prompt(source)
        raw_output = generate_text(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        prediction = normalize_generation(raw_output)

        if prediction == target:
            exact_match += 1

        if source == target:
            clean_total += 1
            if prediction == source:
                clean_correct += 1
            else:
                over_correction += 1
        else:
            error_total += 1
            if prediction == target:
                error_correct += 1
            if prediction == source:
                under_correction += 1

        sighan_metrics.add_sentence(source, target, prediction)

        item = {
            "source": source,
            "target": target,
            "prediction": prediction,
            "raw_output": raw_output,
            "label": example.get("label"),
            "correct": prediction == target,
        }
        predictions.append(item)
        if prediction != target:
            errors.append(item)

        if print_every > 0 and idx % print_every == 0:
            print(f"[{idx}] src={source}")
            print(f"[{idx}] tgt={target}")
            print(f"[{idx}] pred={prediction}")

    total = len(records)
    sent_pred_positive = sum(1 for item in predictions if item["prediction"] != item["source"])
    sent_target_positive = error_total
    sent_tp = error_correct

    sent_precision = sent_tp / sent_pred_positive if sent_pred_positive > 0 else 0.0
    sent_recall = sent_tp / sent_target_positive if sent_target_positive > 0 else 0.0
    sent_f1 = (
        2 * sent_precision * sent_recall / (sent_precision + sent_recall)
        if (sent_precision + sent_recall) > 0
        else 0.0
    )

    metrics = {
        "task_type": "correction",
        "num_samples": total,
        "exact_match_acc": exact_match / total if total > 0 else 0.0,
        "clean_total": clean_total,
        "clean_keep_acc": clean_correct / clean_total if clean_total > 0 else 0.0,
        "error_total": error_total,
        "error_fix_acc": error_correct / error_total if error_total > 0 else 0.0,
        "over_correction_count": over_correction,
        "under_correction_count": under_correction,
        "sent_correct_precision": sent_precision,
        "sent_correct_recall": sent_recall,
        "sent_correct_f1": sent_f1,
        "abnormal_pairs": len(sighan_metrics.abnormal_pairs),
    }

    (
        char_detect_acc,
        char_detect_p,
        char_detect_r,
        char_detect_f1,
        char_correct_acc,
        char_correct_p,
        char_correct_r,
        char_correct_f1,
        sent_detect_acc,
        sent_detect_p,
        sent_detect_r,
        sent_detect_f1,
        sent_correct_acc,
        sent_correct_p,
        sent_correct_r,
        sent_correct_f1_ref,
    ) = sighan_metrics.get_results()

    metrics.update({
        "sighan_char_detect_acc": char_detect_acc,
        "sighan_char_detect_p": char_detect_p,
        "sighan_char_detect_r": char_detect_r,
        "sighan_char_detect_f1": char_detect_f1,
        "sighan_char_correct_acc": char_correct_acc,
        "sighan_char_correct_p": char_correct_p,
        "sighan_char_correct_r": char_correct_r,
        "sighan_char_correct_f1": char_correct_f1,
        "sighan_sent_detect_acc": sent_detect_acc,
        "sighan_sent_detect_p": sent_detect_p,
        "sighan_sent_detect_r": sent_detect_r,
        "sighan_sent_detect_f1": sent_detect_f1,
        "sighan_sent_correct_acc": sent_correct_acc,
        "sighan_sent_correct_p": sent_correct_p,
        "sighan_sent_correct_r": sent_correct_r,
        "sighan_sent_correct_f1": sent_correct_f1_ref,
    })
    return metrics, predictions, errors


def save_jsonl(path: Path, items: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    records = load_jsonl_records(args.data_path, limit=args.limit)
    if not records:
        raise ValueError("测试数据为空")

    task_type = detect_task_type(records[0], args.task_type)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" MASC-CSC | LLM LoRA Evaluation ")
    print("=" * 60)
    print(f"[*] base_model : {args.base_model}")
    print(f"[*] adapter    : {args.adapter_path}")
    print(f"[*] data_path  : {args.data_path}")
    print(f"[*] task_type  : {task_type}")
    print(f"[*] num_samples: {len(records)}")
    print(f"[*] output_dir : {output_dir}")
    print("=" * 60)

    model, tokenizer = load_model_and_tokenizer(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        use_4bit=args.use_4bit,
    )

    if task_type == "choice":
        metrics, predictions, errors = evaluate_choice(
            records=records,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            print_every=args.print_every,
        )
    elif task_type == "targeted_choice_abstain":
        metrics, predictions, errors = evaluate_targeted_choice_abstain(
            records=records,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            print_every=args.print_every,
        )
    elif task_type == "targeted_choice":
        metrics, predictions, errors = evaluate_targeted_choice(
            records=records,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            print_every=args.print_every,
        )
    elif task_type == "targeted_open_repair":
        metrics, predictions, errors = evaluate_targeted_open_repair(
            records=records,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            print_every=args.print_every,
        )
    elif task_type == "targeted_recheck":
        metrics, predictions, errors = evaluate_targeted_recheck(
            records=records,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            print_every=args.print_every,
        )
    else:
        metrics, predictions, errors = evaluate_correction(
            records=records,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            print_every=args.print_every,
        )

    metrics["base_model"] = args.base_model
    metrics["adapter_path"] = args.adapter_path
    metrics["data_path"] = args.data_path

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    save_jsonl(output_dir / "predictions.jsonl", predictions)
    save_jsonl(output_dir / "errors.jsonl", errors)

    print("\n[Done] Metrics:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[Done] predictions.jsonl -> {output_dir / 'predictions.jsonl'}")
    print(f"[Done] errors.jsonl      -> {output_dir / 'errors.jsonl'}")
    print(f"[Done] metrics.json      -> {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
