import argparse
import os
import random
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

try:
    from transformers import modeling_utils as _transformers_modeling_utils
except Exception:
    _transformers_modeling_utils = None


DEFAULT_CORRECTION_PROMPT = (
    "请纠正下面句子中的中文拼写错误；如果原句没有错误，保持原句不变。\n"
    "只输出纠正后的句子，不要添加解释。"
)


# Monkey-patch BitsAndBytesConfig for Baichuan2 custom code compatibility.
def _bnb_getitem(self, key):
    return getattr(self, key)


def _bnb_contains(self, key):
    return hasattr(self, key)


from transformers.utils.quantization_config import BitsAndBytesConfig

BitsAndBytesConfig.__getitem__ = _bnb_getitem
BitsAndBytesConfig.__contains__ = _bnb_contains
BitsAndBytesConfig.get = lambda self, key, default=None: getattr(self, key, default)


def _patch_dispatch_model_for_4bit():
    if _transformers_modeling_utils is None:
        return

    original_dispatch_model = getattr(_transformers_modeling_utils, "dispatch_model", None)
    if original_dispatch_model is None:
        return

    def safe_dispatch_model(model, device_map=None, *args, **kwargs):
        if isinstance(device_map, dict) and device_map and len(set(device_map.values())) == 1:
            # 4-bit 单卡加载时，直接保留模型当前设备，避免 accelerate 再次调用 .to().
            model.hf_device_map = dict(device_map)
            return model
        return original_dispatch_model(model, device_map=device_map, *args, **kwargs)

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

    def safe_forward(self, x, seq_len=None):
        seq_len = seq_len if seq_len is not None else x.shape[-2]
        target_device = x.device

        if self.inv_freq.device != target_device:
            self.inv_freq = self.inv_freq.to(target_device)

        if (
            seq_len > self.max_seq_len_cached
            or self.cos_cached.device != target_device
            or self.sin_cached.device != target_device
        ):
            self._set_cos_sin_cache(
                seq_len=max(seq_len, self.max_seq_len_cached),
                device=target_device,
                dtype=x.dtype,
            )

        return original_forward(self, x, seq_len=seq_len)

    safe_forward._masc_rotary_device_patch = True
    rotary_cls.forward = safe_forward


def _get_model_device(model):
    for param in model.parameters():
        return param.device
    for buffer in model.buffers():
        return buffer.device
    return torch.device("cpu")


# Hack to enable qwen3_5 support for Qwen3.5-2B model on older transformers versions.
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    CONFIG_MAPPING.register("qwen3_5", Qwen2Config)
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES["qwen3_5"] = "Qwen2ForCausalLM"
    MODEL_FOR_CAUSAL_LM_MAPPING.register("qwen3_5", Qwen2ForCausalLM)
except Exception:
    pass

def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))

def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))

def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))

def is_main_process() -> bool:
    return get_rank() == 0

def rank0_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs, flush=True)

def parse_args():
    parser = argparse.ArgumentParser(description="MASC-CSC 验证器 LLM LoRA 微调脚本")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="预训练 LLM 的本地路径或 Hub ID")
    parser.add_argument(
        "--data_path",
        type=str,
        default="datasets/cscd-ns/train.jsonl",
        help="训练数据路径。支持单个 jsonl、目录、或逗号分隔的多个路径",
    )
    parser.add_argument("--output_dir", type=str, default="./ckpt/llm_lora", help="微调后 LoRA 适配器的保存目录")
    parser.add_argument("--use_4bit", action="store_true", help="是否使用 4-bit 量化加载基座模型")
    parser.add_argument("--preflight_only", action="store_true", help="只做训练前检查和模型初始化，不真正开始训练")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="每张显卡的 batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument(
        "--max_error_to_clean_ratio",
        type=float,
        default=1.0,
        help="只对 correction 样本生效；限制 error:clean 的最大比例，传负数关闭下采样",
    )
    parser.add_argument("--max_seq_length", type=int, default=384, help="输入序列的最大长度")
    parser.add_argument("--logging_steps", type=int, default=5, help="多少步记录一次日志")
    parser.add_argument("--save_steps", type=int, default=50, help="多少步保存一次检查点")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA 的秩 (Rank)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA 的缩放系数")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA Dropout 比例")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", help="支持 BF16 的卡优先开启")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--ddp_find_unused_parameters", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--deepspeed", type=str, default=None)
    return parser.parse_args()


def validate_model_dir(model_path: str) -> None:
    model_dir = Path(model_path)
    if not model_dir.exists() or not model_dir.is_dir():
        raise FileNotFoundError(f"找不到模型目录：{model_path}")

    required_files = ["config.json", "tokenizer_config.json"]
    missing = [name for name in required_files if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"模型目录缺少必要文件：{', '.join(missing)}")

    has_tokenizer = any((model_dir / name).exists() for name in ["tokenizer.json", "vocab.json"])
    if not has_tokenizer:
        raise FileNotFoundError("模型目录缺少 tokenizer.json 或 vocab.json")

    has_weights = any(model_dir.glob("model.safetensors*"))
    if not has_weights:
        raise FileNotFoundError("模型目录缺少 safetensors 权重文件")

    import json

    config_data = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    architectures = config_data.get("architectures", [])
    if any(name == "Qwen3_5ForConditionalGeneration" for name in architectures):
        raise ValueError(
            "当前模型是多模态 Qwen3_5ForConditionalGeneration 检查点，不是纯文本 CausalLM；"
            "请改用纯文本模型目录，例如 ./LLM/Qwen2.5-0.5B-Instruct。"
        )

    if config_data.get("model_type") == "qwen3_5" and "auto_map" not in config_data:
        raise ValueError(
            "当前模型目录虽然是 qwen3_5，但缺少 auto_map / 自定义模型代码；"
            "本地 transformers 4.40 无法直接识别 qwen3_5。"
            "建议直接换成当前环境原生支持的纯文本 Qwen2.5-0.5B-Instruct。"
        )

    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        index_data = json.loads(index_file.read_text(encoding="utf-8"))
        shard_names = sorted(set(index_data.get("weight_map", {}).values()))
        missing_shards = [name for name in shard_names if not (model_dir / name).exists()]
        if missing_shards:
            raise FileNotFoundError(f"模型索引引用了不存在的分片文件：{', '.join(missing_shards)}")


def _split_input_paths(data_path: str):
    parts = [item.strip() for item in data_path.split(",") if item.strip()]
    if not parts:
        raise ValueError("data_path 为空")
    return parts


def resolve_data_files(data_path: str):
    resolved = []
    for raw_path in _split_input_paths(data_path):
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"找不到数据路径：{raw_path}")

        if path.is_file():
            if path.suffix != ".jsonl":
                raise ValueError(f"当前仅支持 .jsonl 文件，收到：{path}")
            resolved.append(str(path))
            continue

        if path.is_dir():
            all_jsonl = sorted(str(p) for p in path.rglob("*.jsonl"))
            if not all_jsonl:
                raise FileNotFoundError(f"目录下没有找到任何 .jsonl 文件：{path}")
            train_like = [p for p in all_jsonl if "train" in Path(p).stem.lower()]
            resolved.extend(train_like or all_jsonl)
            continue

        raise ValueError(f"不支持的数据路径：{raw_path}")

    deduped = []
    seen = set()
    for item in resolved:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def get_response_template(tokenizer) -> str:
    if getattr(tokenizer, "chat_template", None):
        return "<|im_start|>assistant\n"
    return "Assistant:"


def format_chat_example(tokenizer, user_content: str, assistant_content: str, system_content: str = "") -> str:
    user_content = str(user_content).strip()
    assistant_content = str(assistant_content).strip()
    system_content = str(system_content).strip() if system_content is not None else ""

    if getattr(tokenizer, "chat_template", None):
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": assistant_content})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    prompt_parts = [item for item in (system_content, user_content) if item]
    prompt = "\n".join(prompt_parts)
    eos = tokenizer.eos_token or ""
    return f"User:\n{prompt}\n\nAssistant:{assistant_content}{eos}"


def build_text(example, tokenizer):
    text = example.get("text")
    if text is not None:
        text = str(text).strip()
        if text:
            return {"text": text, "_sample_label": -1}

    instruction = example.get("instruction")
    input_text = example.get("input")
    output = example.get("output")
    if instruction is not None and output is not None:
        prompt = str(instruction).strip()
        if input_text is not None and str(input_text).strip():
            prompt = f"{prompt}\n\n{str(input_text).strip()}"
        text = format_chat_example(tokenizer, user_content=prompt, assistant_content=str(output).strip())
        return {"text": text, "_sample_label": -1}

    source = example.get("source")
    target = example.get("target")
    if source is not None and target is not None:
        source_text = str(source).strip()
        target_text = str(target).strip()
        text = format_chat_example(
            tokenizer,
            system_content=DEFAULT_CORRECTION_PROMPT,
            user_content=f"原句：{source_text}",
            assistant_content=target_text,
        )
        return {"text": text, "_sample_label": int(source_text != target_text)}

    raise ValueError(
        "训练样本格式无法识别。支持三种 schema："
        "{instruction, input, output}、{instruction, output}、{source, target[, label]}、或 {text}。"
    )


def rebalance_correction_dataset(dataset, max_error_to_clean_ratio: float, seed: int):
    if max_error_to_clean_ratio is None or max_error_to_clean_ratio < 0:
        return dataset

    if "_sample_label" not in dataset.column_names:
        return dataset

    labels = dataset["_sample_label"]
    clean_indices = [idx for idx, label in enumerate(labels) if label == 0]
    error_indices = [idx for idx, label in enumerate(labels) if label == 1]
    other_indices = [idx for idx, label in enumerate(labels) if label not in (0, 1)]

    print(
        f"[*] correction 样本统计: clean={len(clean_indices)}, error={len(error_indices)}, other={len(other_indices)}"
    )

    if not clean_indices or not error_indices:
        return dataset

    max_error_count = int(len(clean_indices) * max_error_to_clean_ratio)
    if max_error_count <= 0:
        raise ValueError("max_error_to_clean_ratio 必须大于 0，或传负数关闭下采样")

    if len(error_indices) <= max_error_count:
        return dataset

    rng = random.Random(seed)
    kept_error_indices = rng.sample(error_indices, max_error_count)
    keep_indices = sorted(clean_indices + kept_error_indices + other_indices)
    rebalanced = dataset.select(keep_indices)
    print(
        f"[*] 已下采样 correction 错句到 error:clean<={max_error_to_clean_ratio:.2f}，"
        f"保留 clean={len(clean_indices)}, error={len(kept_error_indices)}, other={len(other_indices)}, total={len(rebalanced)}"
    )
    return rebalanced


def load_and_validate_dataset(data_path: str, tokenizer, response_template: str, max_error_to_clean_ratio: float, seed: int):
    data_files = resolve_data_files(data_path)
    dataset = load_dataset("json", data_files={"train": data_files})["train"]
    if len(dataset) == 0:
        raise ValueError("训练数据为空，无法启动训练")

    supported = any(
        {"instruction", "output"}.issubset(set(dataset.column_names))
        or {"source", "target"}.issubset(set(dataset.column_names))
        or "text" in set(dataset.column_names)
        for _ in [0]
    )
    if not supported:
        raise ValueError(
            "训练数据缺少可识别字段。当前支持：instruction/output、source/target、text。"
        )

    dataset = dataset.map(build_text, fn_kwargs={"tokenizer": tokenizer}, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda x: x["text"] is not None and str(x["text"]).strip() != "")

    if len(dataset) == 0:
        raise ValueError("格式转换后训练数据为空，无法启动训练")

    dataset = rebalance_correction_dataset(
        dataset,
        max_error_to_clean_ratio=max_error_to_clean_ratio,
        seed=seed,
    )
    dataset = dataset.shuffle(seed=seed)

    sample_text = dataset[0]["text"]
    if response_template not in sample_text:
        raise ValueError("训练样本中缺少 assistant 响应模板，无法计算 completion loss")

    if "_sample_label" in dataset.column_names:
        dataset = dataset.remove_columns(["_sample_label"])

    return dataset, data_files


def main():
    args = parse_args()
    set_seed(args.seed)
    local_rank = get_local_rank()
    world_size = get_world_size()  #可能有问题

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    rank0_print("=" * 60)
    rank0_print(" MASC-CSC | LLM Verifier LoRA Fine-Tuning ")
    rank0_print("=" * 60)
    rank0_print(f"[*] WORLD_SIZE: {world_size}")
    rank0_print(f"[*] LOCAL_RANK: {local_rank}")
    rank0_print(f"[*] 基座模型: {args.model_name_or_path}")
    rank0_print(f"[*] 数据路径: {args.data_path}")
    rank0_print(f"[*] 输出目录: {args.output_dir}")
    if args.preflight_only:
        rank0_print("[*] 模式:     仅预检，不执行 trainer.train()")

    validate_model_dir(args.model_name_or_path)

    print("[*] 正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    response_template = get_response_template(tokenizer)

    dataset, data_files = load_and_validate_dataset(
        args.data_path,
        tokenizer=tokenizer,
        response_template=response_template,
        max_error_to_clean_ratio=args.max_error_to_clean_ratio,
        seed=args.seed,
    )
    print(f"[*] 成功加载训练集，共 {len(dataset)} 条样本，来自 {len(data_files)} 个 jsonl 文件。")
    preview = ", ".join(data_files[:3])
    if len(data_files) > 3:
        preview += ", ..."
    print(f"[*] 数据文件: {preview}\n")
    print(f"[*] assistant 模板: {response_template!r}")

    bnb_config = None
    if args.use_4bit:
        print("[*] 正在启用 4-bit 量化 (BitsAndBytes)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    print("[*] 正在加载基座模型 (Base Model)... 可能需要数分钟...")
    _patch_dispatch_model_for_4bit()
    _patch_qwen2_rotary_embedding_device()
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float16

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }

    if args.use_4bit:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = {"": local_rank}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    )

    model.config.use_cache = False

    print("\n[*] 配置 LoRA...")
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    #检查CPU buffer是否搬到gpu
    target_device = torch.device(f"cuda:{local_rank}")

    # 先把仍留在 CPU 的 buffer 搬到当前 rank 对应的 GPU
    moved_cpu_buffers = []
    for module_name, module in model.named_modules():
        for buffer_name, buf in list(module.named_buffers(recurse=False)):
            if buf is None or buf.device.type != "cpu":
                continue
            full_name = f"{module_name}.{buffer_name}" if module_name else buffer_name
            module._buffers[buffer_name] = buf.to(device=target_device)
            moved_cpu_buffers.append(full_name)

    print(f"[rank={os.environ.get('RANK', '?')}] moved_cpu_buffers_count={len(moved_cpu_buffers)}", flush=True)
    print(f"[rank={os.environ.get('RANK', '?')}] moved_cpu_buffers_sample={moved_cpu_buffers[:20]}", flush=True)

    # 再次检查，确认 DDP 前没有 CPU 参数或 buffer
    cpu_params = [name for name, p in model.named_parameters() if p.device.type != "cuda"]
    cpu_buffers = [name for name, b in model.named_buffers() if b.device.type != "cuda"]

    print(f"[rank={os.environ.get('RANK', '?')}] cpu_params_count={len(cpu_params)}", flush=True)
    print(f"[rank={os.environ.get('RANK', '?')}] cpu_buffers_count={len(cpu_buffers)}", flush=True)
    print(f"[rank={os.environ.get('RANK', '?')}] cpu_params={cpu_params[:20]}", flush=True)
    print(f"[rank={os.environ.get('RANK', '?')}] cpu_buffers={cpu_buffers[:50]}", flush=True)

    if cpu_params or cpu_buffers:
        raise RuntimeError("Model still has CPU params/buffers before DDP.")
    #检查结束
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=not args.bf16,
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        report_to=args.report_to,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=True,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        group_by_length=True,
        logging_first_step=True,
        deepspeed=args.deepspeed,
    )


    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
        packing=False,
    )

    if args.preflight_only:
        print("\n[*] 正在执行首个 batch 预检...")
        first_batch = next(iter(trainer.get_train_dataloader()))
        batch_summary = {key: tuple(value.shape) for key, value in first_batch.items() if hasattr(value, "shape")}
        print(f"[*] 首个 batch 预检通过：{batch_summary}")

        print("[*] 正在执行单步前向预检...")
        model_device = _get_model_device(model)
        forward_batch = {key: value.to(model_device) if hasattr(value, "to") else value for key, value in first_batch.items()}
        with torch.no_grad():
            loss = model(**forward_batch).loss
        print(f"[*] 单步前向预检通过：loss={loss.detach().float().cpu().item():.6f}")
        print("[*] 预检通过：数据、Tokenizer、模型、LoRA、Trainer、batch collator 和单步前向均成功。")
        return

    print("\n[*] ================= 开始微调训练 =================")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    print(f"\n[*] 训练结束，正在保存 LoRA 权重至：{args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[*] 全部任务完成！")


if __name__ == "__main__":
    main()
