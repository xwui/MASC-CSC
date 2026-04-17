#!/bin/bash
# ===============================================================================
# train_llm_lora.sh
# 验证器大模型 (LLM Verifier) LoRA 参数高效微调脚本
# 推荐基座：Qwen2.5-7B-Instruct
# 推荐训练数据：先用 scripts/generate_llm_training_data.py 生成 targeted closed-choice JSONL
# ===============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DEFAULT_PYTHON="/home/bkai/anaconda3/envs/cwq_masc_csc/bin/python"
if [ -x "${DEFAULT_PYTHON}" ]; then
    PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
else
    PYTHON_BIN="${PYTHON_BIN:-python}"
fi
TORCHRUN_PYTHON="${TORCHRUN_PYTHON:-${PYTHON_BIN}}"

MIN_FREE_GB="${MIN_FREE_GB:-18}"
MODEL_PATH="${MODEL_PATH:-./LLM/Qwen2.5-7B-Instruct}"
DATA_PATH="${DATA_PATH:-./data/llm_targeted_choice_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./ckpt/llm_lora_qwen25_7b_targeted_choice}"
NV_CUSPARSE="/home/bkai/anaconda3/envs/cwq_masc_csc/lib/python3.10/site-packages/nvidia/cusparse/lib"
NV_CUDART="/home/bkai/anaconda3/envs/cwq_masc_csc/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
export LD_LIBRARY_PATH="/home/bkai/anaconda3/envs/cwq_masc_csc/lib/:${NV_CUSPARSE}:${NV_CUDART}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/bkai/cwq/packages:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

validate_runtime() {
    MODEL_PATH="$MODEL_PATH" DATA_PATH="$DATA_PATH" "$PYTHON_BIN" - <<'PY'
import importlib
import os
from pathlib import Path

model_path = Path(os.environ["MODEL_PATH"])
data_path = os.environ["DATA_PATH"]

required_modules = ["torch", "transformers", "datasets", "peft", "trl", "bitsandbytes", "accelerate"]
for name in required_modules:
    importlib.import_module(name)

if not model_path.exists() or not model_path.is_dir():
    raise FileNotFoundError(f"找不到模型目录：{model_path}")

for required in ["config.json", "tokenizer_config.json"]:
    if not (model_path / required).exists():
        raise FileNotFoundError(f"模型目录缺少必要文件：{required}")

if not any((model_path / name).exists() for name in ["tokenizer.json", "vocab.json"]):
    raise FileNotFoundError("模型目录缺少 tokenizer.json 或 vocab.json")

if not any(model_path.glob("model.safetensors*")):
    raise FileNotFoundError("模型目录缺少 safetensors 权重文件")

import json

config_data = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
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

index_file = model_path / "model.safetensors.index.json"
if index_file.exists():
    index_data = json.loads(index_file.read_text(encoding="utf-8"))
    shard_names = sorted(set(index_data.get("weight_map", {}).values()))
    missing_shards = [name for name in shard_names if not (model_path / name).exists()]
    if missing_shards:
        raise FileNotFoundError(f"模型索引引用了不存在的分片文件：{', '.join(missing_shards)}")


def split_input_paths(raw):
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if not parts:
        raise ValueError("DATA_PATH 为空")
    return parts


def resolve_data_files(raw):
    resolved = []
    for item in split_input_paths(raw):
        path = Path(item)
        if not path.exists():
            raise FileNotFoundError(f"找不到训练数据路径：{item}")
        if path.is_file():
            if path.suffix != ".jsonl":
                raise ValueError(f"当前仅支持 .jsonl 文件，收到：{path}")
            resolved.append(path)
            continue
        if path.is_dir():
            all_jsonl = sorted(path.rglob("*.jsonl"))
            if not all_jsonl:
                raise FileNotFoundError(f"目录下没有找到任何 .jsonl 文件：{path}")
            train_like = [p for p in all_jsonl if "train" in p.stem.lower()]
            resolved.extend(train_like or all_jsonl)
            continue
        raise ValueError(f"不支持的数据路径：{item}")
    deduped = []
    seen = set()
    for item in resolved:
        item = item.resolve()
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


data_files = resolve_data_files(data_path)
nonempty = 0
for file_path in data_files:
    lines = [line for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if lines:
        nonempty += 1

if nonempty == 0:
    raise ValueError("训练数据为空，无法启动训练")
PY
}



echo "╔══════════════════════════════════════════════════════╗"
echo "║          MASC-CSC | LLM Verifier LoRA 微调            ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "👉 Python: ${PYTHON_BIN}"

validate_runtime

echo "✅ 依赖、模型文件和训练数据预检通过。"
echo "👉 模型位置: ${MODEL_PATH}"
echo "👉 数据路径: ${DATA_PATH}"
REQUESTED_NUM_GPUS="${NUM_GPUS:-4}"
AUTO_SELECT_GPUS="${AUTO_SELECT_GPUS:-1}"
if [ -n "${MASTER_PORT:-}" ]; then
    MASTER_PORT="${MASTER_PORT}"
else
    MASTER_PORT="$(${PYTHON_BIN} - <<'PY' 
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("", 0))
    print(s.getsockname()[1])
PY
)"
fi
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    SELECTED_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
elif [ "${AUTO_SELECT_GPUS}" = "1" ]; then
    SELECTED_CUDA_VISIBLE_DEVICES="$(${PYTHON_BIN} - <<'PY' 
import os
import subprocess

requested = int(os.environ.get("REQUESTED_NUM_GPUS", "4"))
min_free_gb = float(os.environ.get("MIN_FREE_GB", "4"))
min_free_mb = int(min_free_gb * 1024)
cmd = [
    "nvidia-smi",
    "--query-gpu=index,memory.total,memory.used,memory.free",
    "--format=csv,noheader,nounits",
]
try:
    output = subprocess.check_output(cmd, text=True)
except Exception as exc:
    raise SystemExit(f"无法执行 nvidia-smi 自动选卡：{exc}")
rows = []
for line in output.splitlines():
    if not line.strip():
        continue
    idx, total, used, free = [part.strip() for part in line.split(",")]
    rows.append({"idx": int(idx), "total": int(total), "used": int(used), "free": int(free)})
eligible = [row for row in rows if row["free"] >= min_free_mb]
eligible.sort(key=lambda row: (-row["free"], row["idx"]))
if len(eligible) < requested:
    details = "; ".join(f"gpu{row['idx']}:free={row['free']}MB,used={row['used']}MB" for row in rows)
    raise SystemExit(f"可用 GPU 不足：需要 {requested} 张且每张至少 {min_free_mb}MB 空闲；当前 {details}")
selected = eligible[:requested]
print(",".join(str(row["idx"]) for row in selected))
PY
)"
else
    SELECTED_CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
fi
CUDA_VISIBLE_DEVICES="${SELECTED_CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${REQUESTED_NUM_GPUS}"
echo "👉 分布式端口: ${MASTER_PORT}"
echo "👉 使用 GPU: ${CUDA_VISIBLE_DEVICES}"
RUN_MODE="${RUN_MODE:-train}"   # train / preflight

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-32}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"  # targeted closed-choice verifier 通常更适合更小学习率
MAX_ERROR_TO_CLEAN_RATIO="${MAX_ERROR_TO_CLEAN_RATIO:-1.0}"
NUM_EPOCHS="${NUM_EPOCHS:-8}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-384}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-100}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
LORA_R="${LORA_R:-16}" 
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
BF16="${BF16:-1}"

export CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export NCCL_ASYNC_ERROR_HANDLING=1

COMMON_ARGS=(
    --model_name_or_path "${MODEL_PATH}"
    --data_path "${DATA_PATH}"
    --output_dir "${OUTPUT_DIR}"
    --use_4bit
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRAD_ACC_STEPS}"
    --learning_rate "${LEARNING_RATE}"
    --max_error_to_clean_ratio "${MAX_ERROR_TO_CLEAN_RATIO}"
    --num_train_epochs "${NUM_EPOCHS}"
    --lora_r "${LORA_R}"
    --lora_alpha "${LORA_ALPHA}"
    --lora_dropout "${LORA_DROPOUT}"
    --logging_steps "${LOGGING_STEPS}"
    --save_steps "${SAVE_STEPS}"
    --max_seq_length "${MAX_SEQ_LENGTH}"
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
    --warmup_ratio "${WARMUP_RATIO}"
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
)

if [ "${BF16}" = "1" ]; then
    COMMON_ARGS+=(--bf16)
fi

if [ "${RUN_MODE}" = "preflight" ]; then
    COMMON_ARGS+=(--preflight_only)
fi

if [ "${NUM_GPUS}" = "1" ]; then
    "${PYTHON_BIN}" scripts/train_llm_lora.py \
        "${COMMON_ARGS[@]}"
else
    "${TORCHRUN_PYTHON}" -m torch.distributed.run \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port="${MASTER_PORT}" \
        scripts/train_llm_lora.py \
        "${COMMON_ARGS[@]}"
fi


echo ""
echo "🎉 LoRA 训练结束！最终微调权重已存入: ${OUTPUT_DIR}"
echo "   将来在 run_masc_csc.py 脚本里，你可以直接把 adapter_path 参数指向这里加载！"
