#!/bin/bash
# ===============================================================================
# train_llm_lora.sh
# 验证器大模型 (LLM Verifier) LoRA 参数高效微调脚本
# 推荐基座：Qwen2.5-0.5B-Instruct
# 默认训练数据：datasets/cscd-ns/train.jsonl
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

MIN_FREE_GB="${MIN_FREE_GB:-4}"
MODEL_PATH="${MODEL_PATH:-./LLM/Qwen2.5-0.5B-Instruct}"
DATA_PATH="${DATA_PATH:-./datasets/cscd-ns/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./ckpt/llm_lora_qwen25_05b}"
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

select_gpu() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "❌ 缺少 nvidia-smi，无法安全确认空闲 GPU；为避免影响其他任务，本次不启动训练。" >&2
        exit 1
    fi

    MIN_FREE_GB="$MIN_FREE_GB" "$PYTHON_BIN" - <<'PY'
import os
import subprocess
import sys

min_free_gb = float(os.environ["MIN_FREE_GB"])
min_free_mb = min_free_gb * 1024


def run_query(args):
    proc = subprocess.run(args, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.strip().splitlines() if line.strip()]


gpus = []
for line in run_query(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]):
    parts = [p.strip() for p in line.split(',')]
    if len(parts) != 2:
        continue
    gpus.append((parts[0], float(parts[1])))

if not gpus:
    sys.exit(1)

uuid_to_idx = {}
for row in run_query(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"]):
    parts = [p.strip() for p in row.split(',')]
    if len(parts) == 2:
        uuid_to_idx[parts[1]] = parts[0]

busy = set()
for row in run_query(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"]):
    parts = [p.strip() for p in row.split(',')]
    if len(parts) >= 2 and parts[0] in uuid_to_idx:
        busy.add(uuid_to_idx[parts[0]])

candidates = [(idx, free_mb) for idx, free_mb in gpus if idx not in busy and free_mb >= min_free_mb]
if not candidates:
    sys.exit(1)

candidates.sort(key=lambda item: item[1], reverse=True)
print(candidates[0][0])
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
echo "👉 仅会选择没有现有计算进程、且空闲显存 >= ${MIN_FREE_GB}GB 的 GPU。"

SELECTED_GPU="$(select_gpu || true)"
if [ -z "${SELECTED_GPU}" ]; then
    echo "❌ 没有找到满足条件的空闲 GPU。为了避免影响其他任务，本次不启动训练。"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${SELECTED_GPU}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "👉 已选择 GPU: ${SELECTED_GPU}"
echo "👉 执行一次模型与 Trainer 初始化预检..."

nice -n 10 "$PYTHON_BIN" scripts/train_llm_lora.py \
    --model_name_or_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --use_4bit \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --lora_r 16 \
    --lora_alpha 32 \
    --logging_steps 5 \
    --max_seq_length 384 \
    --preflight_only

echo "👉 预检通过，开始正式训练。"

nice -n 10 "$PYTHON_BIN" scripts/train_llm_lora.py \
    --model_name_or_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --use_4bit \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --lora_r 16 \
    --lora_alpha 32 \
    --logging_steps 5 \
    --max_seq_length 384

echo ""
echo "🎉 LoRA 训练结束！最终微调权重已存入: ${OUTPUT_DIR}"
echo "   将来在 run_masc_csc.py 脚本里，你可以直接把 adapter_path 参数指向这里加载！"
