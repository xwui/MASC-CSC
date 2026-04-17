#!/bin/bash
# ===============================================================================
# train_router.sh
# 句子级 Router 调用门控训练脚本
# 默认前端：本地 HuggingFace NamBERT 目录
# 支持多卡并行特征提取
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

FRONTEND_DIR="${FRONTEND_DIR:-./ckpt/chinese-macbert-base}"
DATA_PATH="${DATA_PATH:-./datasets/Lemon,./datasets/Sighan/sighan15_test.jsonl,./datasets/Sighan/sighan14_test.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-./ckpt/selective_router_nambert.pt}"
DEVICE="${DEVICE:-auto}"
FRONTEND_DEVICE="${FRONTEND_DEVICE:-}"
LLM_DEVICE="${LLM_DEVICE:-}"
ROUTER_DEVICE="${ROUTER_DEVICE:-}"
FEATURE_DEVICES="${FEATURE_DEVICES:-}"
LLM_DEVICES="${LLM_DEVICES:-}"

LABEL_MODE="${LABEL_MODE:-llm-gain}"   # llm-gain / frontend-error
LLM_PATH="${LLM_PATH:-./LLM/Qwen2.5-7B-Instruct}"
LLM_ADAPTER="${LLM_ADAPTER:-}"
LLM_MODE="${LLM_MODE:-targeted}"

TOP_K="${TOP_K:-5}"
FEATURE_TOP_R="${FEATURE_TOP_R:-3}"
LIMIT="${LIMIT:--1}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-1e-3}"
BATCH_SIZE="${BATCH_SIZE:-64}"
VAL_RATIO="${VAL_RATIO:-0.2}"
SEED="${SEED:-42}"
POS_WEIGHT="${POS_WEIGHT:-0}"
MIN_GAIN="${MIN_GAIN:-1e-4}"

NV_CUSPARSE="/home/bkai/anaconda3/envs/cwq_masc_csc/lib/python3.10/site-packages/nvidia/cusparse/lib"
NV_CUDART="/home/bkai/anaconda3/envs/cwq_masc_csc/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
export LD_LIBRARY_PATH="/home/bkai/anaconda3/envs/cwq_masc_csc/lib/:${NV_CUSPARSE}:${NV_CUDART}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

validate_runtime() {
    FRONTEND_DIR="$FRONTEND_DIR" DATA_PATH="$DATA_PATH" LABEL_MODE="$LABEL_MODE" LLM_PATH="$LLM_PATH" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

frontend_dir = Path(os.environ["FRONTEND_DIR"])
data_path = os.environ["DATA_PATH"]
label_mode = os.environ["LABEL_MODE"]
llm_path = os.environ["LLM_PATH"]

if not frontend_dir.exists() or not frontend_dir.is_dir():
    raise FileNotFoundError(f"找不到 HuggingFace 前端目录: {frontend_dir}")

required_frontend_files = ["config.json", "tokenizer_config.json", "csc_model.py", "csc_tokenizer.py"]
for name in required_frontend_files:
    if not (frontend_dir / name).exists():
        raise FileNotFoundError(f"前端目录缺少必要文件: {name}")

parts = [item.strip() for item in data_path.split(",") if item.strip()]
if not parts:
    raise ValueError("DATA_PATH 为空")

found_files = []
for item in parts:
    path = Path(item)
    if not path.exists():
        raise FileNotFoundError(f"找不到数据路径: {item}")
    if path.is_file():
        if path.suffix not in {".csv", ".jsonl"}:
            raise ValueError(f"仅支持 .csv / .jsonl 文件，收到: {path}")
        found_files.append(path)
    elif path.is_dir():
        found_files.extend(sorted(path.rglob("*.csv")))
        found_files.extend(sorted(path.rglob("*.jsonl")))
    else:
        raise ValueError(f"不支持的数据路径: {item}")

if not found_files:
    raise FileNotFoundError("没有找到任何可用于 router 训练的 .csv / .jsonl 数据文件")

if label_mode == "llm-gain":
    if not llm_path:
        raise ValueError("LABEL_MODE=llm-gain 时必须提供 LLM_PATH")
    if not Path(llm_path).exists():
        raise FileNotFoundError(f"找不到 LLM 模型路径: {llm_path}")
PY
}

echo "╔══════════════════════════════════════════════════════╗"
echo "║         MASC-CSC | Router Invoke Head 训练            ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "👉 Python: ${PYTHON_BIN}"

validate_runtime

echo "✅ 依赖与路径预检通过。"
echo "👉 前端目录: ${FRONTEND_DIR}"
echo "👉 数据路径: ${DATA_PATH}"
echo "👉 标签模式: ${LABEL_MODE}"
echo "👉 输出路径: ${OUTPUT_PATH}"
[ -n "${FEATURE_DEVICES}" ] && echo "👉 Feature devices: ${FEATURE_DEVICES}"
[ -n "${LLM_DEVICES}" ] && echo "👉 LLM devices: ${LLM_DEVICES}"

COMMON_ARGS=(
    --ckpt-path "${FRONTEND_DIR}"
    --data "${DATA_PATH}"
    --output "${OUTPUT_PATH}"
    --device "${DEVICE}"
    --label-mode "${LABEL_MODE}"
    --top-k "${TOP_K}"
    --feature-top-r "${FEATURE_TOP_R}"
    --limit "${LIMIT}"
    --epochs "${EPOCHS}"
    --lr "${LR}"
    --batch-size "${BATCH_SIZE}"
    --val-ratio "${VAL_RATIO}"
    --seed "${SEED}"
    --pos-weight "${POS_WEIGHT}"
    --min-gain "${MIN_GAIN}"
)

[ -n "${FRONTEND_DEVICE}" ] && COMMON_ARGS+=(--frontend-device "${FRONTEND_DEVICE}")
[ -n "${LLM_DEVICE}" ] && COMMON_ARGS+=(--llm-device "${LLM_DEVICE}")
[ -n "${ROUTER_DEVICE}" ] && COMMON_ARGS+=(--router-device "${ROUTER_DEVICE}")
[ -n "${FEATURE_DEVICES}" ] && COMMON_ARGS+=(--feature-devices "${FEATURE_DEVICES}")
[ -n "${LLM_DEVICES}" ] && COMMON_ARGS+=(--llm-devices "${LLM_DEVICES}")

if [ "${LABEL_MODE}" = "llm-gain" ]; then
    COMMON_ARGS+=(
        --llm-path "${LLM_PATH}"
        --llm-mode "${LLM_MODE}"
    )
    if [ -n "${LLM_ADAPTER}" ]; then
        COMMON_ARGS+=(--llm-adapter "${LLM_ADAPTER}")
    fi
fi

"${PYTHON_BIN}" scripts/train_router.py "${COMMON_ARGS[@]}"

echo ""
echo "训练结束。"
echo "Router checkpoint: ${OUTPUT_PATH}"
echo "推理时可这样加载:"
echo "  --router selective --router-ckpt ${OUTPUT_PATH}"
