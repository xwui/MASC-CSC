#!/bin/bash
# ==============================================================================
# train_stage1_pretrain.sh
# 阶段一：在 Wang271K + SIGHAN 合并训练集上对多模态前端进行大规模预训练
# 目标：让拼音（Pinyin）和字形（Glyph）双通道充分收敛，奠定纠错基础
#
# 使用方法（在项目根目录执行）：
#   bash scripts/train_stage1_pretrain.sh
#
# 前提条件：
#   1. 已运行 python scripts/prepare_datasets.py 生成了 CSV 数据文件
#   2. datasets/wang271k_sighan_train.csv 已存在
# ==============================================================================

set -e  # 任意步骤报错则立刻退出

WORK_DIR="./outputs/stage1"
CKPT_DIR="./ckpt/stage1"

echo "========================================================"
echo " MASC-CSC 阶段一训练：Wang271K + SIGHAN 大规模预训练"
echo " 输出目录：${WORK_DIR}"
echo " 检查点目录：${CKPT_DIR}"
echo "========================================================"

# 检查数据文件是否存在
if [ ! -f "./datasets/wang271k_sighan_train.csv" ]; then
    echo "[ERROR] 找不到 datasets/wang271k_sighan_train.csv"
    echo "  请先运行：python scripts/prepare_datasets.py"
    exit 1
fi

HF_ENDPOINT=https://hf-mirror.com python train.py \
    --model multimodal_frontend \
    --datas wang271k_sighan_train.csv \
    --batch-size 64 \
    --epochs 10 \
    --min-epochs 5 \
    --valid-ratio 0.02 \
    --no-resume \
    --early-stop 3 \
    --accumulate_grad_batches 1 \
    --precision "16-mixed" \
    --workers 0 \
    --work-dir "${WORK_DIR}" \
    --ckpt-dir "${CKPT_DIR}"

echo ""
echo "========================================================"
echo " 阶段一训练完成！最优检查点已保存至：${CKPT_DIR}/best.ckpt"
echo " 请继续运行阶段二精调脚本："
echo "   bash scripts/train_stage2_finetune.sh"
echo "========================================================"
