#!/bin/bash
# ==============================================================================
# train_all.sh
# 一键执行 MASC-CSC 前端模型完整训练流程（两阶段课程学习）
#
# 流程：
#   Step 0: 数据准备（调用 prepare_datasets.py）
#   Step 1: 阶段一大规模预训练（Wang271K + SIGHAN）
#   Step 2: 阶段二跨域精调（CSCD-NS + CSCD-IME）
#
# 使用方法（在项目根目录执行）：
#   bash train_all.sh
#
# 说明：
#   - 如果已手动完成数据准备，可跳过 Step 0，直接运行各阶段脚本。
#   - 如果已有阶段一的权重，可设置环境变量跳过：
#       SKIP_STAGE1=1 bash train_all.sh
# ==============================================================================

set -e

SKIP_STAGE1="${SKIP_STAGE1:-0}"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║       MASC-CSC 前端模型完整训练流程（两阶段）         ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ─────────────────────────────────
# Step 0: 数据准备
# ─────────────────────────────────
echo "─────────────────────────────────────"
echo " Step 0: 数据准备与格式转换"
echo "─────────────────────────────────────"

# 检查是否已有训练数据，避免重复转换
if [ -f "./datasets/wang271k_sighan_train.csv" ] && \
   [ -f "./datasets/sighan_2015_test.csv" ]; then
    echo "[INFO] 检测到已有 CSV 数据文件，跳过数据准备步骤。"
    echo "       如需强制重新生成，请手动运行："
    echo "       python scripts/prepare_datasets.py"
else
    echo "[INFO] 正在运行数据准备脚本..."
    HF_ENDPOINT=https://hf-mirror.com python scripts/prepare_datasets.py
fi

# ─────────────────────────────────
# Step 1: 阶段一预训练
# ─────────────────────────────────
echo ""
echo "─────────────────────────────────────"
echo " Step 1: 阶段一 - Wang271K + SIGHAN 预训练"
echo "─────────────────────────────────────"

if [ "${SKIP_STAGE1}" = "1" ]; then
    echo "[INFO] 已设置 SKIP_STAGE1=1，跳过阶段一训练。"
    if [ ! -f "./ckpt/stage1/best.ckpt" ]; then
        echo "[WARN] 跳过阶段一但未找到 ckpt/stage1/best.ckpt，阶段二可能失败！"
    fi
else
    bash scripts/train_stage1_pretrain.sh
fi

# ─────────────────────────────────
# Step 2: 阶段二精调
# ─────────────────────────────────
echo ""
echo "─────────────────────────────────────"
echo " Step 2: 阶段二 - CSCD 跨域精调"
echo "─────────────────────────────────────"

# 检查 CSCD 数据是否存在；如果不存在就跳过阶段二
if [ ! -f "./datasets/cscd_ns_train.csv" ] && [ ! -f "./datasets/cscd_ime_train.csv" ]; then
    echo "[WARN] 未找到 CSCD 训练数据，跳过阶段二。"
    echo "       最终模型权重位于：ckpt/stage1/best.ckpt"
else
    bash scripts/train_stage2_finetune.sh
fi

# ─────────────────────────────────
# 完成提示
# ─────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║               🎉 完整训练流程结束！                   ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║ 最优权重路径（阶段二）：ckpt/stage2/best.ckpt        ║"
echo "║ 回退权重路径（阶段一）：ckpt/stage1/best.ckpt        ║"
echo "║                                                      ║"
echo "║ 推理命令示例：                                        ║"
echo "║  HF_ENDPOINT=https://hf-mirror.com \\                ║"
echo "║  python scripts/run_masc_csc.py \\                   ║"
echo "║    --ckpt-path ckpt/stage2/best.ckpt \\              ║"
echo "║    --sentence '我喜换吃平果，逆呢？'                  ║"
echo "╚══════════════════════════════════════════════════════╝"
