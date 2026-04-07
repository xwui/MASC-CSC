#!/bin/bash
# ==============================================================================
# train_stage2_finetune.sh
# 阶段二：在 CSCD-NS + CSCD-IME 混合数据上对前端模型进行跨域精调
# 目标：弥补 Wang271K/SIGHAN 与真实母语者错误分布之间的 domain gap，
#        激活 MASC-CSC 路由层的过度纠错防御机制
#
# 使用方法（在项目根目录执行）：
#   bash scripts/train_stage2_finetune.sh
#
# 前提条件：
#   1. 已运行 python scripts/prepare_datasets.py
#   2. 阶段一已完成，ckpt/stage1/best.ckpt 已存在
#      （也可修改 PRETRAIN_CKPT 指向已有的其他检查点）
# ==============================================================================

set -e

PRETRAIN_CKPT="./ckpt/stage1/best.ckpt"
WORK_DIR="./outputs/stage2"
CKPT_DIR="./ckpt/stage2"

echo "========================================================"
echo " MASC-CSC 阶段二训练：CSCD 跨域精调"
echo " 预训练检查点：${PRETRAIN_CKPT}"
echo " 输出目录：${WORK_DIR}"
echo " 检查点目录：${CKPT_DIR}"
echo "========================================================"

# 检查预训练权重是否存在
if [ ! -f "${PRETRAIN_CKPT}" ]; then
    echo "[ERROR] 找不到预训练检查点：${PRETRAIN_CKPT}"
    echo "  请先运行阶段一：bash scripts/train_stage1_pretrain.sh"
    echo "  或手动修改本脚本中的 PRETRAIN_CKPT 变量指向已有权重。"
    exit 1
fi

# 检查数据文件是否存在
if [ ! -f "./datasets/cscd_ns_train.csv" ] && [ ! -f "./datasets/cscd_ime_train.csv" ]; then
    echo "[ERROR] 找不到 CSCD 训练数据（cscd_ns_train.csv 或 cscd_ime_train.csv）"
    echo "  请先运行：python scripts/prepare_datasets.py"
    exit 1
fi

# 动态决定训练集：优先使用 CSCD-NS + CSCD-IME，若缺少 IME 则只用 NS
TRAIN_DATAS="cscd_ns_train.csv"
if [ -f "./datasets/cscd_ime_train.csv" ]; then
    TRAIN_DATAS="cscd_ns_train.csv,cscd_ime_train.csv"
    echo "[INFO] 同时使用 CSCD-NS 和 CSCD-IME 进行精调"
else
    echo "[INFO] 未找到 CSCD-IME，仅使用 CSCD-NS 进行精调"
fi

# 动态决定验证集和测试集
VAL_DATA_ARG=""
TEST_DATA=""
if [ -f "./datasets/cscd_ns_test.csv" ]; then
    VAL_DATA_ARG="--val-data cscd_ns_test.csv"
fi

# 构建测试集参数（仅包含已存在的文件）
for f in sighan_2013_test.csv sighan_2014_test.csv sighan_2015_test.csv cscd_ns_test.csv; do
    if [ -f "./datasets/${f}" ]; then
        TEST_DATA="${TEST_DATA}${f},"
    fi
done
TEST_DATA="${TEST_DATA%,}"  # 去掉末尾逗号

TEST_DATA_ARG=""
if [ -n "${TEST_DATA}" ]; then
    TEST_DATA_ARG="--test-data ${TEST_DATA}"
    TEST_DATA_ARG="--eval ${TEST_DATA_ARG}"
fi

HF_ENDPOINT=https://hf-mirror.com python train.py \
    --model multimodal_frontend \
    --datas "${TRAIN_DATAS}" \
    ${VAL_DATA_ARG} \
    --batch-size 32 \
    --epochs 20 \
    --min-epochs 5 \
    --no-resume \
    --finetune \
    --ckpt-path "${PRETRAIN_CKPT}" \
    --early-stop 5 \
    --accumulate_grad_batches 2 \
    --precision "16-mixed" \
    --workers 4 \
    ${TEST_DATA_ARG} \
    --work-dir "${WORK_DIR}" \
    --ckpt-dir "${CKPT_DIR}"

echo ""
echo "========================================================"
echo " 阶段二精调完成！"
echo " 最优检查点：${CKPT_DIR}/best.ckpt"
echo ""
echo " 可以用以下命令进行推理测试："
echo "   HF_ENDPOINT=https://hf-mirror.com python scripts/run_masc_csc.py \\"
echo "     --ckpt-path ${CKPT_DIR}/best.ckpt \\"
echo "     --sentence '我喜换吃平果，逆呢？'"
echo "========================================================"
