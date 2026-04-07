#!/bin/bash
# ==============================================================================
# setup_env.sh
# 一键搭建 MASC-CSC 训练环境（Linux + CUDA 11.8）
#
# 使用方法：
#   bash setup_env.sh
#
# 前提条件：
#   1. 已安装 Anaconda 或 Miniconda
#   2. 服务器有 NVIDIA GPU 且已安装 CUDA Driver >= 11.8
# ==============================================================================

set -e

ENV_NAME="cwq_masc_csc"

echo "========================================================"
echo " MASC-CSC 环境自动配置脚本"
echo " 目标环境：${ENV_NAME}"
echo "========================================================"

# ── Step 1: 创建 Conda 虚拟环境 ──
echo ""
echo "[1/4] 创建 Conda 环境 ${ENV_NAME} (Python 3.10)..."
if conda info --envs | grep -q "${ENV_NAME}"; then
    echo "  环境 ${ENV_NAME} 已存在，跳过创建。"
else
    conda create -n "${ENV_NAME}" python=3.10 -y
fi

# 激活环境（兼容 conda init 未执行的情况）
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "  Python: $(python --version)"
echo "  Path:   $(which python)"

# ── Step 2: 安装 PyTorch (CUDA 11.8) ──
echo ""
echo "[2/4] 安装 PyTorch 2.1.2 + CUDA 11.8..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

# ── Step 3: 安装项目依赖 ──
echo ""
echo "[3/4] 安装项目依赖 (requirements.txt)..."
pip install -r requirements.txt

# ── Step 4: 验证安装 ──
echo ""
echo "[4/4] 验证安装结果..."
python -c "
import torch
import transformers
import lightning
print('  PyTorch:       ', torch.__version__)
print('  CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  GPU Name:      ', torch.cuda.get_device_name(0))
    print('  CUDA Version:  ', torch.version.cuda)
print('  Transformers:  ', transformers.__version__)
print('  Lightning:     ', lightning.__version__)
print()
print('  ✅ 所有核心依赖安装成功！')
"

echo ""
echo "========================================================"
echo " 🎉 环境配置完成！"
echo ""
echo " 激活环境：conda activate ${ENV_NAME}"
echo " 开始训练：bash train_all.sh"
echo "========================================================"
