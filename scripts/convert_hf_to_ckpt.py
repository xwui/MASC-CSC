import torch
import os

# 读取从 HuggingFace 也就是 NamBert-for-csc 下载的权重
hf_state_dict = torch.load("ckpt/chinese-macbert-base/pytorch_model.bin", map_location='cpu')

# Pytorch Lightning 期望有个包装，放在 state_dict 这个 key 下面
lightning_ckpt = {
    "state_dict": hf_state_dict,
    "epoch": 0,
    "global_step": 0
}

ckpt_path = "ckpt/hf_pretrained_frontend.ckpt"
torch.save(lightning_ckpt, ckpt_path)
print(f"转换成功！完美的预训练权重已经保存到: {ckpt_path}")
