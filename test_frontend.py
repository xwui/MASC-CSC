from argparse import Namespace
import torch
from models.multimodal_frontend import MultimodalCSCFrontend

args = Namespace(
    hyper_params={'dropout': 0.1},
    device='cpu',
    limit_batches=-1
)
model = MultimodalCSCFrontend(args)
print("Loading weights...")
state_dict = torch.load("ckpt/chinese-macbert-base/pytorch_model.bin", map_location='cpu')
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("Missing:", missing)
print("Unexpected:", unexpected)
