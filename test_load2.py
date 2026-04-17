import torch
import argparse
from models.multimodal_frontend import MultimodalCSCFrontend

args = argparse.Namespace(hyper_params={"dropout": 0.1}, device="cpu")
model = MultimodalCSCFrontend(args)
state_dict = torch.load("ckpt/hf_pretrained_frontend.ckpt")["state_dict"]

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("Missing keys:", missing)
if "cls.predictions.decoder.weight" in missing:
    print("FATAL: CLS weights are missing!!!")
if "bert.embeddings.word_embeddings.weight" in missing:
    print("FATAL: BERT weights are missing!!!")

# Get random sentence prediction
print("Prediction test:", model.predict("今天气色真好"))
