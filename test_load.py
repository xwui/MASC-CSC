from transformers import AutoModel
import sys
try:
    model = AutoModel.from_pretrained("ckpt/chinese-macbert-base", trust_remote_code=True)
    print("Model type:", type(model))
    print("Keys loaded:", hasattr(model, 'pinyin_embeddings'), hasattr(model, 'cls'))
except Exception as e:
    print("Error:", e)
