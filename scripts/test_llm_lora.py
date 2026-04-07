import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_CORRECTION_PROMPT = (
    "请纠正下面句子中的中文拼写错误；如果原句没有错误，保持原句不变。\n"
    "只输出纠正后的句子，不要添加解释。"
)

DEFAULT_TEST_SENTENCES = [
    "我喜换吃平果。",
    "请提钱把文件发给我。",
    "今天天气很不错。",
    "施魏因施泰格在欧冠决赛媒体公开日时说到“我们每天都能看到他的到来是多么的重要，他毫不吝啬付出自己的全部”。",
]


def _bnb_getitem(self, key):
    return getattr(self, key)


def _bnb_contains(self, key):
    return hasattr(self, key)


BitsAndBytesConfig.__getitem__ = _bnb_getitem
BitsAndBytesConfig.__contains__ = _bnb_contains
BitsAndBytesConfig.get = lambda self, key, default=None: getattr(self, key, default)


def _patch_qwen2_rotary_embedding_device():
    try:
        from transformers.models.qwen2 import modeling_qwen2 as qwen2_modeling
    except Exception:
        return

    rotary_cls = getattr(qwen2_modeling, "Qwen2RotaryEmbedding", None)
    if rotary_cls is None:
        return

    original_forward = getattr(rotary_cls, "forward", None)
    if original_forward is None or getattr(original_forward, "_masc_rotary_device_patch", False):
        return

    def safe_forward(self, x, seq_len=None):
        seq_len = seq_len if seq_len is not None else x.shape[-2]
        target_device = x.device

        if self.inv_freq.device != target_device:
            self.inv_freq = self.inv_freq.to(target_device)

        if (
            seq_len > self.max_seq_len_cached
            or self.cos_cached.device != target_device
            or self.sin_cached.device != target_device
        ):
            self._set_cos_sin_cache(
                seq_len=max(seq_len, self.max_seq_len_cached),
                device=target_device,
                dtype=x.dtype,
            )

        return original_forward(self, x, seq_len=seq_len)

    safe_forward._masc_rotary_device_patch = True
    rotary_cls.forward = safe_forward


def parse_args():
    parser = argparse.ArgumentParser(description="使用基座模型 + LoRA adapter 做中文拼写纠错推理")
    parser.add_argument("--base_model", default="./LLM/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_path", default="./ckpt/llm_lora_qwen25_05b")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("sentences", nargs="*", help="要测试的句子；不传则使用内置样例")
    return parser.parse_args()


def build_prompt(sentence: str) -> str:
    return (
        f"User:\n{DEFAULT_CORRECTION_PROMPT}\n原句：{sentence.strip()}\n\n"
        f"Assistant:"
    )


def main():
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    _patch_qwen2_rotary_embedding_device()

    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    sentences = args.sentences or DEFAULT_TEST_SENTENCES
    for idx, sentence in enumerate(sentences, start=1):
        prompt = build_prompt(sentence)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
        print(f"[{idx}] 输入: {sentence}")
        print(f"[{idx}] 输出: {prediction}")
        print()


if __name__ == "__main__":
    main()
