# Qwen2.5-3B-Instruct 训练目录要求

当前仓库里的 `scripts/train_llm_lora.sh` 默认会去找：

- `./LLM/Qwen2.5-3B-Instruct`

这是当前环境下最稳妥的默认选择，因为本机 `transformers 4.40.0` 能直接支持 Qwen2.5 纯文本 CausalLM。

## 1. 可以用于当前训练脚本的模型目录

至少需要这些文件：

- `config.json`
- `tokenizer_config.json`
- `tokenizer.json` 或 `vocab.json`
- `model.safetensors` 或 `model.safetensors.index.json` 对应的全部分片文件

推荐模型：

- `Qwen/Qwen2.5-3B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`

## 2. 不能直接用于当前训练脚本的模型目录

下面这种目录不能直接训练：

- `architectures` 包含 `Qwen3_5ForConditionalGeneration`
- `config.json` 含有 `vision_config`
- 实际上是图文多模态检查点

当前仓库里的 `./LLM/Qwen3.5-2B` 就属于这一类。

另外，如果目录仍然是 `model_type=qwen3_5`，但又没有 `auto_map` 和对应的自定义代码，当前环境也不能直接训练。

## 3. 推荐做法

如果你按当前默认配置训练，请把本地模型仓库放到：

- `./LLM/Qwen2.5-3B-Instruct`

然后执行：

```bash
bash scripts/train_llm_lora.sh
```

如果你拿到的是别的纯文本 CausalLM，也可以临时指定：

```bash
MODEL_PATH=/你的/纯文本模型目录 bash scripts/train_llm_lora.sh
```
