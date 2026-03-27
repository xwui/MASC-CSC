# MASC-CSC

`MASC-CSC` stands for `Mechanism-Aware Selective Collaboration for Chinese Spelling Correction`.

This repository contains a Chinese spelling correction project with two layers:

- a `multimodal frontend` for character-level correction
- a `MASC-CSC collaboration layer` for mechanism inference, constrained candidate generation, selective routing, and local LLM verification

The repository is intended not only for running code, but also for continued development and research handoff.

## 1. Repository Layout

```text
MASC_CSC/
├── docs/
│   ├── 01_implementation_thinking_zh.md
│   ├── 02_architecture_and_interfaces_zh.md
│   └── 03_development_handover_and_limitations_zh.md
├── models/
│   └── multimodal_frontend.py
├── masc_csc/
│   ├── mechanism.py
│   ├── candidate_generator.py
│   ├── router.py
│   ├── llm_verifier.py
│   └── pipeline.py
├── scripts/
│   ├── data_process.py
│   └── run_masc_csc.py
├── utils/
├── train.py
├── train_frontend.sh
└── requirements.txt
```

## 2. Environment

Recommended environment:

- Python >= 3.8
- PyTorch >= 2.0
- PyTorch Lightning >= 2.0

Example:

```bash
conda create -n masc_csc python=3.10
conda activate masc_csc
cd MASC_CSC
pip install -r requirements.txt
```

Current `requirements.txt` includes:

- `lightning`
- `transformers`
- `torch`
- `torchvision`
- `pypinyin`

## 3. Data Format

The current frontend training pipeline expects CSV files with the following format:

```text
src,tgt
我喜换吃平果,我喜欢吃苹果
今田天气很好,今天天气很好
```

Requirements:

- `src` and `tgt` must have the same length
- commas are not allowed inside the text
- the first line must be the header `src,tgt`

## 4. Data Preparation

If you already have CSC pickle data in the expected format, you can convert it with:

```bash
python scripts/data_process.py
```

By default, this script generates:

- `sighan_2013_test.csv`
- `sighan_2014_test.csv`
- `sighan_2015_test.csv`
- `train.csv`

## 5. Frontend Training

The current `train.py` is for the frontend model only.

Supported model names:

- `multimodal_frontend`
- `frontend`
- `masc_frontend`

### Minimal training example

```bash
python train.py \
  --model multimodal_frontend \
  --datas train.csv \
  --batch-size 32 \
  --epochs 20 \
  --workers 0
```

### Use the provided shell script

```bash
sh train_frontend.sh
```

### Test a frontend checkpoint

```bash
python train.py \
  --model multimodal_frontend \
  --data sighan_2015_test.csv \
  --batch-size 32 \
  --ckpt-path ./ckpt/frontend.ckpt \
  --test
```

## 6. MASC-CSC Inference

Run the collaboration pipeline on a single sentence:

```bash
python scripts/run_masc_csc.py \
  --ckpt-path ./ckpt/frontend.ckpt \
  --sentence "我喜换吃平果，逆呢？"
```

### With local LLM verifier enabled

```bash
python scripts/run_masc_csc.py \
  --ckpt-path ./ckpt/frontend.ckpt \
  --sentence "我喜换吃平果，逆呢？" \
  --use-llm \
  --llm-model Qwen/Qwen3-8B \
  --llm-base-url http://127.0.0.1:8000/v1
```

Expected local verifier requirements:

- an OpenAI-compatible local service
- `/v1/chat/completions` endpoint
- a model that can follow structured choice prompts

## 7. What Is Implemented Now

Currently implemented:

- frontend model training
- `predict_with_metadata()` for frontend output exposure
- heuristic error mechanism inference
- constrained candidate generation
- risk-aware routing
- local LLM verifier interface
- end-to-end single-sentence collaboration pipeline

## 8. What Is Not Finished Yet

Not fully implemented yet:

- explicit trainable mechanism classification head
- batch-level MASC-CSC evaluation script
- verifier fine-tuning / LoRA
- clean-sentence over-correction benchmark script
- mechanism-wise evaluation script
- cost analysis script

This means the repository is already usable for research prototyping, but it is not yet a fully polished benchmark package.

## 9. Recommended Reading Order

If someone receives this repository and wants to continue development, read in this order:

1. `README.md`
2. `docs/01_implementation_thinking_zh.md`
3. `docs/02_architecture_and_interfaces_zh.md`
4. `docs/03_development_handover_and_limitations_zh.md`
5. `models/multimodal_frontend.py`
6. `masc_csc/pipeline.py`

## 10. Documentation

This repository includes three handoff-oriented Chinese documents:

- `docs/01_implementation_thinking_zh.md`
  - explains the implementation logic and research motivation
- `docs/02_architecture_and_interfaces_zh.md`
  - explains architecture, modules, and interface contracts
- `docs/03_development_handover_and_limitations_zh.md`
  - explains known limitations and recommended next development steps

## 11. Current Development Focus

If you continue this project, the most recommended next steps are:

1. verify the quality of `predict_with_metadata()`
2. improve `mechanism.py`
3. improve `candidate_generator.py`
4. connect and stress-test the local verifier
5. add experiment scripts for paper use
