#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/bkai/anaconda3/envs/cwq_masc_csc/bin/python}"
CKPT_PATH="${CKPT_PATH:-./ckpt/chinese-macbert-base}"
DATA_PATH="${DATA_PATH:-./datasets/cscd-ns/test.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-./results/cscd_ns_frontend_only_final.jsonl}"
DEVICE="${DEVICE:-cuda:1}"

export CKPT_PATH DATA_PATH OUTPUT_PATH DEVICE

cd "${ROOT_DIR}"

"${PYTHON_BIN}" - <<'PYCODE'
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path('/home/bkai/cwq/MASC-CSC')
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from masc_csc.frontend_adapter import HFNamBertFrontendAdapter

ckpt_path = os.environ['CKPT_PATH']
data_path = os.environ['DATA_PATH']
output_path = os.environ['OUTPUT_PATH']
device = os.environ['DEVICE']
summary_path = output_path + '.summary.json'

pairs = []
with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        src = obj.get('source', obj.get('src', '')).rstrip("\n")
        tgt = obj.get('target', obj.get('tgt', '')).rstrip("\n")
        if src and tgt:
            pairs.append((src, tgt))

print(f'[*] Frontend-only eval start: {len(pairs)} samples')
print(f'[*] Model: {ckpt_path}')
print(f'[*] Data: {data_path}')
print(f'[*] Device: {device}')

model = HFNamBertFrontendAdapter(ckpt_path, device)

results = []
total = len(pairs)
correct_count = 0
tp = fp = fn = tn = 0
start = time.time()

for i, (src, tgt) in enumerate(pairs, 1):
    if i % 100 == 0:
        print(f'  进度: {i}/{total}', flush=True)
    pred = model.predict(src)
    correct = pred == tgt
    if correct:
        correct_count += 1
    if src == tgt:
        if pred == tgt:
            tn += 1
        else:
            fp += 1
    else:
        if pred == tgt:
            tp += 1
        elif pred == src:
            fn += 1
        else:
            fp += 1
    results.append({
        'source': src,
        'target': tgt,
        'prediction': pred,
        'correct': correct,
        'reason': 'frontend_only',
    })

elapsed = time.time() - start
accuracy = correct_count / total if total else 0.0
precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump({
        'dataset': data_path,
        'total': total,
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'elapsed_seconds': round(elapsed, 2),
    }, f, ensure_ascii=False, indent=2)

print("\n[✓] Frontend-only eval completed")
print(f'output: {output_path}')
print(f'summary: {summary_path}')
print(f'accuracy={accuracy:.4f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}')
print(f'TP={tp} FP={fp} FN={fn} TN={tn}')
print(f'elapsed_seconds={elapsed:.2f}')
PYCODE
