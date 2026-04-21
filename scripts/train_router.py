"""Train the sentence-level invoke head for the selective router."""

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _ensure_cuda_runtime_env():
    env = os.environ.copy()
    updated = False

    required_lib_dirs = [
        "/home/bkai/anaconda3/envs/cwq_masc_csc/lib",
        "/home/bkai/anaconda3/envs/cwq_masc_csc/lib/python3.10/site-packages/nvidia/cusparse/lib",
        "/home/bkai/anaconda3/envs/cwq_masc_csc/lib/python3.10/site-packages/nvidia/cuda_runtime/lib",
    ]
    current_ld = env.get("LD_LIBRARY_PATH", "")
    ld_parts = [part for part in current_ld.split(":") if part]
    for lib_dir in reversed(required_lib_dirs):
        if lib_dir and lib_dir not in ld_parts:
            ld_parts.insert(0, lib_dir)
            updated = True
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)

    current_pythonpath = env.get("PYTHONPATH", "")
    py_parts = [part for part in current_pythonpath.split(":") if part]
    if "/home/bkai/cwq/packages" not in py_parts:
        py_parts.insert(0, "/home/bkai/cwq/packages")
        env["PYTHONPATH"] = ":".join(py_parts)
        updated = True

    if env.get("TOKENIZERS_PARALLELISM") != "false":
        env["TOKENIZERS_PARALLELISM"] = "false"
        updated = True

    if updated and env.get("MASC_ROUTER_ENV_READY") != "1":
        env["MASC_ROUTER_ENV_READY"] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)

    os.environ.update(
        {
            "LD_LIBRARY_PATH": env["LD_LIBRARY_PATH"],
            "PYTHONPATH": env.get("PYTHONPATH", ""),
            "TOKENIZERS_PARALLELISM": env["TOKENIZERS_PARALLELISM"],
            "MASC_ROUTER_ENV_READY": env.get("MASC_ROUTER_ENV_READY", "0"),
        }
    )


_ensure_cuda_runtime_env()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from masc_csc.candidate_generator import MechanismAwareCandidateGenerator
from masc_csc.llm_verifier import BaichuanLocalVerifier
from masc_csc.mechanism import MechanismInferencer
from masc_csc.selective_escalation import SelectiveRouterMLP, extract_selective_features
from masc_csc.types import ErrorMechanism, PositionPrediction, SentencePrediction, TokenAlternative


def parse_args():
    parser = argparse.ArgumentParser(description='训练句子级 gain-aware selective router 调用头')
    parser.add_argument('--ckpt-path', type=str, default=None,
                        help='前端权重来源。支持本地 HuggingFace NamBert 模型目录，或 frontend checkpoint 文件。')
    parser.add_argument('--data', type=str, required=True,
                        help='训练数据路径，支持 CSV/JSONL/目录/逗号分隔。')
    parser.add_argument('--output', type=str, default='./ckpt/selective_router.pt',
                        help='训练好的 selective router checkpoint 保存路径')
    parser.add_argument('--device', type=str, default='auto',
                        help='兼容旧参数的基础设备: auto, cpu, cuda, cuda:0。未单独指定 frontend/llm/router 设备时回退到它。')
    parser.add_argument('--frontend-device', type=str, default=None,
                        help='单 worker 模式下前端设备；未指定时回退到 --device。')
    parser.add_argument('--llm-device', type=str, default=None,
                        help='单 worker 模式下 LLM 设备；未指定时回退到 --device。')
    parser.add_argument('--router-device', type=str, default=None,
                        help='训练小 MLP 调用头的设备；未指定时回退到 --device。')
    parser.add_argument('--feature-devices', type=str, default=None,
                        help='逗号分隔的设备列表。长度>1 时，按 shard 启动多个 worker 并行提取 router 特征，例如 cuda:0,cuda:1。')
    parser.add_argument('--llm-devices', type=str, default=None,
                        help='逗号分隔的 LLM 设备列表。仅在 label-mode=llm-gain 且 feature-devices>1 时使用；长度需为1或与 feature-devices 一致。')
    parser.add_argument('--llm-path', type=str, default=None,
                        help='用于生成 gain 标签的 LLM 路径。label-mode=llm-gain 时必填。')
    parser.add_argument('--llm-adapter', type=str, default=None,
                        help='可选的 LoRA adapter 路径。')
    parser.add_argument('--llm-mode', type=str, default='targeted', choices=['targeted', 'choice'])
    parser.add_argument('--targeted-stage', type=str, default='full', choices=['full', 'stage1_only', 'unified'],
                        help='当 llm-mode=targeted 时，指定 verifier 执行 full（三阶段）、stage1_only 或 unified（单阶段候选优先+生成补充）。')
    parser.add_argument('--label-mode', type=str, default='llm-gain', choices=['llm-gain', 'frontend-error'],
                        help='llm-gain: 学习 LLM 净收益；frontend-error: 旧版前端错误检测基线。')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--feature-top-r', type=int, default=3,
                        help='抽取 router 特征时参与 top-r 聚合的位置数。')
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pos-weight', type=float, default=0.0,
                        help='正样本权重。<=0 时自动按类别比例估计。')
    parser.add_argument('--min-gain', type=float, default=1e-4,
                        help='只有 LLM utility 超过 frontend utility 至少该值时才记为正样本。')
    return parser.parse_args()


def normalize_device(device: Optional[str], fallback: Optional[str] = None) -> str:
    device = (device or '').strip()
    if not device:
        device = fallback or 'auto'
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def parse_device_list(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    return [item.strip() for item in raw.split(',') if item.strip()]


def resolve_runtime_devices(args):
    args.device = normalize_device(args.device)
    args.frontend_device = normalize_device(args.frontend_device, args.device)
    args.llm_device = normalize_device(args.llm_device, args.device)
    args.router_device = normalize_device(args.router_device, args.device)

    feature_devices = parse_device_list(args.feature_devices)
    args.feature_devices_list = feature_devices or [args.frontend_device]

    if args.label_mode == 'llm-gain':
        llm_devices = parse_device_list(args.llm_devices)
        if llm_devices:
            if len(llm_devices) == 1 and len(args.feature_devices_list) > 1:
                print('[WARNING] --llm-devices 只有 1 张卡；多 worker 会共享同一张 LLM 卡，可能变慢或 OOM。')
                llm_devices = llm_devices * len(args.feature_devices_list)
            elif len(llm_devices) != len(args.feature_devices_list):
                raise ValueError('--llm-devices 的数量必须为 1，或与 --feature-devices 数量一致。')
        else:
            if len(args.feature_devices_list) > 1:
                print('[*] 未提供 --llm-devices；多 worker + llm-gain 模式下默认每个 worker 在各自的 feature device 上加载一份 LLM。')
                llm_devices = list(args.feature_devices_list)
            else:
                llm_devices = [args.llm_device]
    else:
        llm_devices = [args.llm_device] * len(args.feature_devices_list)

    args.llm_devices_list = llm_devices

    print(f'[*] Router head 训练设备: {args.router_device}')
    print(f'[*] Feature worker 设备: {args.feature_devices_list}')
    if args.label_mode == 'llm-gain':
        print(f'[*] LLM worker 设备: {args.llm_devices_list}')


def load_csv_pairs(data_path: str):
    pairs = []
    files = [p.strip() for p in data_path.split(',') if p.strip()]
    for filepath in files:
        if not os.path.exists(filepath):
            print(f'[WARNING] 文件不存在，跳过: {filepath}')
            continue
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                continue
            for row in reader:
                if len(row) >= 2:
                    src, tgt = row[0].strip(), row[1].strip()
                    if src and tgt and len(src) == len(tgt):
                        pairs.append((src, tgt))
    return pairs


def load_jsonl_pairs(data_path: str):
    pairs = []
    files = [p.strip() for p in data_path.split(',') if p.strip()]
    for filepath in files:
        if not os.path.exists(filepath):
            print(f'[WARNING] 文件不存在，跳过: {filepath}')
            continue
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                src = obj.get('source', obj.get('src', '')).strip()
                tgt = obj.get('target', obj.get('tgt', '')).strip()
                if src and tgt and len(src) == len(tgt):
                    pairs.append((src, tgt))
    return pairs


def load_frontend_model(ckpt_path: str, device: str):
    if ckpt_path and os.path.isdir(ckpt_path):
        from masc_csc.frontend_adapter import HFNamBertFrontendAdapter

        model = HFNamBertFrontendAdapter(ckpt_path, device)
        print(f'[*] 已加载本地 HuggingFace 前端目录: {ckpt_path} @ {device}')
        return model

    from models.multimodal_frontend import MultimodalCSCFrontend

    args = SimpleNamespace(device=device, hyper_params={})
    model = MultimodalCSCFrontend(args)
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location='cpu')
        state_dict = state['state_dict'] if 'state_dict' in state else state
        model.load_state_dict(state_dict, strict=False)
        print(f'[*] 已加载前端 checkpoint: {ckpt_path} @ {device}')
    else:
        print(f'[WARNING] checkpoint 不存在或未指定: {ckpt_path}，使用未训练的模型')

    model = model.to(device)
    model.eval()
    return model


def build_sentence_prediction(metadata: dict, mechanism_inferencer: MechanismInferencer) -> SentencePrediction:
    positions = []
    n_tokens = min(len(metadata['source_tokens']), len(metadata.get('topk_tokens', [])))
    for index in range(n_tokens):
        source_token = metadata['source_tokens'][index]
        alternative_tokens = metadata['topk_tokens'][index]
        alternative_ids = metadata['topk_ids'][index]
        alternative_scores = metadata['topk_probs'][index]
        alternatives = [
            TokenAlternative(token=token, token_id=token_id, score=float(score))
            for token, token_id, score in zip(alternative_tokens, alternative_ids, alternative_scores)
        ]
        mechanism = mechanism_inferencer.infer_from_alternatives(source_token, alternatives)
        positions.append(
            PositionPrediction(
                index=index,
                source_token=source_token,
                predicted_token=metadata['predicted_tokens'][index],
                detection_score=float(metadata['detection_scores'][index]),
                uncertainty=float(metadata['uncertainty_scores'][index]),
                mechanism=mechanism if isinstance(mechanism, ErrorMechanism) else ErrorMechanism.UNCERTAIN,
                alternatives=alternatives,
            )
        )
    return SentencePrediction(
        source_text=metadata['source_text'],
        predicted_text=metadata['predicted_text'],
        positions=positions,
    )


def correction_utility(pred_text: str, tgt_text: str) -> float:
    if not tgt_text:
        return 0.0
    max_len = max(len(pred_text), len(tgt_text), 1)
    matches = sum(1 for pred_ch, tgt_ch in zip(pred_text, tgt_text) if pred_ch == tgt_ch)
    char_score = matches / max_len
    exact_bonus = 1.0 if pred_text == tgt_text else 0.0
    return exact_bonus + char_score


def load_pairs(args):
    raw_paths = [p.strip() for p in args.data.split(',') if p.strip()]
    csv_files = []
    jsonl_files = []
    for path in raw_paths:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for filename in files:
                    if filename.endswith('.csv'):
                        csv_files.append(os.path.join(root, filename))
                    elif filename.endswith('.jsonl'):
                        jsonl_files.append(os.path.join(root, filename))
        else:
            if path.endswith('.csv'):
                csv_files.append(path)
            elif path.endswith('.jsonl'):
                jsonl_files.append(path)

    pairs = []
    if csv_files:
        pairs.extend(load_csv_pairs(','.join(csv_files)))
    if jsonl_files:
        pairs.extend(load_jsonl_pairs(','.join(jsonl_files)))

    if args.limit > 0:
        pairs = pairs[:args.limit]

    print(f'[*] 扫描到 {len(csv_files)} 个 CSV 文件, {len(jsonl_files)} 个 JSONL 文件')
    print(f'[*] 加载了 {len(pairs)} 条 (src, tgt) 对')
    return pairs


def build_verifier_for_device(label_mode: str, llm_path: Optional[str], llm_adapter: Optional[str], llm_mode: str,
                              targeted_stage: str, device: str):
    if label_mode == 'frontend-error':
        return None
    if not llm_path:
        raise ValueError('label-mode=llm-gain 时必须提供 --llm-path')
    return BaichuanLocalVerifier(
        model_path=llm_path,
        adapter_path=llm_adapter,
        device=device,
        mode=llm_mode,
        targeted_stage=targeted_stage,
    )


def collect_features_and_labels(frontend_model, pairs, mechanism_inferencer: MechanismInferencer,
                                candidate_generator, verifier, top_k: int, feature_top_r: int,
                                label_mode: str, min_gain: float, worker_tag: str = 'main'):
    all_features = []
    all_labels = []
    stats = {
        'total': 0,
        'frontend_exact': 0,
        'llm_exact': 0,
        'positive': 0,
        'llm_improved_exact': 0,
        'llm_improved_utility': 0,
    }

    print(f'[*] [{worker_tag}] 正在对 {len(pairs)} 条样本提取特征与 gain 标签...')

    for i, (src, tgt) in enumerate(pairs):
        if (i + 1) % 50 == 0:
            print(f'    [{worker_tag}] 进度: {i + 1}/{len(pairs)}')

        try:
            with torch.no_grad():
                metadata = frontend_model.predict_with_metadata(src, top_k=top_k)
        except Exception as exc:
            print(f'    [{worker_tag}] [跳过] 第{i}条前端出错: {exc}')
            continue

        prediction = build_sentence_prediction(metadata, mechanism_inferencer)
        features = extract_selective_features(prediction, top_r=feature_top_r).cpu()

        frontend_pred = prediction.predicted_text
        frontend_score = correction_utility(frontend_pred, tgt)

        if frontend_pred == tgt:
            stats['frontend_exact'] += 1

        if label_mode == 'frontend-error':
            label = 1.0 if frontend_pred != tgt else 0.0
        else:
            try:
                candidates = candidate_generator.generate(prediction)
                llm_result = verifier.verify(
                    prediction,
                    candidates,
                    selected_positions=None,
                )
                llm_pred = llm_result.text
                llm_score = correction_utility(llm_pred, tgt)
            except Exception as exc:
                print(f'    [{worker_tag}] [跳过] 第{i}条 LLM 标签生成出错: {exc}')
                continue

            gain = llm_score - frontend_score
            label = 1.0 if gain > min_gain else 0.0

            if llm_pred == tgt:
                stats['llm_exact'] += 1
            if frontend_pred != tgt and llm_pred == tgt:
                stats['llm_improved_exact'] += 1
            if gain > min_gain:
                stats['llm_improved_utility'] += 1

        stats['total'] += 1
        stats['positive'] += int(label > 0.5)
        all_features.append(features)
        all_labels.append(label)

    if not all_features:
        return None, None, stats

    features_tensor = torch.stack(all_features).cpu()
    labels_tensor = torch.tensor(all_labels, dtype=torch.float32).cpu()
    return features_tensor, labels_tensor, stats


def summarize_feature_stats(stats: Dict[str, int], label_mode: str):
    print('[*] 特征提取完成:')
    print(f"    总样本: {stats['total']}")
    print(f"    正样本: {stats['positive']} ({stats['positive'] / max(stats['total'], 1):.2%})")
    print(f"    前端 exact: {stats['frontend_exact']}")
    if label_mode == 'llm-gain':
        print(f"    LLM exact: {stats['llm_exact']}")
        print(f"    LLM exact 改善: {stats['llm_improved_exact']}")
        print(f"    LLM utility 改善: {stats['llm_improved_utility']}")


def merge_stats(stats_list: Sequence[Dict[str, int]]) -> Dict[str, int]:
    merged: Dict[str, int] = {
        'total': 0,
        'frontend_exact': 0,
        'llm_exact': 0,
        'positive': 0,
        'llm_improved_exact': 0,
        'llm_improved_utility': 0,
    }
    for stats in stats_list:
        for key in merged:
            merged[key] += int(stats.get(key, 0))
    return merged


def shard_pairs(pairs: Sequence[Tuple[str, str]], num_shards: int) -> List[List[Tuple[str, str]]]:
    shards: List[List[Tuple[str, str]]] = [[] for _ in range(num_shards)]
    for idx, pair in enumerate(pairs):
        shards[idx % num_shards].append(pair)
    return shards


def _feature_worker_entry(payload):
    worker_id, pairs, ckpt_path, frontend_device, label_mode, llm_path, llm_adapter, llm_mode, targeted_stage, llm_device, top_k, feature_top_r, min_gain = payload
    worker_tag = f'worker-{worker_id}@{frontend_device}'
    torch.set_num_threads(1)

    frontend_model = load_frontend_model(ckpt_path, frontend_device)
    mechanism_inferencer = MechanismInferencer()
    candidate_generator = MechanismAwareCandidateGenerator(mechanism_inferencer=mechanism_inferencer)
    verifier = build_verifier_for_device(label_mode, llm_path, llm_adapter, llm_mode, targeted_stage, llm_device)

    features, labels, stats = collect_features_and_labels(
        frontend_model,
        pairs,
        mechanism_inferencer,
        candidate_generator,
        verifier,
        top_k=top_k,
        feature_top_r=feature_top_r,
        label_mode=label_mode,
        min_gain=min_gain,
        worker_tag=worker_tag,
    )
    return features, labels, stats


def collect_features_and_labels_parallel(pairs, args):
    shards = shard_pairs(pairs, len(args.feature_devices_list))
    payloads = []
    for worker_id, shard in enumerate(shards):
        if not shard:
            continue
        payloads.append((
            worker_id,
            shard,
            args.ckpt_path,
            args.feature_devices_list[worker_id],
            args.label_mode,
            args.llm_path,
            args.llm_adapter,
            args.llm_mode,
            args.targeted_stage,
            args.llm_devices_list[worker_id],
            args.top_k,
            args.feature_top_r,
            args.min_gain,
        ))

    if not payloads:
        return None, None, {'total': 0, 'frontend_exact': 0, 'llm_exact': 0, 'positive': 0,
                            'llm_improved_exact': 0, 'llm_improved_utility': 0}

    print(f'[*] 启动 {len(payloads)} 个 feature worker 并行提取特征...')
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=len(payloads)) as pool:
        results = pool.map(_feature_worker_entry, payloads)

    feature_parts = [features for features, _, _ in results if features is not None and len(features) > 0]
    label_parts = [labels for _, labels, _ in results if labels is not None and len(labels) > 0]
    merged_stats = merge_stats([stats for _, _, stats in results])

    summarize_feature_stats(merged_stats, args.label_mode)

    if not feature_parts:
        return None, None, merged_stats

    return torch.cat(feature_parts, dim=0), torch.cat(label_parts, dim=0), merged_stats


def collect_features_single_worker(pairs, args):
    frontend_model = load_frontend_model(args.ckpt_path, args.frontend_device)
    mechanism_inferencer = MechanismInferencer()
    candidate_generator = MechanismAwareCandidateGenerator(mechanism_inferencer=mechanism_inferencer)
    verifier = build_verifier_for_device(
        args.label_mode,
        args.llm_path,
        args.llm_adapter,
        args.llm_mode,
        args.targeted_stage,
        args.llm_devices_list[0],
    )

    features, labels, stats = collect_features_and_labels(
        frontend_model,
        pairs,
        mechanism_inferencer,
        candidate_generator,
        verifier,
        top_k=args.top_k,
        feature_top_r=args.feature_top_r,
        label_mode=args.label_mode,
        min_gain=args.min_gain,
        worker_tag='main',
    )
    if features is not None:
        summarize_feature_stats(stats, args.label_mode)
    return features, labels, stats


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def compute_metrics_from_probs(probs: torch.Tensor, labels: torch.Tensor, threshold: float):
    preds = (probs >= threshold).float()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    acc = safe_div(tp + tn, tp + fp + fn + tn)
    return {
        'threshold': threshold,
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
    }


def select_best_threshold(probs: torch.Tensor, labels: torch.Tensor):
    best = None
    for threshold in [i / 20 for i in range(1, 20)]:
        metrics = compute_metrics_from_probs(probs, labels, threshold)
        score = (metrics['f1'], metrics['recall'], metrics['acc'])
        if best is None or score > best[0]:
            best = (score, metrics)
    return best[1]


def train_invoke_head(features: torch.Tensor, labels: torch.Tensor, args):
    torch.manual_seed(args.seed)

    n = len(features)
    perm = torch.randperm(n)
    val_size = max(1, int(n * args.val_ratio)) if n > 1 else 0
    train_size = n - val_size
    if train_size <= 0:
        train_size = n
        val_size = 0

    train_idx = perm[:train_size]
    val_idx = perm[train_size:] if val_size > 0 else perm[:0]

    train_features = features[train_idx]
    train_labels = labels[train_idx]
    val_features = features[val_idx]
    val_labels = labels[val_idx]

    print(f'\n[*] 训练集: {len(train_features)} 条, 验证集: {len(val_features)} 条')
    print(f"    训练集正样本比例: {train_labels.mean().item():.2%}")
    if len(val_labels) > 0:
        print(f"    验证集正样本比例: {val_labels.mean().item():.2%}")

    train_loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=args.batch_size, shuffle=True)
    model = SelectiveRouterMLP().to(args.router_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    positive = train_labels.sum().item()
    negative = len(train_labels) - positive
    auto_pos_weight = negative / max(positive, 1.0)
    pos_weight = args.pos_weight if args.pos_weight > 0 else auto_pos_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=args.router_device))

    best_payload = None
    best_score = None

    print(
        f"\n[*] 开始训练 sentence-level invoke head "
        f"(device={args.router_device}, epochs={args.epochs}, lr={args.lr}, pos_weight={pos_weight:.3f})"
    )
    print('=' * 72)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(args.router_device)
            batch_labels = batch_labels.to(args.router_device)

            logits = model.forward_logits(batch_features)
            loss = criterion(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_labels)
            total_samples += len(batch_labels)

        scheduler.step()
        train_loss = total_loss / max(total_samples, 1)

        model.eval()
        with torch.no_grad():
            train_probs = model(train_features.to(args.router_device)).detach().cpu()
            train_metrics = select_best_threshold(train_probs, train_labels.cpu())

            if len(val_features) > 0:
                val_probs = model(val_features.to(args.router_device)).detach().cpu()
                val_metrics = select_best_threshold(val_probs, val_labels.cpu())
            else:
                val_metrics = train_metrics

        current_score = (val_metrics['f1'], val_metrics['recall'], val_metrics['acc'])
        if best_score is None or current_score > best_score:
            best_score = current_score
            best_payload = {
                'state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                'threshold': float(val_metrics['threshold']),
                'feature_dim': int(train_features.shape[1]),
                'feature_top_r': int(args.feature_top_r),
                'training_meta': {
                    'label_mode': args.label_mode,
                    'min_gain': args.min_gain,
                    'epochs': args.epochs,
                    'lr': args.lr,
                    'pos_weight': pos_weight,
                    'router_device': args.router_device,
                    'feature_devices': args.feature_devices_list,
                    'llm_devices': args.llm_devices_list,
                    'val_metrics': val_metrics,
                },
            }

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"  Epoch {epoch:3d} | Train Loss={train_loss:.4f} | "
                f"Train F1={train_metrics['f1']:.3f} @ {train_metrics['threshold']:.2f} | "
                f"Val F1={val_metrics['f1']:.3f} P={val_metrics['precision']:.3f} "
                f"R={val_metrics['recall']:.3f} Acc={val_metrics['acc']:.3f} @ {val_metrics['threshold']:.2f}"
            )

    print('=' * 72)
    if best_payload is None:
        print('[WARNING] 未找到可保存的模型。')
        return None

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save(best_payload, args.output)
    print(f"[*] 训练完成，模型已保存至: {args.output}")
    print(f"    推荐阈值: {best_payload['threshold']:.2f}")
    return best_payload


def main():
    args = parse_args()
    resolve_runtime_devices(args)

    pairs = load_pairs(args)
    if not pairs:
        print('[ERROR] 没有可用数据，退出。')
        return

    if len(args.feature_devices_list) > 1:
        features, labels, stats = collect_features_and_labels_parallel(pairs, args)
    else:
        features, labels, stats = collect_features_single_worker(pairs, args)

    if features is None or len(features) == 0:
        print('[ERROR] 特征提取失败，没有有效样本。')
        return

    payload = train_invoke_head(features, labels, args)
    if payload is None:
        return

    print('\n[*] 全部完成！')
    print(f"    使用方式: SelectiveEscalationRouter(checkpoint_path='{args.output}', proposal_budget={args.feature_top_r})")
    print(f"    训练标签模式: {args.label_mode}")
    print(f"    样本统计: total={stats['total']}, positive={stats['positive']}")


if __name__ == '__main__':
    main()
