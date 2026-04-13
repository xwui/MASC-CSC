"""
MASC-CSC Router MLP 训练脚本
=============================

训练思路：
  对于每条 (src, tgt) 样本，用已训练好的 BERT 前端跑一遍 predict_with_metadata()，
  对比 BERT 的预测和真值：
    - BERT 预测正确 → label = 0（不需要调 LLM）
    - BERT 预测错误 → label = 1（需要调 LLM 补救）
  从 SentencePrediction 中提取 10 维特征，训练 RouterMLP 做二分类。

用法：
  python scripts/train_router.py \
    --ckpt-path ./ckpt/frontend.ckpt \
    --data datasets/Sighan/sighan15_test.csv \
    --output ./ckpt/router_mlp.pt \
    --epochs 50
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 设置项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from masc_csc.mechanism import MechanismInferencer
from masc_csc.router import RouterMLP, extract_features
from masc_csc.types import (
    ErrorMechanism,
    PositionPrediction,
    SentencePrediction,
    TokenAlternative,
)


def parse_args():
    parser = argparse.ArgumentParser(description="训练 MASC-CSC Router MLP")
    parser.add_argument("--ckpt-path", type=str, required=True,
                        help="已训练好的 BERT 前端 checkpoint 路径")
    parser.add_argument("--data", type=str, required=True,
                        help="训练数据路径，CSV 格式 (src,tgt)。支持逗号分隔多个文件")
    parser.add_argument("--output", type=str, default="./ckpt/router_mlp.pt",
                        help="训练好的 Router MLP 权重保存路径")
    parser.add_argument("--device", type=str, default="auto",
                        help="设备: auto, cpu, cuda")
    parser.add_argument("--epochs", type=int, default=50,
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="学习率")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="训练 batch size")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="验证集比例")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pos-weight", type=float, default=1.0,
                        help="正样本（需要LLM）的损失权重，用于处理类别不平衡")
    return parser.parse_args()


def load_csv_pairs(data_path: str):
    """加载 CSV 数据，返回 (src, tgt) 列表"""
    pairs = []
    files = [p.strip() for p in data_path.split(",") if p.strip()]
    for filepath in files:
        filepath = filepath.strip()
        if not os.path.exists(filepath):
            print(f"[WARNING] 文件不存在，跳过: {filepath}")
            continue
        with open(filepath, "r", encoding="utf-8") as f:
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
    """加载 JSONL 数据，返回 (src, tgt) 列表"""
    import json
    pairs = []
    files = [p.strip() for p in data_path.split(",") if p.strip()]
    for filepath in files:
        if not os.path.exists(filepath):
            print(f"[WARNING] 文件不存在，跳过: {filepath}")
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                src = obj.get("source", obj.get("src", "")).strip()
                tgt = obj.get("target", obj.get("tgt", "")).strip()
                if src and tgt and len(src) == len(tgt):
                    pairs.append((src, tgt))
    return pairs


def load_frontend_model(ckpt_path: str, device: str):
    """加载已训练好的 BERT 前端模型"""
    from models.multimodal_frontend import MultimodalCSCFrontend

    args = SimpleNamespace(device=device, hyper_params={})
    model = MultimodalCSCFrontend(args)

    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        state_dict = state["state_dict"] if "state_dict" in state else state
        model.load_state_dict(state_dict, strict=False)
        print(f"[*] 已加载前端 checkpoint: {ckpt_path}")
    else:
        print(f"[WARNING] checkpoint 不存在或未指定: {ckpt_path}，使用未训练的模型")

    model = model.to(device)
    model.eval()
    return model


def build_sentence_prediction(metadata: dict, mechanism_inferencer: MechanismInferencer) -> SentencePrediction:
    """将前端元数据转为 SentencePrediction（与 pipeline 中逻辑一致）"""
    positions = []
    for index, source_token in enumerate(metadata["source_tokens"]):
        alternative_tokens = metadata["topk_tokens"][index]
        alternative_ids = metadata["topk_ids"][index]
        alternative_scores = metadata["topk_probs"][index]
        alternatives = [
            TokenAlternative(token=token, token_id=token_id, score=float(score))
            for token, token_id, score in zip(alternative_tokens, alternative_ids, alternative_scores)
        ]
        mechanism = mechanism_inferencer.infer_from_alternatives(source_token, alternatives)
        positions.append(
            PositionPrediction(
                index=index,
                source_token=source_token,
                predicted_token=metadata["predicted_tokens"][index],
                detection_score=float(metadata["detection_scores"][index]),
                uncertainty=float(metadata["uncertainty_scores"][index]),
                mechanism=mechanism if isinstance(mechanism, ErrorMechanism) else ErrorMechanism.UNCERTAIN,
                alternatives=alternatives,
            )
        )
    return SentencePrediction(
        source_text=metadata["source_text"],
        predicted_text=metadata["predicted_text"],
        positions=positions,
    )


def collect_features_and_labels(
        frontend_model,
        pairs,
        mechanism_inferencer: MechanismInferencer,
        device: str,
):
    """
    遍历所有 (src, tgt) 对，提取特征和标签。

    标签策略：
      - BERT 预测 == tgt → label = 0（不需要 LLM）
      - BERT 预测 != tgt → label = 1（需要 LLM）
    """
    all_features = []
    all_labels = []
    n_correct = 0
    n_wrong = 0

    print(f"[*] 正在对 {len(pairs)} 条样本提取特征...")

    for i, (src, tgt) in enumerate(pairs):
        if (i + 1) % 100 == 0:
            print(f"    进度: {i + 1}/{len(pairs)}")

        try:
            with torch.no_grad():
                metadata = frontend_model.predict_with_metadata(src, top_k=5)
        except Exception as e:
            print(f"    [跳过] 第{i}条出错: {e}")
            continue

        prediction = build_sentence_prediction(metadata, mechanism_inferencer)
        features = extract_features(prediction)

        # 判断 BERT 预测是否正确
        bert_pred = prediction.predicted_text
        if bert_pred == tgt:
            label = 0  # BERT 搞定了，不需要 LLM
            n_correct += 1
        else:
            label = 1  # BERT 没搞定，需要 LLM
            n_wrong += 1

        all_features.append(features)
        all_labels.append(label)

    print(f"[*] 特征提取完成: BERT正确={n_correct}, BERT错误={n_wrong}, "
          f"总计={n_correct + n_wrong}")
    print(f"    正样本（需要LLM）比例: {n_wrong / max(n_correct + n_wrong, 1):.2%}")

    features_tensor = torch.stack(all_features)
    labels_tensor = torch.tensor(all_labels, dtype=torch.float32)
    return features_tensor, labels_tensor


def train_mlp(
        features: torch.Tensor,
        labels: torch.Tensor,
        args,
):
    """训练 RouterMLP"""
    torch.manual_seed(args.seed)

    # 划分训练集和验证集
    n = len(features)
    perm = torch.randperm(n)
    val_size = int(n * args.val_ratio)
    train_size = n - val_size

    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    train_features = features[train_idx]
    train_labels = labels[train_idx]
    val_features = features[val_idx]
    val_labels = labels[val_idx]

    print(f"\n[*] 训练集: {train_size} 条, 验证集: {val_size} 条")
    print(f"    训练集正样本比例: {train_labels.mean():.2%}")
    print(f"    验证集正样本比例: {val_labels.mean():.2%}")

    # 创建 DataLoader
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 初始化模型
    device = args.device
    model = RouterMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 损失函数（支持正样本加权以处理类别不平衡）
    pos_weight = torch.tensor([args.pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_acc = 0.0
    best_state = None

    print(f"\n[*] 开始训练 RouterMLP (epochs={args.epochs}, lr={args.lr})")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        # ---- 训练 ----
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            # RouterMLP 内部有 sigmoid，这里需要拿 logits
            # 所以直接用 net 的 raw output
            logits = model.net(batch_features).squeeze(-1)
            loss = criterion(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += (preds == batch_labels).sum().item()
            total_samples += len(batch_labels)
            total_loss += loss.item() * len(batch_labels)

        scheduler.step()
        train_acc = total_correct / max(total_samples, 1)
        train_loss = total_loss / max(total_samples, 1)

        # ---- 验证 ----
        model.eval()
        with torch.no_grad():
            val_feat = val_features.to(device)
            val_lab = val_labels.to(device)
            val_logits = model.net(val_feat).squeeze(-1)
            val_preds = (torch.sigmoid(val_logits) >= 0.5).float()
            val_acc = (val_preds == val_lab).float().mean().item()

            # 计算更多指标
            tp = ((val_preds == 1) & (val_lab == 1)).sum().item()
            fp = ((val_preds == 1) & (val_lab == 0)).sum().item()
            fn = ((val_preds == 0) & (val_lab == 1)).sum().item()
            tn = ((val_preds == 0) & (val_lab == 0)).sum().item()

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | "
                  f"Train Loss={train_loss:.4f} Acc={train_acc:.3f} | "
                  f"Val Acc={val_acc:.3f} P={precision:.3f} R={recall:.3f} F1={f1:.3f} | "
                  f"(TP={tp} FP={fp} FN={fn} TN={tn})")

    print("=" * 60)
    print(f"[*] 训练完成. 最佳验证准确率: {best_val_acc:.4f}")

    # 保存最优模型
    if best_state is not None:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        torch.save(best_state, args.output)
        print(f"[*] 模型已保存至: {args.output}")
        print(f"    参数量: {sum(p.numel() for p in model.parameters())}")
    else:
        print("[WARNING] 未找到最优模型，未保存。")


def main():
    args = parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] 设备: {args.device}")

    # 加载数据
    data_files = args.data
    if any(f.strip().endswith(".jsonl") for f in data_files.split(",")):
        pairs = load_jsonl_pairs(data_files)
    else:
        pairs = load_csv_pairs(data_files)
    print(f"[*] 加载了 {len(pairs)} 条 (src, tgt) 对")

    if len(pairs) == 0:
        print("[ERROR] 没有可用数据，退出。")
        return

    # 加载前端模型
    frontend_model = load_frontend_model(args.ckpt_path, args.device)

    # 提取特征和标签
    mechanism_inferencer = MechanismInferencer()
    features, labels = collect_features_and_labels(
        frontend_model, pairs, mechanism_inferencer, args.device
    )

    if len(features) == 0:
        print("[ERROR] 特征提取失败，没有有效样本。")
        return

    # 训练 MLP
    train_mlp(features, labels, args)

    print("\n[*] 全部完成！")
    print(f"    使用方式: MLPRouter(checkpoint_path='{args.output}')")


if __name__ == "__main__":
    main()
