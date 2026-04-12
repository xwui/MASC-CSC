import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from masc_csc import MechanismInferencer, RiskAwareRouter  # noqa: E402
from masc_csc.types import ErrorMechanism, PositionPrediction, SentencePrediction, TokenAlternative  # noqa: E402
from models.multimodal_frontend import MultimodalCSCFrontend  # noqa: E402
from utils.metrics import SighanCSCMetrics  # noqa: E402


def build_model_args(device: str):
    return SimpleNamespace(
        device=device,
        hyper_params={},
    )


def load_frontend_model(checkpoint_path: str, device: str):
    args = build_model_args(device=device)
    model = MultimodalCSCFrontend(args)
    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state["state_dict"] if "state_dict" in state else state
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def resolve_data_files(data_path: str) -> List[Path]:
    resolved = []
    for raw_path in [item.strip() for item in data_path.split(",") if item.strip()]:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"找不到数据路径: {raw_path}")

        if path.is_file():
            resolved.append(path)
            continue

        if path.is_dir():
            resolved.extend(sorted(path.rglob("*.jsonl")))
            resolved.extend(sorted(path.rglob("*.csv")))
            continue

        raise ValueError(f"不支持的数据路径: {raw_path}")

    unique = []
    seen = set()
    for path in resolved:
        path = path.resolve()
        if path not in seen:
            unique.append(path)
            seen.add(path)
    if not unique:
        raise ValueError("没有解析到任何数据文件")
    return unique


def normalize_text(text: str) -> str:
    return "".join(str(text).replace(" ", "").replace("\u3000", ""))


def load_jsonl(path: Path) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "source" not in record or "target" not in record:
                continue
            yield {
                "source": normalize_text(record["source"]),
                "target": normalize_text(record["target"]),
                "label": record.get("label"),
                "meta": {
                    "source_file": str(path),
                },
            }


def load_csv(path: Path) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        lower_map = {name.lower(): name for name in fieldnames}

        src_key = lower_map.get("source") or lower_map.get("src")
        tgt_key = lower_map.get("target") or lower_map.get("tgt")
        if src_key is None or tgt_key is None:
            raise ValueError(f"CSV 缺少 source/target 或 src/tgt 列: {path}")

        for row in reader:
            yield {
                "source": normalize_text(row[src_key]),
                "target": normalize_text(row[tgt_key]),
                "label": row.get(lower_map.get("label")) if lower_map.get("label") else None,
                "meta": {
                    "source_file": str(path),
                },
            }


def load_records(data_path: str, limit: int = -1) -> List[Dict]:
    records: List[Dict] = []
    for path in resolve_data_files(data_path):
        if path.suffix == ".jsonl":
            iterator = load_jsonl(path)
        elif path.suffix == ".csv":
            iterator = load_csv(path)
        else:
            continue

        for record in iterator:
            if not record["source"] or not record["target"]:
                continue
            records.append(record)
            if limit > 0 and len(records) >= limit:
                return records
    return records


def build_sentence_prediction(frontend_model, mechanism_inferencer, sentence: str, top_k: int) -> SentencePrediction:
    metadata = frontend_model.predict_with_metadata(sentence, top_k=top_k)
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


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def build_router(args) -> RiskAwareRouter:
    router = RiskAwareRouter(
        detection_threshold=args.detection_threshold,
        uncertainty_threshold=args.uncertainty_threshold,
        margin_threshold=args.margin_threshold,
        multi_edit_threshold=args.multi_edit_threshold,
        risk_threshold=args.risk_threshold,
        detection_floor=args.detection_floor,
        detection_ceiling=args.detection_ceiling,
        uncertainty_floor=args.uncertainty_floor,
        uncertainty_ceiling=args.uncertainty_ceiling,
    )
    router.weight_uncertainty = args.weight_uncertainty
    router.weight_low_margin = args.weight_low_margin
    router.weight_detection = args.weight_detection
    router.weight_edited = args.weight_edited
    router.weight_mechanism_uncertain = args.weight_mechanism_uncertain
    router.sentence_weight_max = args.sentence_weight_max
    router.sentence_weight_top2 = args.sentence_weight_top2
    router.sentence_weight_edit_count = args.sentence_weight_edit_count
    return router


def parse_args():
    parser = argparse.ArgumentParser(description="只评估 frontend+router，不调用 LLM。")
    parser.add_argument("--ckpt-path", type=str, required=True, help="BERT frontend checkpoint 路径。")
    parser.add_argument("--data-path", type=str, required=True, help="测试数据路径，支持 jsonl/csv/目录/逗号分隔。")
    parser.add_argument("--output-dir", type=str, required=True, help="评估结果输出目录。")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=-1)

    parser.add_argument("--detection-threshold", type=float, default=0.35)
    parser.add_argument("--uncertainty-threshold", type=float, default=1.5)
    parser.add_argument("--margin-threshold", type=float, default=0.20)
    parser.add_argument("--multi-edit-threshold", type=int, default=2)
    parser.add_argument("--risk-threshold", type=float, default=0.50)
    parser.add_argument("--detection-floor", type=float, default=0.15)
    parser.add_argument("--detection-ceiling", type=float, default=0.60)
    parser.add_argument("--uncertainty-floor", type=float, default=0.80)
    parser.add_argument("--uncertainty-ceiling", type=float, default=1.80)

    parser.add_argument("--weight-uncertainty", type=float, default=0.40)
    parser.add_argument("--weight-low-margin", type=float, default=0.30)
    parser.add_argument("--weight-detection", type=float, default=0.20)
    parser.add_argument("--weight-edited", type=float, default=0.05)
    parser.add_argument("--weight-mechanism-uncertain", type=float, default=0.05)
    parser.add_argument("--sentence-weight-max", type=float, default=0.55)
    parser.add_argument("--sentence-weight-top2", type=float, default=0.25)
    parser.add_argument("--sentence-weight-edit-count", type=float, default=0.20)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" Router-Only Evaluation ")
    print("=" * 60)
    print(f"[*] ckpt-path : {args.ckpt_path}")
    print(f"[*] data-path : {args.data_path}")
    print(f"[*] output-dir: {output_dir}")
    print(f"[*] device    : {args.device}")
    print("=" * 60)

    frontend_model = load_frontend_model(args.ckpt_path, args.device)
    mechanism_inferencer = MechanismInferencer()
    router = build_router(args)
    records = load_records(args.data_path, limit=args.limit)
    if not records:
        raise ValueError("没有加载到任何评估样本")

    predictions_path = output_dir / "predictions.jsonl"
    missed_path = output_dir / "missed_bad_cases.jsonl"
    false_route_path = output_dir / "false_routes.jsonl"

    tp = fp = fn = tn = 0
    frontend_correct_count = 0
    equal_length_count = 0
    unequal_length_count = 0
    routed_risk_sum = 0.0
    skipped_risk_sum = 0.0
    wrong_risk_sum = 0.0
    correct_risk_sum = 0.0
    routed_count = 0
    skipped_count = 0
    reason_counter = Counter()
    sighan_metrics = SighanCSCMetrics()

    with open(predictions_path, "w", encoding="utf-8") as pred_f, \
            open(missed_path, "w", encoding="utf-8") as missed_f, \
            open(false_route_path, "w", encoding="utf-8") as false_route_f:
        for record in tqdm(records, desc="Evaluating router"):
            source = record["source"]
            target = record["target"]
            prediction = build_sentence_prediction(
                frontend_model=frontend_model,
                mechanism_inferencer=mechanism_inferencer,
                sentence=source,
                top_k=args.top_k,
            )
            routing = router.decide(prediction)

            frontend_pred = prediction.predicted_text
            frontend_correct = frontend_pred == target
            should_route = not frontend_correct

            if frontend_correct:
                frontend_correct_count += 1
                correct_risk_sum += routing.risk_score
            else:
                wrong_risk_sum += routing.risk_score

            if len(source) == len(target) == len(frontend_pred):
                equal_length_count += 1
                sighan_metrics.add_sentence(source, target, frontend_pred)
            else:
                unequal_length_count += 1

            if routing.invoke_llm and should_route:
                tp += 1
            elif routing.invoke_llm and not should_route:
                fp += 1
            elif (not routing.invoke_llm) and should_route:
                fn += 1
            else:
                tn += 1

            if routing.invoke_llm:
                routed_count += 1
                routed_risk_sum += routing.risk_score
            else:
                skipped_count += 1
                skipped_risk_sum += routing.risk_score

            for reason in routing.reasons:
                normalized = reason.split("(", 1)[0]
                reason_counter[normalized] += 1

            item = {
                "source": source,
                "target": target,
                "frontend_prediction": frontend_pred,
                "frontend_correct": frontend_correct,
                "should_route": should_route,
                "invoke_llm": routing.invoke_llm,
                "risk_score": routing.risk_score,
                "reasons": routing.reasons,
                "edited_count": len(prediction.edited_positions),
                "edited_indices": [position.index for position in prediction.edited_positions],
                "meta": record.get("meta", {}),
            }
            pred_f.write(json.dumps(item, ensure_ascii=False) + "\n")

            if should_route and (not routing.invoke_llm):
                missed_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            if (not should_route) and routing.invoke_llm:
                false_route_f.write(json.dumps(item, ensure_ascii=False) + "\n")

    total = len(records)
    route_precision = safe_div(tp, tp + fp)
    route_recall = safe_div(tp, tp + fn)
    route_f1 = compute_f1(route_precision, route_recall)
    good_case_skip_rate = safe_div(tn, tn + fp)
    false_route_rate = safe_div(fp, tn + fp)
    missed_bad_case_rate = safe_div(fn, tp + fn)
    route_rate = safe_div(tp + fp, total)
    router_accuracy = safe_div(tp + tn, total)
    router_balanced_accuracy = 0.5 * (route_recall + good_case_skip_rate)
    router_quality_score = 0.6 * route_f1 + 0.4 * router_balanced_accuracy

    frontend_exact_match_acc = safe_div(frontend_correct_count, total)
    mean_risk_routed = safe_div(routed_risk_sum, routed_count)
    mean_risk_skipped = safe_div(skipped_risk_sum, skipped_count)
    mean_risk_frontend_wrong = safe_div(wrong_risk_sum, total - frontend_correct_count)
    mean_risk_frontend_correct = safe_div(correct_risk_sum, frontend_correct_count)

    metrics = {
        "num_samples": total,
        "frontend_exact_match_acc": frontend_exact_match_acc,
        "frontend_error_count": total - frontend_correct_count,
        "frontend_correct_count": frontend_correct_count,
        "router_confusion": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "route_rate": route_rate,
        "router_accuracy": router_accuracy,
        "router_bad_case_precision": route_precision,
        "router_bad_case_recall": route_recall,
        "router_bad_case_f1": route_f1,
        "router_good_case_skip_rate": good_case_skip_rate,
        "false_route_rate": false_route_rate,
        "missed_bad_case_rate": missed_bad_case_rate,
        "router_balanced_accuracy": router_balanced_accuracy,
        "router_quality_score": router_quality_score,
        "mean_risk_routed": mean_risk_routed,
        "mean_risk_skipped": mean_risk_skipped,
        "mean_risk_frontend_wrong": mean_risk_frontend_wrong,
        "mean_risk_frontend_correct": mean_risk_frontend_correct,
        "equal_length_count": equal_length_count,
        "unequal_length_count": unequal_length_count,
        "top_reason_counts": dict(reason_counter.most_common(10)),
        "router_config": {
            "detection_threshold": args.detection_threshold,
            "uncertainty_threshold": args.uncertainty_threshold,
            "margin_threshold": args.margin_threshold,
            "multi_edit_threshold": args.multi_edit_threshold,
            "risk_threshold": args.risk_threshold,
            "detection_floor": args.detection_floor,
            "detection_ceiling": args.detection_ceiling,
            "uncertainty_floor": args.uncertainty_floor,
            "uncertainty_ceiling": args.uncertainty_ceiling,
            "weight_uncertainty": args.weight_uncertainty,
            "weight_low_margin": args.weight_low_margin,
            "weight_detection": args.weight_detection,
            "weight_edited": args.weight_edited,
            "weight_mechanism_uncertain": args.weight_mechanism_uncertain,
            "sentence_weight_max": args.sentence_weight_max,
            "sentence_weight_top2": args.sentence_weight_top2,
            "sentence_weight_edit_count": args.sentence_weight_edit_count,
        },
        "artifacts": {
            "predictions": str(predictions_path),
            "missed_bad_cases": str(missed_path),
            "false_routes": str(false_route_path),
        },
    }

    if equal_length_count > 0:
        (
            char_detect_acc,
            char_detect_p,
            char_detect_r,
            char_detect_f1,
            char_correct_acc,
            char_correct_p,
            char_correct_r,
            char_correct_f1,
            sent_detect_acc,
            sent_detect_p,
            sent_detect_r,
            sent_detect_f1,
            sent_correct_acc,
            sent_correct_p,
            sent_correct_r,
            sent_correct_f1,
        ) = sighan_metrics.get_results()
        metrics["frontend_sighan_metrics"] = {
            "char_detect_acc": char_detect_acc,
            "char_detect_p": char_detect_p,
            "char_detect_r": char_detect_r,
            "char_detect_f1": char_detect_f1,
            "char_correct_acc": char_correct_acc,
            "char_correct_p": char_correct_p,
            "char_correct_r": char_correct_r,
            "char_correct_f1": char_correct_f1,
            "sent_detect_acc": sent_detect_acc,
            "sent_detect_p": sent_detect_p,
            "sent_detect_r": sent_detect_r,
            "sent_detect_f1": sent_detect_f1,
            "sent_correct_acc": sent_correct_acc,
            "sent_correct_p": sent_correct_p,
            "sent_correct_r": sent_correct_r,
            "sent_correct_f1": sent_correct_f1,
        }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n=== Router-Only Evaluation Finished ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[*] metrics      : {metrics_path}")
    print(f"[*] predictions  : {predictions_path}")
    print(f"[*] missed cases : {missed_path}")
    print(f"[*] false routes : {false_route_path}")


if __name__ == "__main__":
    main()
