"""
MASC-CSC 全流程运行脚本
========================

完整跑通 BERT前端 → 机制推理 → 候选生成 → Router路由 → LLM定点纠正 的全流程。

支持两种运行模式：
  1. 交互模式：手动输入句子，逐条纠错
  2. 评测模式：读取数据集文件，批量纠错并输出指标

用法示例：

  # 交互模式（不需要 LLM，Router 做完决定后用前端候选）
  python scripts/run_pipeline.py --mode interactive

  # 交互模式（带 LLM 验证）
  python scripts/run_pipeline.py --mode interactive --llm-path /path/to/llm

  # 评测模式（批量跑数据集）
  python scripts/run_pipeline.py --mode eval \
    --data datasets/Sighan/sighan15_test.jsonl \
    --output results/sighan15_results.jsonl

  # 评测模式（不使用 LLM，纯前端+Router）
  python scripts/run_pipeline.py --mode eval \
    --data datasets/Sighan/sighan15_test.jsonl \
    --no-llm
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import logging
logging.basicConfig(level=logging.INFO)

# 设置项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["MASC_SKIP_GLYPH_CACHE"] = "1"


def parse_args():
    parser = argparse.ArgumentParser(description="MASC-CSC 全流程运行脚本")

    # 模式
    parser.add_argument("--mode", type=str, default="interactive",
                        choices=["interactive", "eval"],
                        help="运行模式: interactive(交互) / eval(评测)")

    # 前端模型
    parser.add_argument("--ckpt-path", type=str, default=None,
                        help="（可选）前端权重来源。支持本地 HuggingFace NamBert 模型目录，"
                             "或前端 Lightning .ckpt/.bin 文件")
    parser.add_argument("--frontend-device", type=str, default="cuda:0",
                        help="前端 BERT 设备 (如 cuda:0, cpu)")
    parser.add_argument("--llm-device", type=str, default="cuda:1",
                        help="LLM 设备 (如 cuda:1, cuda:0, cpu)")

    # Router
    parser.add_argument("--router", type=str, default="selective",
                        choices=["selective", "mlp", "heuristic"],
                        help="路由器类型: selective(位置提议+选择性升级) / mlp / heuristic")
    parser.add_argument("--router-ckpt", type=str, default=None,
                        help="MLP Router 权重路径（如 ckpt/router_mlp.pt）")
    parser.add_argument("--router-threshold", type=float, default=0.45,
                        help="Router 调用 LLM 的阈值")
    parser.add_argument("--proposal-detection-threshold", type=float, default=0.08,
                        help="可疑位置提议的 detection 阈值")
    parser.add_argument("--proposal-uncertainty-threshold", type=float, default=0.80,
                        help="可疑位置提议的 uncertainty 阈值")
    parser.add_argument("--proposal-margin-threshold", type=float, default=0.20,
                        help="可疑位置提议的 top1-top2 margin 阈值")
    parser.add_argument("--proposal-budget", type=int, default=3,
                        help="Selective router 聚合句子级特征时使用的 top-r 位置数")

    # LLM
    parser.add_argument("--llm-path", type=str, default=None,
                        help="LLM 模型路径（如 baichuan-inc/Baichuan-7B）。不传则不使用 LLM。")
    parser.add_argument("--llm-adapter", type=str, default=None,
                        help="（可选）LLM LoRA adapter 路径")
    parser.add_argument("--llm-mode", type=str, default="targeted",
                        choices=["targeted", "choice"],
                        help="LLM 工作模式: targeted(定点纠正) / choice(选择题)")
    parser.add_argument("--targeted-stage", type=str, default="full",
                        choices=["full", "stage1_only"],
                        help="targeted 模式下的执行阶段：full(默认三阶段) / stage1_only(只跑 stage1，N 直接回退原字)")
    parser.add_argument("--no-llm", action="store_true",
                        help="禁用 LLM，即使 Router 判定需要也跳过")

    # 评测模式参数
    parser.add_argument("--data", type=str, default=None,
                        help="评测数据路径 (CSV 或 JSONL)")
    parser.add_argument("--output", type=str, default=None,
                        help="评测结果输出路径")

    return parser.parse_args()


# ──────────────────────────────────────────────────────────
# 加载各组件
# ──────────────────────────────────────────────────────────

def load_frontend(ckpt_path, device):
    """加载前端 BERT 模型"""
    if ckpt_path and os.path.isdir(ckpt_path):
        from masc_csc.frontend_adapter import HFNamBertFrontendAdapter

        model = HFNamBertFrontendAdapter(ckpt_path, device)
        print(f"[✓] 本地 HuggingFace 前端目录已加载: {ckpt_path}")
        return model

    from models.multimodal_frontend import MultimodalCSCFrontend

    args = SimpleNamespace(device=device, hyper_params={})
    model = MultimodalCSCFrontend(args)

    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        state_dict = state["state_dict"] if "state_dict" in state else state
        model.load_state_dict(state_dict, strict=False)
        print(f"[✓] 前端 checkpoint 已加载: {ckpt_path}")
    else:
        print(f"[!] 未指定前端 checkpoint，使用预训练 BERT + 随机初始化的自定义层")

    model = model.to(device)
    model.eval()
    return model


def build_pipeline(args):
    """组装完整的 Pipeline"""
    from masc_csc.candidate_generator import MechanismAwareCandidateGenerator
    from masc_csc.mechanism import MechanismInferencer
    from masc_csc.pipeline import MASCCSCPipeline
    from masc_csc.router import HeuristicRouter, MLPRouter
    from masc_csc.selective_escalation import SelectiveEscalationRouter
    from masc_csc.llm_verifier import BaichuanLocalVerifier, NoOpVerifier

    frontend_device = args.frontend_device
    llm_device = args.llm_device
    print(f"[*] 前端设备: {frontend_device}, LLM 设备: {llm_device}")

    # 1. 前端
    print("[1/4] 加载前端 BERT 模型...")
    frontend = load_frontend(args.ckpt_path, frontend_device)

    # 2. 机制推理 + 候选生成
    print("[2/4] 初始化机制推理 & 候选生成器...")
    mechanism_inferencer = MechanismInferencer()
    candidate_generator = MechanismAwareCandidateGenerator(
        mechanism_inferencer=mechanism_inferencer,
    )

    # 3. Router
    print("[3/4] 初始化路由器...")
    if args.router == "selective":
        router = SelectiveEscalationRouter(
            proposal_budget=args.proposal_budget,
            llm_invoke_threshold=args.router_threshold,
            checkpoint_path=args.router_ckpt,
            device=frontend_device,
        )
        print(
            "  → Selective Escalation Router "
            f"(top_r={args.proposal_budget}, invoke_threshold={args.router_threshold}, "
            f"checkpoint={args.router_ckpt or 'none'})"
        )
    elif args.router == "mlp":
        router = MLPRouter(
            checkpoint_path=args.router_ckpt,
            threshold=args.router_threshold,
            device=frontend_device,
        )
        if args.router_ckpt:
            print(f"  → MLP Router (checkpoint: {args.router_ckpt}, threshold: {args.router_threshold})")
        else:
            print(f"  → MLP Router (未训练模式, threshold: {args.router_threshold})")
    else:
        router = HeuristicRouter()
        print(f"  → 启发式 Router (Baseline)")

    # 4. LLM Verifier
    print("[4/4] 初始化 LLM 验证器...")
    if args.no_llm or args.llm_path is None:
        verifier = NoOpVerifier()
        print("  → NoOp Verifier (不使用 LLM，直接选前端最高分候选)")
    else:
        verifier = BaichuanLocalVerifier(
            model_path=args.llm_path,
            adapter_path=args.llm_adapter,
            device=llm_device,
            mode=args.llm_mode,
            targeted_stage=args.targeted_stage,
        )
        print(
            f"  → LLM Verifier (model: {args.llm_path}, mode: {args.llm_mode}, "
            f"targeted_stage: {args.targeted_stage})"
        )

    # 组装 Pipeline
    pipeline = MASCCSCPipeline(
        frontend_model=frontend,
        candidate_generator=candidate_generator,
        router=router,
        verifier=verifier,
        mechanism_inferencer=mechanism_inferencer,
    )
    print("\n[✓] Pipeline 组装完成!\n")
    return pipeline


# ──────────────────────────────────────────────────────────
# 交互模式
# ──────────────────────────────────────────────────────────

def run_interactive(pipeline):
    """交互式纠错：手动输入句子"""
    print("=" * 60)
    print("  MASC-CSC 交互纠错模式")
    print("  输入中文句子进行纠错，输入 'quit' 或 'q' 退出")
    print("=" * 60)

    while True:
        try:
            sentence = input("\n输入句子 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not sentence or sentence.lower() in ("quit", "q", "exit"):
            print("再见!")
            break

        start_time = time.time()
        try:
            result = pipeline.correct(sentence)
            elapsed = time.time() - start_time

            print(f"\n  原  句: {sentence}")
            print(f"  纠  正: {result.text}")

            # 标出改了哪些字
            if sentence != result.text:
                diffs = []
                for i, (s, t) in enumerate(zip(sentence, result.text)):
                    if s != t:
                        diffs.append(f"    位置{i+1}: '{s}' → '{t}'")
                if diffs:
                    print(f"  修改处:")
                    for d in diffs:
                        print(d)
            else:
                print(f"  结  果: 未发现错误")

            print(f"  来  源: {result.selected_source}")
            print(f"  耗  时: {elapsed:.3f}s")
            print(f"  路由说明: {result.reason}")

        except Exception as e:
            print(f"  [ERROR] 纠错失败: {e}")
            import traceback
            traceback.print_exc()


# ──────────────────────────────────────────────────────────
# 评测模式
# ──────────────────────────────────────────────────────────

def load_eval_data(data_path):
    """加载评测数据，返回 [(src, tgt), ...]"""
    pairs = []
    if data_path.endswith(".jsonl"):
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                src = obj.get("source", obj.get("src", "")).strip()
                tgt = obj.get("target", obj.get("tgt", "")).strip()
                if src and tgt:
                    pairs.append((src, tgt))
    elif data_path.endswith(".csv"):
        with open(data_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 2:
                    src, tgt = row[0].strip(), row[1].strip()
                    if src and tgt:
                        pairs.append((src, tgt))
    return pairs


def run_eval(pipeline, args):
    """批量评测模式"""
    if not args.data:
        print("[ERROR] 评测模式必须指定 --data 参数")
        return

    pairs = load_eval_data(args.data)
    print(f"[*] 加载了 {len(pairs)} 条评测数据")

    # 统计指标
    total = len(pairs)
    correct_count = 0       # 系统输出 == 目标
    src_eq_tgt = 0          # 原句本身就是对的
    tp = 0                  # 有错 & 改对了
    fp = 0                  # 没错 & 改错了 / 有错 & 改成其他错的
    fn = 0                  # 有错 & 没改
    tn = 0                  # 没错 & 没改 
    router_invoke_count = 0

    results = []
    start_time = time.time()

    for i, (src, tgt) in enumerate(pairs):
        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{total}")

        try:
            result = pipeline.correct(src)
            pred = result.text
        except Exception as e:
            print(f"  [跳过] 第{i+1}条出错: {e}")
            pred = src

        has_error = (src != tgt)
        system_changed = (pred != src)

        if pred == tgt:
            correct_count += 1

        if not has_error:
            src_eq_tgt += 1
            if system_changed:
                fp += 1  # 误报
            else:
                tn += 1  # 正确保持
        else:
            if pred == tgt:
                tp += 1  # 真正纠对了
            elif system_changed:
                fp += 1  # 改了但改错了
                fn += 1  # 同时算漏报（正确答案没给出）
            else:
                fn += 1  # 有错但没改

        if "mlp_router_high_risk" in str(getattr(result, 'reason', '')) or \
           "Router skipped" not in str(getattr(result, 'reason', '')):
            router_invoke_count += 1

        results.append({
            "source": src,
            "target": tgt,
            "prediction": pred,
            "correct": pred == tgt,
            "reason": getattr(result, 'reason', ''),
        })

    elapsed = time.time() - start_time

    # 计算指标
    accuracy = correct_count / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    print("\n" + "=" * 60)
    print(f"  MASC-CSC 评测结果")
    print("=" * 60)
    print(f"  数据集: {args.data}")
    print(f"  总样本数: {total}")
    print(f"  其中无错句: {src_eq_tgt}, 有错句: {total - src_eq_tgt}")
    print(f"  ─────────────────────────────────")
    print(f"  句级准确率 (Accuracy): {accuracy:.4f} ({correct_count}/{total})")
    print(f"  句级精确率 (Precision): {precision:.4f}")
    print(f"  句级召回率 (Recall):    {recall:.4f}")
    print(f"  句级 F1:               {f1:.4f}")
    print(f"  ─────────────────────────────────")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Router 调用 LLM 次数: {router_invoke_count}/{total}")
    print(f"  总用时: {elapsed:.1f}s ({elapsed/max(total,1):.3f}s/句)")
    print("=" * 60)

    # 保存结果
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n[✓] 详细结果已保存至: {args.output}")

    # 同时保存指标摘要
    summary_path = (args.output or "results/eval") + ".summary.json"
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": args.data,
            "total": total,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "router_invoke_count": router_invoke_count,
            "elapsed_seconds": round(elapsed, 1),
        }, f, ensure_ascii=False, indent=2)
    print(f"[✓] 指标摘要已保存至: {summary_path}")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    args = parse_args()
    pipeline = build_pipeline(args)

    if args.mode == "interactive":
        run_interactive(pipeline)
    elif args.mode == "eval":
        run_eval(pipeline, args)


if __name__ == "__main__":
    main()
