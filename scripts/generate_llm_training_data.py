"""
MASC-CSC LLM 训练数据生成脚本
================================
通过运行前端模型的完整管线 (Frontend → Mechanism → Generator → Router)，
收集 Router 判定为高风险的样本，格式化为 LLM 微调所需的 JSONL 训练数据。

用法：
    python scripts/generate_llm_training_data.py \
        --ckpt-path ./ckpt/pretrained.ckpt \
        --data train.csv \
        --output ./data/llm_train_data.jsonl \
        --risk-threshold 1.5
"""

import argparse
import json
import string
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from masc_csc import (  # noqa: E402
    MechanismAwareCandidateGenerator,
    MechanismInferencer,
    RiskAwareRouter,
)
from masc_csc.types import ErrorMechanism  # noqa: E402
from models.multimodal_frontend import MultimodalCSCFrontend  # noqa: E402
from masc_csc.pipeline import MASCCSCPipeline  # noqa: E402
from masc_csc.llm_verifier import NoOpVerifier  # noqa: E402
from utils.dataset import CSCDataset  # noqa: E402

MECHANISM_LABEL_ZH = {
    ErrorMechanism.PHONOLOGICAL: "音近错误（读音相似）",
    ErrorMechanism.VISUAL: "形近错误（字形相似）",
    ErrorMechanism.UNCERTAIN: "不确定",
}


def load_frontend_model(checkpoint_path: str, device: str):
    args = SimpleNamespace(device=device, hyper_params={})
    model = MultimodalCSCFrontend(args)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        state_dict = state.get("state_dict", state)
        model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def find_correct_choice(candidates, target_text: str) -> int:
    """在候选中找到与真实标签完全匹配的选项索引"""
    for i, candidate in enumerate(candidates):
        if candidate.text == target_text:
            return i
    # 如果没有完全匹配，找编辑距离最小的
    min_dist = float('inf')
    best_idx = 0
    for i, candidate in enumerate(candidates):
        dist = sum(1 for a, b in zip(candidate.text, target_text) if a != b)
        dist += abs(len(candidate.text) - len(target_text))
        if dist < min_dist:
            min_dist = dist
            best_idx = i
    return best_idx


def build_training_sample(
        source_text: str,
        prediction,
        candidates,
        correct_index: int,
) -> dict:
    """构建一条 LLM SFT 训练样本"""
    source_chars = list(source_text)

    # 错误位置分析
    error_lines = []
    for position in prediction.edited_positions:
        idx = position.index
        char = source_chars[idx] if idx < len(source_chars) else "?"
        mech_label = MECHANISM_LABEL_ZH.get(position.mechanism, "不确定")
        error_lines.append(f"  - 位置{idx + 1} (\"{char}\")：{mech_label}")

    error_section = ""
    if error_lines:
        error_section = "检测到的潜在错误：\n" + "\n".join(error_lines) + "\n\n"

    # 候选列表
    candidate_lines = []
    for index, candidate in enumerate(candidates):
        label = string.ascii_uppercase[index]
        if not candidate.edited_indices:
            annotation = "（保持原句不变）"
        else:
            edited_pos_str = ", ".join(str(i + 1) for i in candidate.edited_indices)
            annotation = f" [修改位置: {edited_pos_str}]"
        candidate_lines.append(f"{label}. {candidate.text}{annotation}")

    input_text = (
        f"源句子：\"{source_text}\"\n"
        f"{error_section}"
        f"候选修正方案：\n"
        f"{chr(10).join(candidate_lines)}\n\n"
        f"请从以上候选方案中选择最合适的修正结果。\n"
        f"严格只输出一个字母选项（如 A、B、C），不要输出其他任何内容。\n"
        f"选项："
    )

    correct_label = string.ascii_uppercase[correct_index]

    return {
        "instruction": "请从以下候选中选择最正确的拼写纠正结果，只输出字母。",
        "input": input_text,
        "output": correct_label,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate LLM training data from MASC-CSC pipeline.")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Frontend model checkpoint path.")
    parser.add_argument("--data", type=str, required=True, help="Training CSV file (src,tgt format).")
    parser.add_argument("--output", type=str, default="./data/llm_train_data.jsonl", help="Output JSONL path.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--risk-threshold", type=float, default=1.5,
                        help="Only include samples with risk_score >= this threshold.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--max-positions", type=int, default=2)
    parser.add_argument("--limit", type=int, default=-1, help="Max samples to process (-1 for all).")
    args = parser.parse_args()

    # 加载模型和管线
    print("Loading frontend model...")
    frontend_model = load_frontend_model(args.ckpt_path, args.device)
    mechanism_inferencer = MechanismInferencer()
    candidate_generator = MechanismAwareCandidateGenerator(
        mechanism_inferencer=mechanism_inferencer,
        max_positions=args.max_positions,
        max_candidates=args.max_candidates,
    )
    router = RiskAwareRouter()
    pipeline = MASCCSCPipeline(
        frontend_model=frontend_model,
        candidate_generator=candidate_generator,
        router=router,
        verifier=NoOpVerifier(),
        mechanism_inferencer=mechanism_inferencer,
    )

    # 加载数据集
    dataset = CSCDataset(args.data)
    data = dataset.data
    if args.limit > 0:
        data = data[:args.limit]
    print(f"Processing {len(data)} samples...")

    # 生成训练数据
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    saved = 0
    skipped_low_risk = 0
    skipped_no_match = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for src, tgt in tqdm(data, desc="Generating LLM training data"):
            total += 1
            try:
                # 跑前端管线
                prediction = pipeline.analyze(src, top_k=args.top_k)
                candidates = candidate_generator.generate(prediction)
                routing = router.decide(prediction)

                # 只保留高风险样本
                if routing.risk_score < args.risk_threshold:
                    skipped_low_risk += 1
                    continue

                # 找到正确答案对应的候选索引
                correct_index = find_correct_choice(candidates, tgt)

                # 构建训练样本
                sample = build_training_sample(src, prediction, candidates, correct_index)
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                saved += 1

            except Exception as e:
                print(f"Error processing '{src}': {e}")
                continue

    print(f"\n=== Data Generation Complete ===")
    print(f"Total processed:    {total}")
    print(f"Saved (high risk):  {saved}")
    print(f"Skipped (low risk): {skipped_low_risk}")
    print(f"Output file:        {output_path}")


if __name__ == "__main__":
    main()
