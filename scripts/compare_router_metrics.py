import argparse
import json
from pathlib import Path


HIGHER_IS_BETTER = {
    "router_quality_score": True,
    "router_bad_case_f1": True,
    "router_bad_case_precision": True,
    "router_bad_case_recall": True,
    "router_good_case_skip_rate": True,
    "router_balanced_accuracy": True,
    "route_rate": False,
    "false_route_rate": False,
    "missed_bad_case_rate": False,
}


def parse_args():
    parser = argparse.ArgumentParser(description="比较两次 router-only 评估结果。")
    parser.add_argument("--baseline", type=str, required=True, help="基线 metrics.json")
    parser.add_argument("--candidate", type=str, required=True, help="新参数 metrics.json")
    parser.add_argument("--output", type=str, required=True, help="输出报告路径，建议 .md")
    parser.add_argument("--tolerance", type=float, default=1e-8)
    return parser.parse_args()


def load_metrics(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def delta_line(metric_name: str, baseline_value: float, candidate_value: float) -> str:
    delta = candidate_value - baseline_value
    sign = "+" if delta >= 0 else ""
    direction = "higher_better" if HIGHER_IS_BETTER.get(metric_name, True) else "lower_better"
    return (
        f"| `{metric_name}` | {baseline_value:.6f} | {candidate_value:.6f} | "
        f"{sign}{delta:.6f} | {direction} |"
    )


def main():
    args = parse_args()
    baseline = load_metrics(args.baseline)
    candidate = load_metrics(args.candidate)

    baseline_score = float(baseline["router_quality_score"])
    candidate_score = float(candidate["router_quality_score"])
    improved = candidate_score > baseline_score + args.tolerance

    compare_keys = [
        "router_quality_score",
        "router_bad_case_f1",
        "router_bad_case_precision",
        "router_bad_case_recall",
        "router_good_case_skip_rate",
        "router_balanced_accuracy",
        "route_rate",
        "false_route_rate",
        "missed_bad_case_rate",
    ]

    lines = [
        "# Router 参数对比报告",
        "",
        f"- 基线文件: `{args.baseline}`",
        f"- 新结果文件: `{args.candidate}`",
        f"- 是否变好: `{'YES' if improved else 'NO'}`",
        f"- 基线 router_quality_score: `{baseline_score:.6f}`",
        f"- 新结果 router_quality_score: `{candidate_score:.6f}`",
        "",
        "## 结论",
        "",
    ]

    if improved:
        lines.append(
            "新参数更好。判断标准是 `router_quality_score` 提高；"
            "这个分数由 `0.6 * router_bad_case_f1 + 0.4 * router_balanced_accuracy` 组成。"
        )
    else:
        lines.append(
            "新参数没有优于基线。判断标准仍然是 `router_quality_score`；"
            "如果你想更激进地提高召回，可以接受更高的 `route_rate` 和 `false_route_rate`。"
        )

    lines.extend([
        "",
        "## 指标变化",
        "",
        "| 指标 | Baseline | Candidate | Delta | 方向 |",
        "|---|---:|---:|---:|---|",
    ])

    for key in compare_keys:
        lines.append(delta_line(key, float(baseline[key]), float(candidate[key])))

    lines.extend([
        "",
        "## 参数快照",
        "",
        "### Baseline",
        "",
        "```json",
        json.dumps(baseline.get("router_config", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "### Candidate",
        "",
        "```json",
        json.dumps(candidate.get("router_config", {}), ensure_ascii=False, indent=2),
        "```",
        "",
    ])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\n[*] 对比报告已写入: {output_path}")


if __name__ == "__main__":
    main()
