# 2026-04-11 Router 调参工具变更说明

本次修改新增了两份脚本，用于在**不调用 LLM**的前提下，仅依赖 **BERT 前端 + Router** 评估和比较 router 参数。

## 新增文件

- `scripts/evaluate_router_only.py`
  - 作用：加载 frontend checkpoint，跑 `source -> frontend prediction -> router decision`。
  - 输出：
    - `metrics.json`
    - `predictions.jsonl`
    - `missed_bad_cases.jsonl`
    - `false_routes.jsonl`
  - 评估逻辑：
    - 把“frontend 预测错误的样本”视为 router 应该路由到 LLM 的正类。
    - 把“frontend 预测正确的样本”视为 router 应该跳过 LLM 的负类。
    - 输出 router 的 precision / recall / F1 / balanced accuracy / route rate 等指标。
    - 额外输出 `router_quality_score = 0.6 * router_bad_case_f1 + 0.4 * router_balanced_accuracy`，用于后续直接比较参数好坏。

- `scripts/compare_router_metrics.py`
  - 作用：读取两次 `metrics.json`，生成一份 Markdown 对比报告。
  - 主要判断：
    - 默认用 `router_quality_score` 判定新参数是否优于基线。
  - 输出：
    - 一份 markdown 报告，直接标出 `YES/NO`。

## 为什么这样设计

当前需求明确要求：

- 不走 LLM
- 只看 BERT 小模型和 router
- 需要一套可以反复调参数并判断是否变好的方法

在这种约束下，最合理的 router 目标是：

- 尽量把 **frontend 会错的样本** 路由出来
- 尽量把 **frontend 本来就对的样本** 留在本地，不要过度路由

所以这套工具把 router 视作一个二分类器来评估：

- 正类：frontend 错误，应该路由
- 负类：frontend 正确，应该跳过

这比单纯看 `route_rate` 更能反映 router 参数是否合理。

## 建议使用方式

1. 先跑一次基线参数：
   - 得到一份 `metrics.json`
2. 修改 router 参数后再跑一次：
   - 得到第二份 `metrics.json`
3. 用 `scripts/compare_router_metrics.py` 比较：
   - 看 `router_quality_score`
   - 再看 `route_rate / false_route_rate / missed_bad_case_rate`

## 备注

- 本次没有改动 LLM 训练逻辑。
- 本次没有把任何 router 参数搜索逻辑写死成网格搜索；当前实现更适合你手动调一组参数、跑一轮、比较一轮。
