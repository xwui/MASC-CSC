"""
MASC-CSC Router
================
路由器模块：决定是否需要调用 LLM 进行验证。

包含两种实现：
- HeuristicRouter: 基于人工阈值的启发式路由（原始版本，Baseline）
- MLPRouter: 基于可学习 MLP 的自适应路由（改进版本）
"""

from typing import List, Optional

import torch
import torch.nn as nn

from masc_csc.types import ErrorMechanism, PositionPrediction, RouterDecision, SentencePrediction


# ──────────────────────────────────────────────────────────
# 1. 启发式路由器（原始版本，保留用于 Baseline 对比）
# ──────────────────────────────────────────────────────────

class HeuristicRouter:
    """
    基于人工阈值的启发式路由器。
    通过位置级风险评分 → 句子级聚合 → 阈值截断来决定是否调用 LLM。
    保留原始逻辑不变，作为消融实验的 Baseline。
    """

    def __init__(
            self,
            detection_threshold: float = 0.35,
            uncertainty_threshold: float = 1.5,
            margin_threshold: float = 0.20,
            multi_edit_threshold: int = 2,
            risk_threshold: float = 0.50,
            detection_floor: float = 0.15,
            detection_ceiling: float = 0.60,
            uncertainty_floor: float = 0.80,
            uncertainty_ceiling: float = 1.80,
    ):
        # 兼容你原来已有的 4 个阈值参数
        self.detection_threshold = detection_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.margin_threshold = margin_threshold
        self.multi_edit_threshold = max(1, multi_edit_threshold)

        # 最终二分阈值：risk_score >= risk_threshold 就调用 LLM
        self.risk_threshold = risk_threshold

        # 连续化打分区间
        self.detection_floor = detection_floor
        self.detection_ceiling = max(detection_ceiling, detection_floor + 1e-6)
        self.uncertainty_floor = uncertainty_floor
        self.uncertainty_ceiling = max(uncertainty_ceiling, uncertainty_floor + 1e-6)

        # 位置级风险的权重
        self.weight_uncertainty = 0.40
        self.weight_low_margin = 0.30
        self.weight_detection = 0.20
        self.weight_edited = 0.05
        self.weight_mechanism_uncertain = 0.05

        # 句子级风险的权重
        self.sentence_weight_max = 0.55
        self.sentence_weight_top2 = 0.25
        self.sentence_weight_edit_count = 0.20

    @staticmethod
    def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def _safe_mean(values) -> float:
        values = list(values)
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _normalize_high(self, value: float, floor: float, ceiling: float) -> float:
        if value <= floor:
            return 0.0
        if value >= ceiling:
            return 1.0
        return (value - floor) / (ceiling - floor)

    def _normalize_low(self, value: float, floor: float, ceiling: float) -> float:
        # value 越小，风险越高
        if value <= floor:
            return 1.0
        if value >= ceiling:
            return 0.0
        return (ceiling - value) / (ceiling - floor)

    def _position_risk(self, position: PositionPrediction) -> float:
        detection_term = self._normalize_high(
            position.detection_score,
            self.detection_floor,
            self.detection_ceiling,
        )
        uncertainty_term = self._normalize_high(
            position.uncertainty,
            self.uncertainty_floor,
            self.uncertainty_ceiling,
        )
        low_margin_term = self._normalize_low(
            position.margin,
            0.05,
            max(self.margin_threshold, 0.05 + 1e-6),
        )
        edited_term = 1.0 if position.is_edited else 0.0
        mechanism_uncertain_term = 1.0 if position.mechanism == ErrorMechanism.UNCERTAIN else 0.0

        weighted_sum = (
            self.weight_uncertainty * uncertainty_term
            + self.weight_low_margin * low_margin_term
            + self.weight_detection * detection_term
            + self.weight_edited * edited_term
            + self.weight_mechanism_uncertain * mechanism_uncertain_term
        )
        total_weight = (
            self.weight_uncertainty
            + self.weight_low_margin
            + self.weight_detection
            + self.weight_edited
            + self.weight_mechanism_uncertain
        )
        return self._clip(weighted_sum / total_weight)

    def _collect_suspicious_positions(self, prediction: SentencePrediction):
        suspicious_positions = []
        for position in prediction.positions:
            if (
                position.is_edited
                or position.detection_score >= self.detection_floor
                or position.uncertainty >= self.uncertainty_floor
            ):
                suspicious_positions.append(position)
        return suspicious_positions

    def decide(self, prediction: SentencePrediction) -> RouterDecision:
        edited_positions = prediction.edited_positions
        suspicious_positions = self._collect_suspicious_positions(prediction)

        if not suspicious_positions:
            return RouterDecision(
                invoke_llm=False,
                risk_score=0.0,
                reasons=["no_suspicious_positions"],
            )

        scored_positions = [
            (position, self._position_risk(position))
            for position in suspicious_positions
        ]
        scored_positions.sort(key=lambda item: item[1], reverse=True)

        position_risks = [risk for _, risk in scored_positions]
        max_risk = position_risks[0]
        mean_top2 = self._safe_mean(position_risks[:2])
        edit_count_term = self._clip(len(edited_positions) / float(self.multi_edit_threshold + 1))

        sentence_risk = self._clip(
            self.sentence_weight_max * max_risk
            + self.sentence_weight_top2 * mean_top2
            + self.sentence_weight_edit_count * edit_count_term
        )

        reasons = []

        if max_risk >= 0.70:
            reasons.append(f"high_position_risk(max={max_risk:.2f})")
        elif max_risk >= 0.50:
            reasons.append(f"moderate_position_risk(max={max_risk:.2f})")

        if len(scored_positions) >= 2 and mean_top2 >= 0.50:
            reasons.append(f"multiple_risky_positions(mean_top2={mean_top2:.2f})")

        if len(edited_positions) >= self.multi_edit_threshold:
            reasons.append(f"multiple_edits(count={len(edited_positions)})")

        if any(position.margin <= self.margin_threshold for position in edited_positions):
            reasons.append("small_margin")

        if any(position.uncertainty >= self.uncertainty_threshold for position in suspicious_positions):
            reasons.append("high_uncertainty")

        if any(position.detection_score >= self.detection_threshold for position in suspicious_positions):
            reasons.append("high_detection")

        if any(position.mechanism == ErrorMechanism.UNCERTAIN for position in suspicious_positions):
            reasons.append("mechanism_uncertain")

        if not reasons:
            reasons.append("low_to_moderate_risk")

        return RouterDecision(
            invoke_llm=sentence_risk >= self.risk_threshold,
            risk_score=sentence_risk,
            reasons=reasons,
        )


# ──────────────────────────────────────────────────────────
# 2. 可学习 MLP 路由器（改进版本）
# ──────────────────────────────────────────────────────────

# 特征维度定义
_FEATURE_DIM = 10


def extract_features(prediction: SentencePrediction) -> torch.Tensor:
    """
    从 SentencePrediction 中提取固定维度的特征向量，供 MLP 路由器使用。

    提取的 10 维特征：
    [0] max_detection_score   - 所有位置中最大的检测分
    [1] mean_detection_score  - 所有位置检测分的均值
    [2] max_uncertainty       - 所有位置中最大的不确定度
    [3] mean_uncertainty      - 所有位置不确定度的均值
    [4] min_margin            - 所有位置中最小的 margin（top1-top2 概率差）
    [5] num_edited_ratio      - 被编辑位置数 / 总位置数
    [6] num_uncertain_mech    - 机制为 UNCERTAIN 的位置占比
    [7] num_phono_mech        - 机制为 PHONOLOGICAL 的位置占比
    [8] num_visual_mech       - 机制为 VISUAL 的位置占比
    [9] sentence_length_norm  - 句子长度归一化 (len / 128)
    """
    positions = prediction.positions
    if not positions:
        return torch.zeros(_FEATURE_DIM)

    detection_scores = [p.detection_score for p in positions]
    uncertainties = [p.uncertainty for p in positions]
    margins = [p.margin for p in positions]
    n = len(positions)

    num_edited = sum(1 for p in positions if p.is_edited)
    num_uncertain = sum(1 for p in positions if p.mechanism == ErrorMechanism.UNCERTAIN)
    num_phono = sum(1 for p in positions if p.mechanism == ErrorMechanism.PHONOLOGICAL)
    num_visual = sum(1 for p in positions if p.mechanism == ErrorMechanism.VISUAL)

    features = torch.tensor([
        max(detection_scores),                          # [0]
        sum(detection_scores) / n,                      # [1]
        max(uncertainties),                             # [2]
        sum(uncertainties) / n,                         # [3]
        min(margins),                                   # [4]
        num_edited / n,                                 # [5]
        num_uncertain / n,                              # [6]
        num_phono / n,                                  # [7]
        num_visual / n,                                 # [8]
        min(n / 128.0, 1.0),                            # [9]
    ], dtype=torch.float32)

    return features


class RouterMLP(nn.Module):
    """
    轻量级 MLP 二分类网络，输入特征向量，输出调用 LLM 的概率。

    结构：10 → 32 → 16 → 1
    参数量：约 900，极其轻量。
    """

    def __init__(self, input_dim: int = _FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回调用 LLM 的概率 (sigmoid 后的标量)"""
        return torch.sigmoid(self.net(x))


class MLPRouter:
    """
    基于可学习 MLP 的自适应路由器。

    使用训练好的 RouterMLP 网络替代人工阈值，
    从 SentencePrediction 中提取特征，通过 MLP 预测是否需要 LLM 验证。

    如果没有提供预训练权重，会降级为使用默认阈值 0.5 的启发式判断。

    用法:
        # 使用预训练权重
        router = MLPRouter(checkpoint_path="./ckpt/router_mlp.pt")

        # 未训练时（降级模式）
        router = MLPRouter()
    """

    def __init__(
            self,
            checkpoint_path: Optional[str] = None,
            threshold: float = 0.5,
            device: str = "cpu",
    ):
        self.threshold = threshold
        self.device = device
        self.mlp = RouterMLP()

        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.mlp.load_state_dict(state_dict)

        self.mlp.to(device)
        self.mlp.eval()

    def decide(self, prediction: SentencePrediction) -> RouterDecision:
        features = extract_features(prediction).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prob = self.mlp(features).item()

        invoke_llm = prob >= self.threshold

        reasons = []
        if invoke_llm:
            reasons.append(f"mlp_router_high_risk(prob={prob:.3f})")
        else:
            reasons.append(f"mlp_router_low_risk(prob={prob:.3f})")

        return RouterDecision(
            invoke_llm=invoke_llm,
            risk_score=prob,
            reasons=reasons,
        )


# ──────────────────────────────────────────────────────────
# 3. 向后兼容：保留 RiskAwareRouter 别名
# ──────────────────────────────────────────────────────────

# 默认使用启发式路由器，保持现有代码不被破坏
RiskAwareRouter = HeuristicRouter
