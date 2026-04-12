from masc_csc.types import ErrorMechanism, PositionPrediction, RouterDecision, SentencePrediction

class RiskAwareRouter:
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
