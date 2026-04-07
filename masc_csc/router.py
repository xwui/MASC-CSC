from masc_csc.types import RouterDecision, SentencePrediction


class RiskAwareRouter:
    def __init__(
            self,
            detection_threshold: float = 0.35,
            uncertainty_threshold: float = 1.5,
            margin_threshold: float = 0.20,
            multi_edit_threshold: int = 2,
    ):
        self.detection_threshold = detection_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.margin_threshold = margin_threshold
        self.multi_edit_threshold = multi_edit_threshold

    def decide(self, prediction: SentencePrediction) -> RouterDecision:
        risk_score = 0.0
        reasons = []

        high_detection_positions = [
            position for position in prediction.positions
            if position.detection_score >= self.detection_threshold
        ]
        if high_detection_positions:
            risk_score += 1.0
            reasons.append("high_detection")

        if prediction.mean_uncertainty >= self.uncertainty_threshold:
            risk_score += 1.0
            reasons.append("high_uncertainty")

        if len(prediction.edited_positions) >= self.multi_edit_threshold:
            risk_score += 1.0
            reasons.append("multiple_edits")

        small_margin_positions = [
            position for position in prediction.edited_positions
            if position.margin <= self.margin_threshold
        ]
        if small_margin_positions:
            risk_score += 1.0
            reasons.append("small_margin")

        mechanism_sensitive = [
            position for position in prediction.edited_positions
            if position.mechanism.value != "uncertain-or-other"
        ]
        if mechanism_sensitive:
            risk_score += 0.5
            reasons.append("mechanism_sensitive")

        return RouterDecision(
            invoke_llm=risk_score >= 1.5,
            risk_score=risk_score,
            reasons=reasons,
        )
