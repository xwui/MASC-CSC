from dataclasses import dataclass, field
from enum import Enum
from typing import List


class ErrorMechanism(str, Enum):
    PHONOLOGICAL = "phonological-dominant"
    VISUAL = "visual-dominant"
    UNCERTAIN = "uncertain-or-other"


@dataclass
class TokenAlternative:
    token: str
    token_id: int
    score: float


@dataclass
class PositionPrediction:
    index: int
    source_token: str
    predicted_token: str
    detection_score: float
    uncertainty: float
    mechanism: ErrorMechanism
    alternatives: List[TokenAlternative] = field(default_factory=list)

    @property
    def is_edited(self) -> bool:
        return self.source_token != self.predicted_token

    @property
    def margin(self) -> float:
        if len(self.alternatives) < 2:
            return 1.0
        return self.alternatives[0].score - self.alternatives[1].score


@dataclass
class SentencePrediction:
    source_text: str
    predicted_text: str
    positions: List[PositionPrediction]

    @property
    def edited_positions(self) -> List[PositionPrediction]:
        return [position for position in self.positions if position.is_edited]

    @property
    def mean_uncertainty(self) -> float:
        if not self.positions:
            return 0.0
        return sum(position.uncertainty for position in self.positions) / len(self.positions)


@dataclass
class CandidateSentence:
    text: str
    edited_indices: List[int]
    score: float
    source: str
    mechanism_info: dict = field(default_factory=dict)  # {position_idx: ErrorMechanism}


@dataclass
class RouterDecision:
    invoke_llm: bool
    risk_score: float
    use_supplement_mode: bool = False
    supplement_positions: List[int] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


@dataclass
class VerificationResult:
    text: str
    selected_source: str
    reason: str
    candidates: List[CandidateSentence] = field(default_factory=list)
