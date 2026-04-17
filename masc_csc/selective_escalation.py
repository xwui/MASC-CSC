"""Sentence-level selective escalation router."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from masc_csc.mechanism import MechanismInferencer
from masc_csc.types import ErrorMechanism, PositionPrediction, RouterDecision, SentencePrediction
from utils.str_utils import is_chinese

_FEATURE_DIM = 8
_EPS = 1e-8
_TOP_R = 3
_MECHANISM = MechanismInferencer()


@lru_cache(maxsize=8192)
def _phonological_affinity(source_token: str, candidate_token: str) -> float:
    if source_token == candidate_token:
        return 1.0
    if _MECHANISM.is_phonological_match(source_token, candidate_token):
        return 1.0
    if _MECHANISM.is_fuzzy_phonological_match(source_token, candidate_token):
        return 0.5
    return 0.0


@lru_cache(maxsize=8192)
def _glyph_affinity(source_token: str, candidate_token: str) -> float:
    if source_token == candidate_token:
        return 1.0
    if not is_chinese(source_token) or not is_chinese(candidate_token):
        return 0.0
    similarity = _MECHANISM.glyph_similarity(source_token, candidate_token)
    return max(0.0, min(1.0, (similarity + 1.0) / 2.0))


def _safe_mean(values: Sequence[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _normalize_scores(scores: Sequence[float]) -> torch.Tensor:
    tensor = torch.tensor(list(scores), dtype=torch.float32)
    tensor = tensor.clamp_min(0.0)
    total = float(tensor.sum().item())
    if total <= _EPS:
        return torch.full((len(tensor),), 1.0 / max(len(tensor), 1), dtype=torch.float32)
    return tensor / total


def _js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    p = p.clamp_min(_EPS)
    q = q.clamp_min(_EPS)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * torch.log2(p / m))
    kl_qm = torch.sum(q * torch.log2(q / m))
    return float(0.5 * (kl_pm + kl_qm).item())


def _candidate_bank(position: PositionPrediction) -> List[str]:
    candidates = []
    seen = set()
    for alternative in position.alternatives:
        if alternative.token in seen:
            continue
        seen.add(alternative.token)
        candidates.append(alternative.token)
    if position.source_token not in seen:
        candidates.append(position.source_token)
    return candidates


def _semantic_distribution(position: PositionPrediction, candidates: Sequence[str]) -> torch.Tensor:
    score_map = {}
    for alternative in position.alternatives:
        score_map[alternative.token] = max(float(alternative.score), score_map.get(alternative.token, 0.0))

    source_prior = max(0.0, 1.0 - float(position.detection_score))
    score_map[position.source_token] = max(score_map.get(position.source_token, 0.0), source_prior)
    scores = [score_map.get(candidate, 0.0) for candidate in candidates]
    return _normalize_scores(scores)


def _phonological_distribution(source_token: str, candidates: Sequence[str]) -> torch.Tensor:
    scores = [_phonological_affinity(source_token, candidate) for candidate in candidates]
    return _normalize_scores(scores)


def _glyph_distribution(source_token: str, candidates: Sequence[str]) -> torch.Tensor:
    scores = [_glyph_affinity(source_token, candidate) for candidate in candidates]
    return _normalize_scores(scores)


def _position_conflict(position: PositionPrediction) -> float:
    candidates = _candidate_bank(position)
    if len(candidates) <= 1:
        return 0.0

    fused = _semantic_distribution(position, candidates)
    phonological = _phonological_distribution(position.source_token, candidates)
    glyph = _glyph_distribution(position.source_token, candidates)

    return (
        _js_divergence(fused, phonological)
        + _js_divergence(fused, glyph)
        + _js_divergence(phonological, glyph)
    ) / 3.0


def _mean_top(values: Sequence[float], top_r: int) -> float:
    ranked = sorted((float(value) for value in values), reverse=True)
    if not ranked:
        return 0.0
    return _safe_mean(ranked[:max(1, top_r)])


def multimodal_conflict_score(prediction: SentencePrediction, top_r: int = _TOP_R) -> float:
    positions = sorted(
        prediction.positions,
        key=lambda position: position.detection_score,
        reverse=True,
    )[:max(1, top_r)]
    if not positions:
        return 0.0
    return _safe_mean(_position_conflict(position) for position in positions)


def extract_selective_features(
    prediction: SentencePrediction,
    top_r: int = _TOP_R,
) -> torch.Tensor:
    positions = prediction.positions
    if not positions:
        return torch.zeros(_FEATURE_DIM, dtype=torch.float32)

    detection_scores = [position.detection_score for position in positions]
    uncertainties = [position.uncertainty for position in positions]
    margins = [position.margin for position in positions]
    n = len(positions)

    edited_ratio = sum(1 for position in positions if position.is_edited) / n
    uncertain_mechanism_ratio = sum(
        1 for position in positions if position.mechanism == ErrorMechanism.UNCERTAIN
    ) / n

    features = torch.tensor([
        max(detection_scores),
        max(uncertainties),
        min(margins),
        edited_ratio,
        uncertain_mechanism_ratio,
        multimodal_conflict_score(prediction, top_r=top_r),
        _mean_top(uncertainties, top_r=top_r),
        _mean_top(detection_scores, top_r=top_r),
    ], dtype=torch.float32)
    return features


class SelectiveRouterMLP(nn.Module):
    """Lightweight sentence-level invoke head."""

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

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return logits.squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))


class SelectiveEscalationRouter:
    """Sentence-level gate for deciding whether LLM escalation is worthwhile."""

    def __init__(
        self,
        proposal_detection_threshold: float = 0.08,
        proposal_uncertainty_threshold: float = 0.80,
        proposal_margin_threshold: float = 0.20,
        proposal_budget: int = _TOP_R,
        llm_invoke_threshold: Optional[float] = None,
        checkpoint_path: Optional[str] = None,
        device: str = 'cpu',
        uncertainty_floor: float = 0.60,
        uncertainty_ceiling: float = 1.80,
        margin_floor: float = 0.05,
        margin_ceiling: float = 0.35,
    ):
        del proposal_detection_threshold, proposal_uncertainty_threshold, proposal_margin_threshold
        self.top_r = max(1, proposal_budget)
        self.llm_invoke_threshold = 0.45 if llm_invoke_threshold is None else llm_invoke_threshold
        self.device = device
        self.uncertainty_floor = uncertainty_floor
        self.uncertainty_ceiling = max(uncertainty_ceiling, uncertainty_floor + 1e-6)
        self.margin_floor = margin_floor
        self.margin_ceiling = max(margin_ceiling, margin_floor + 1e-6)

        self.invoke_head: Optional[SelectiveRouterMLP] = None
        if checkpoint_path is not None:
            payload = torch.load(checkpoint_path, map_location='cpu')
            state_dict = payload.get('state_dict', payload) if isinstance(payload, dict) else payload
            self.invoke_head = SelectiveRouterMLP()
            self.invoke_head.load_state_dict(state_dict)
            self.invoke_head.to(device)
            self.invoke_head.eval()
            if llm_invoke_threshold is None and isinstance(payload, dict) and 'threshold' in payload:
                self.llm_invoke_threshold = float(payload['threshold'])

    @staticmethod
    def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
        return max(lower, min(upper, value))

    def _normalize_high(self, value: float, floor: float, ceiling: float) -> float:
        if value <= floor:
            return 0.0
        if value >= ceiling:
            return 1.0
        return (value - floor) / (ceiling - floor)

    def _normalize_low(self, value: float, floor: float, ceiling: float) -> float:
        if value <= floor:
            return 1.0
        if value >= ceiling:
            return 0.0
        return (ceiling - value) / (ceiling - floor)

    def _heuristic_sentence_risk(self, features: torch.Tensor) -> float:
        (
            max_detection,
            max_uncertainty,
            min_margin,
            edited_ratio,
            uncertain_mechanism_ratio,
            conflict_score,
            mean_top_uncertainty,
            mean_top_detection,
        ) = features.tolist()

        normalized_uncertainty = self._normalize_high(
            max_uncertainty,
            self.uncertainty_floor,
            self.uncertainty_ceiling,
        )
        normalized_top_uncertainty = self._normalize_high(
            mean_top_uncertainty,
            self.uncertainty_floor,
            self.uncertainty_ceiling,
        )
        low_margin = self._normalize_low(
            min_margin,
            self.margin_floor,
            self.margin_ceiling,
        )

        score = _safe_mean([
            max_detection,
            normalized_uncertainty,
            low_margin,
            edited_ratio,
            uncertain_mechanism_ratio,
            conflict_score,
            normalized_top_uncertainty,
            mean_top_detection,
        ])
        return self._clip(score)

    def decide(self, prediction: SentencePrediction) -> RouterDecision:
        if not prediction.positions:
            return RouterDecision(
                invoke_llm=False,
                risk_score=0.0,
                reasons=['empty_prediction'],
            )

        features = extract_selective_features(prediction, top_r=self.top_r)
        feature_values = features.tolist()

        reasons = [
            f'max_detection={feature_values[0]:.3f}',
            f'max_uncertainty={feature_values[1]:.3f}',
            f'min_margin={feature_values[2]:.3f}',
            f'edited_ratio={feature_values[3]:.3f}',
            f'uncertain_mechanism_ratio={feature_values[4]:.3f}',
            f'multimodal_conflict_score={feature_values[5]:.3f}',
            f'mean_top{self.top_r}_uncertainty={feature_values[6]:.3f}',
            f'mean_top{self.top_r}_detection={feature_values[7]:.3f}',
        ]

        if self.invoke_head is not None:
            with torch.no_grad():
                sentence_risk = float(self.invoke_head(features.unsqueeze(0).to(self.device)).item())
            reasons.append('sentence_gate=learned_mlp')
        else:
            sentence_risk = self._heuristic_sentence_risk(features)
            reasons.append('sentence_gate=heuristic_fallback')

        invoke_llm = sentence_risk >= self.llm_invoke_threshold
        reasons.append('invoke_llm' if invoke_llm else 'skip_llm')

        return RouterDecision(
            invoke_llm=invoke_llm,
            risk_score=sentence_risk,
            use_supplement_mode=False,
            supplement_positions=[],
            reasons=reasons,
        )
