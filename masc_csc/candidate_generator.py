import itertools
from typing import Dict, List

from masc_csc.mechanism import MechanismInferencer
from masc_csc.types import CandidateSentence, ErrorMechanism, SentencePrediction, TokenAlternative


class MechanismAwareCandidateGenerator:
    def __init__(
            self,
            mechanism_inferencer: MechanismInferencer,
            max_positions: int = 2,
            max_candidates: int = 5,
            max_alternatives_per_position: int = 3,
    ):
        self.mechanism_inferencer = mechanism_inferencer
        self.max_positions = max_positions
        self.max_candidates = max_candidates
        self.max_alternatives_per_position = max_alternatives_per_position

    def _filter_alternatives(
            self,
            source_token: str,
            mechanism: ErrorMechanism,
            alternatives: List[TokenAlternative],
    ) -> List[TokenAlternative]:
        filtered = []
        for alternative in alternatives:
            if alternative.token == source_token:
                filtered.append(alternative)
                continue

            if mechanism == ErrorMechanism.PHONOLOGICAL:
                if self.mechanism_inferencer.is_fuzzy_phonological_match(source_token, alternative.token):
                    filtered.append(alternative)
            elif mechanism == ErrorMechanism.VISUAL:
                if self.mechanism_inferencer.is_visual_match(source_token, alternative.token):
                    filtered.append(alternative)
            else:
                filtered.append(alternative)

        deduplicated: Dict[str, TokenAlternative] = {}
        for alternative in filtered:
            if alternative.token not in deduplicated:
                deduplicated[alternative.token] = alternative

        return list(deduplicated.values())[:self.max_alternatives_per_position]

    @staticmethod
    def _candidate_score(
            alternatives: List[TokenAlternative],
            detection_scores: List[float] = None,
    ) -> float:
        """检测置信度加权的候选评分：高检测分位置的替换对总分贡献更大"""
        if not alternatives:
            return 0.0
        if detection_scores and len(detection_scores) == len(alternatives):
            total_weight = sum(detection_scores) + 1e-8
            weighted = sum(alt.score * w for alt, w in zip(alternatives, detection_scores))
            return float(weighted / total_weight)
        return float(sum(item.score for item in alternatives) / len(alternatives))

    def generate(self, prediction: SentencePrediction) -> List[CandidateSentence]:
        source_tokens = list(prediction.source_text)
        candidates = {
            prediction.source_text: CandidateSentence(
                text=prediction.source_text,
                edited_indices=[],
                score=1.0,
                source="original",
            )
        }

        risky_positions = sorted(
            prediction.positions,
            key=lambda position: (position.detection_score, position.uncertainty),
            reverse=True,
        )[:self.max_positions]

        option_bank = []
        for position in risky_positions:
            filtered_alternatives = self._filter_alternatives(
                source_token=position.source_token,
                mechanism=position.mechanism,
                alternatives=position.alternatives,
            )
            if not filtered_alternatives:
                filtered_alternatives = [
                    TokenAlternative(
                        token=position.source_token,
                        token_id=-1,
                        score=1.0 - position.detection_score,
                    )
                ]
            option_bank.append((position.index, filtered_alternatives))

        # 收集各位置的检测分数，用于加权评分
        position_detection_scores = {pos.index: pos.detection_score for pos in risky_positions}
        # 收集各位置的错误机制，用于传递给 LLM Prompt
        position_mechanisms = {pos.index: pos.mechanism for pos in risky_positions}

        for combination in itertools.product(*[item[1] for item in option_bank] or [[None]]):
            candidate_tokens = source_tokens.copy()
            edited_indices = []
            selected_alternatives = []
            det_scores_for_combination = []

            for (position_index, _), alternative in zip(option_bank, combination):
                if alternative is None:
                    continue
                candidate_tokens[position_index] = alternative.token
                selected_alternatives.append(alternative)
                det_scores_for_combination.append(position_detection_scores.get(position_index, 0.5))
                if alternative.token != source_tokens[position_index]:
                    edited_indices.append(position_index)

            candidate_text = ''.join(candidate_tokens)
            # 构建该候选对应的机制信息
            mech_info = {idx: position_mechanisms.get(idx, ErrorMechanism.UNCERTAIN)
                         for idx in edited_indices}
            candidates[candidate_text] = CandidateSentence(
                text=candidate_text,
                edited_indices=edited_indices,
                score=self._candidate_score(selected_alternatives, det_scores_for_combination),
                source="masc-generator",
                mechanism_info=mech_info,
            )

        predicted_text = prediction.predicted_text
        candidates[predicted_text] = CandidateSentence(
            text=predicted_text,
            edited_indices=[position.index for position in prediction.edited_positions],
            score=1.0 - prediction.mean_uncertainty,
            source="front-end-top1",
        )

        ranked_candidates = sorted(
            candidates.values(),
            key=lambda candidate: (len(candidate.edited_indices) == 0, candidate.score),
            reverse=True,
        )
        return ranked_candidates[:self.max_candidates]
