from typing import Iterable

import torch
import torch.nn.functional as F

from masc_csc.types import ErrorMechanism, TokenAlternative
from utils.str_utils import is_chinese
from utils.utils import convert_char_to_image, convert_char_to_pinyin


class MechanismInferencer:
    def __init__(self, glyph_threshold: float = 0.75):
        self.glyph_threshold = glyph_threshold

    @staticmethod
    def _normalized_pinyin(character: str):
        return tuple(convert_char_to_pinyin(character).tolist())

    def is_phonological_match(self, source_token: str, candidate_token: str) -> bool:
        if not is_chinese(source_token) or not is_chinese(candidate_token):
            return False
        return self._normalized_pinyin(source_token) == self._normalized_pinyin(candidate_token)

    def glyph_similarity(self, source_token: str, candidate_token: str) -> float:
        if not is_chinese(source_token) or not is_chinese(candidate_token):
            return 0.0

        source_image = convert_char_to_image(source_token).reshape(1, -1).float()
        candidate_image = convert_char_to_image(candidate_token).reshape(1, -1).float()
        similarity = F.cosine_similarity(source_image, candidate_image).item()
        return float(similarity)

    def is_visual_match(self, source_token: str, candidate_token: str) -> bool:
        return self.glyph_similarity(source_token, candidate_token) >= self.glyph_threshold

    def infer_from_alternatives(
            self,
            source_token: str,
            alternatives: Iterable[TokenAlternative],
    ) -> ErrorMechanism:
        phonological_votes = 0
        visual_votes = 0

        for alternative in alternatives:
            if alternative.token == source_token:
                continue
            if self.is_phonological_match(source_token, alternative.token):
                phonological_votes += 1
            if self.is_visual_match(source_token, alternative.token):
                visual_votes += 1

        if phonological_votes > visual_votes and phonological_votes > 0:
            return ErrorMechanism.PHONOLOGICAL
        if visual_votes > phonological_votes and visual_votes > 0:
            return ErrorMechanism.VISUAL
        return ErrorMechanism.UNCERTAIN
