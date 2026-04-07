from typing import Iterable

import pypinyin
import torch
import torch.nn.functional as F

from masc_csc.types import ErrorMechanism, TokenAlternative
from utils.str_utils import is_chinese
from utils.utils import convert_char_to_image, convert_char_to_pinyin

# ---- 模糊拼音混淆组 ----
# 中文母语者常见声母混淆
CONFUSED_INITIALS = [
    frozenset({'zh', 'z'}), frozenset({'ch', 'c'}), frozenset({'sh', 's'}),
    frozenset({'n', 'l'}), frozenset({'r', 'l'}), frozenset({'f', 'h'}),
]
# 中文母语者常见韵母混淆
CONFUSED_FINALS = [
    frozenset({'an', 'ang'}), frozenset({'en', 'eng'}), frozenset({'in', 'ing'}),
    frozenset({'ian', 'iang'}), frozenset({'uan', 'uang'}),
]


def _get_initial_final(character: str):
    """获取单个汉字的声母和韵母（去声调）"""
    if not is_chinese(character):
        return '', ''
    initial = pypinyin.pinyin(character, style=pypinyin.Style.INITIALS, strict=False)[0][0]
    final = pypinyin.pinyin(character, style=pypinyin.Style.FINALS, strict=False)[0][0]
    return initial, final


def _is_confused_pair(a: str, b: str, confused_groups) -> bool:
    """判断两个声母/韵母是否属于同一混淆组"""
    if a == b:
        return True
    for group in confused_groups:
        if a in group and b in group:
            return True
    return False


class MechanismInferencer:
    def __init__(self, glyph_threshold: float = 0.75):
        self.glyph_threshold = glyph_threshold

    @staticmethod
    def _normalized_pinyin(character: str):
        return tuple(convert_char_to_pinyin(character).tolist())

    def is_phonological_match(self, source_token: str, candidate_token: str) -> bool:
        """精确拼音匹配（含声调）"""
        if not is_chinese(source_token) or not is_chinese(candidate_token):
            return False
        return self._normalized_pinyin(source_token) == self._normalized_pinyin(candidate_token)

    def is_fuzzy_phonological_match(self, source_token: str, candidate_token: str) -> bool:
        """模糊拼音匹配：声母属于混淆组 + 韵母相同，或韵母属于混淆组 + 声母相同"""
        if not is_chinese(source_token) or not is_chinese(candidate_token):
            return False
        # 先检查精确匹配
        if self.is_phonological_match(source_token, candidate_token):
            return True
        # 拆分声母韵母
        s_initial, s_final = _get_initial_final(source_token)
        c_initial, c_final = _get_initial_final(candidate_token)
        # 声母混淆 + 韵母相同
        if _is_confused_pair(s_initial, c_initial, CONFUSED_INITIALS) and s_final == c_final:
            return True
        # 声母相同 + 韵母混淆
        if s_initial == c_initial and _is_confused_pair(s_final, c_final, CONFUSED_FINALS):
            return True
        return False

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
            # 使用模糊匹配替代精确匹配，扩大音近错误的召回
            if self.is_fuzzy_phonological_match(source_token, alternative.token):
                phonological_votes += 1
            if self.is_visual_match(source_token, alternative.token):
                visual_votes += 1

        if phonological_votes > visual_votes and phonological_votes > 0:
            return ErrorMechanism.PHONOLOGICAL
        if visual_votes > phonological_votes and visual_votes > 0:
            return ErrorMechanism.VISUAL
        return ErrorMechanism.UNCERTAIN

