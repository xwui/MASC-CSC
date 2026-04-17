"""
Frontend adapters for different NamBert loading backends.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

from utils.utils import pred_token_process


logger = logging.getLogger(__name__)


class HFNamBertFrontendAdapter:
    """
    Wrap a HuggingFace NamBert model directory and expose the
    `predict_with_metadata()` interface expected by the router pipeline.
    """

    def __init__(self, model_dir: str, device: str):
        self.model_dir = model_dir
        self.device = device
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
        self._model = self._model.to(device)
        self._model.eval()
        self._min_candidate_search_k = 32
        self._candidate_search_multiplier = 8

        if hasattr(self._model, "set_tokenizer"):
            self._model.set_tokenizer(self._tokenizer)

    @staticmethod
    def _compute_entropy(probabilities: torch.Tensor) -> torch.Tensor:
        return -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=-1)

    @staticmethod
    def _is_valid_char_token(token: str) -> bool:
        if not token:
            return False
        if token.startswith("##"):
            return False
        if token.startswith("[") and token.endswith("]"):
            return False
        return len(token) == 1

    @staticmethod
    def _normalize_scores(scores: List[float]) -> List[float]:
        total = float(sum(max(float(score), 0.0) for score in scores))
        if total <= 1e-12:
            return [1.0 / max(len(scores), 1)] * len(scores)
        return [float(max(score, 0.0) / total) for score in scores]

    def _candidate_search_k(self, top_k: int, vocab_size: int) -> int:
        return min(vocab_size, max(self._min_candidate_search_k, top_k * self._candidate_search_multiplier))

    def _build_valid_candidate_list(
        self,
        token_ids: List[int],
        token_probs: List[float],
        source_token: str,
        source_token_id: int,
        source_prob: float,
        top_k: int,
    ) -> Tuple[List[str], List[int], List[float]]:
        token_map = {}
        raw_tokens = self._tokenizer.convert_ids_to_tokens(token_ids)

        for token, token_id, token_prob in zip(raw_tokens, token_ids, token_probs):
            if not self._is_valid_char_token(token):
                continue
            score = float(token_prob)
            current = token_map.get(token)
            if current is None or score > current[1]:
                token_map[token] = (int(token_id), score)

        if self._is_valid_char_token(source_token):
            current = token_map.get(source_token)
            if current is None or float(source_prob) > current[1]:
                token_map[source_token] = (int(source_token_id), float(source_prob))

        if not token_map:
            return [source_token], [int(source_token_id)], [1.0]

        ranked = sorted(token_map.items(), key=lambda item: item[1][1], reverse=True)[:max(1, top_k)]
        tokens = [token for token, _ in ranked]
        ids = [token_id for _, (token_id, _) in ranked]
        probs = self._normalize_scores([score for _, (_, score) in ranked])
        return tokens, ids, probs

    @staticmethod
    def _build_passthrough_candidates(source_token: str) -> Tuple[List[str], List[int], List[float]]:
        return [source_token], [-1], [1.0]

    def _prepare_single_sentence_inputs(self, sentence: str):
        src_tokens = list(sentence)
        model_tokens: List[str] = []
        source_indices: List[int] = []
        for index, token in enumerate(src_tokens):
            if token.isspace():
                continue
            model_tokens.append(token)
            source_indices.append(index)

        bert_sentence = " ".join(model_tokens)
        inputs = self._tokenizer(bert_sentence, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        return src_tokens, source_indices, inputs

    def predict_with_metadata(
        self,
        sentence: str,
        top_k: int = 5,
        detection_threshold: float = 0.5,
    ) -> Dict[str, List]:
        src_tokens, source_indices, inputs = self._prepare_single_sentence_inputs(sentence)

        filtered_predicted_tokens = list(src_tokens)
        candidate_tokens: List[List[str]] = []
        candidate_ids: List[List[int]] = []
        candidate_probs: List[List[float]] = []
        copy_probs: List[float] = []
        detection_scores: List[float] = []
        uncertainty_scores: List[float] = []

        for source_token in src_tokens:
            tokens, ids, probs = self._build_passthrough_candidates(source_token)
            candidate_tokens.append(tokens)
            candidate_ids.append(ids)
            candidate_probs.append(probs)
            copy_probs.append(1.0)
            detection_scores.append(0.0)
            uncertainty_scores.append(0.0)

        if source_indices:
            with torch.no_grad():
                logits = self._model(**inputs).logits
                probabilities = torch.softmax(logits, dim=-1)

            input_ids = inputs["input_ids"]
            trimmed_probs = probabilities[0, 1:-1]
            trimmed_input_ids = input_ids[0, 1:-1]
            search_k = self._candidate_search_k(top_k=top_k, vocab_size=trimmed_probs.size(-1))
            search_probs, search_ids = torch.topk(trimmed_probs, k=search_k, dim=-1)

            copy_probs_input = trimmed_probs.gather(dim=-1, index=trimmed_input_ids.unsqueeze(-1)).squeeze(-1)
            keep_probs_virtual = trimmed_probs[:, 1]
            model_copy_probs = (copy_probs_input + keep_probs_virtual).detach().cpu().tolist()
            model_detection_scores = (1.0 - (copy_probs_input + keep_probs_virtual)).detach().cpu().tolist()
            model_uncertainty_scores = self._compute_entropy(trimmed_probs).detach().cpu().tolist()

            effective_len = min(
                len(source_indices),
                int(trimmed_probs.size(0)),
                len(model_copy_probs),
                len(model_detection_scores),
                len(model_uncertainty_scores),
            )
            effective_source_indices = source_indices[:effective_len]
            if effective_len != len(source_indices):
                logger.warning(
                    "Frontend alignment mismatch: source_indices=%d, model_steps=%d. Truncating to %d positions.",
                    len(source_indices),
                    int(trimmed_probs.size(0)),
                    effective_len,
                )

            model_source_tokens = [src_tokens[index] for index in effective_source_indices]
            model_predicted_tokens: List[str] = []

            for local_index, source_index in enumerate(effective_source_indices):
                src_tok = src_tokens[source_index]
                valid_tokens, valid_ids, valid_probs = self._build_valid_candidate_list(
                    token_ids=search_ids[local_index].detach().cpu().tolist(),
                    token_probs=search_probs[local_index].detach().cpu().tolist(),
                    source_token=src_tok,
                    source_token_id=int(trimmed_input_ids[local_index].item()),
                    source_prob=float(model_copy_probs[local_index]),
                    top_k=top_k,
                )
                candidate_tokens[source_index] = valid_tokens
                candidate_ids[source_index] = valid_ids
                candidate_probs[source_index] = valid_probs
                model_predicted_tokens.append(valid_tokens[0])

            model_predicted_tokens = pred_token_process(model_source_tokens, model_predicted_tokens)

            for local_index, source_index in enumerate(effective_source_indices):
                src_tok = src_tokens[source_index]
                pred_tok = model_predicted_tokens[local_index]
                det_score = float(model_detection_scores[local_index])
                filtered_predicted_tokens[source_index] = pred_tok if det_score >= detection_threshold else src_tok
                copy_probs[source_index] = float(model_copy_probs[local_index])
                detection_scores[source_index] = det_score
                uncertainty_scores[source_index] = float(model_uncertainty_scores[local_index])

        return {
            "source_text": "".join(src_tokens),
            "source_tokens": src_tokens,
            "predicted_tokens": filtered_predicted_tokens,
            "predicted_text": "".join(filtered_predicted_tokens),
            "topk_ids": candidate_ids,
            "topk_probs": candidate_probs,
            "topk_tokens": candidate_tokens,
            "copy_probs": copy_probs,
            "detection_scores": detection_scores,
            "uncertainty_scores": uncertainty_scores,
        }

    def predict(self, sentence: str) -> str:
        return self.predict_with_metadata(sentence)["predicted_text"]
