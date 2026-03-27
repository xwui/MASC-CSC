import json
import re
import string
from typing import List, Sequence
from urllib import request

from masc_csc.types import CandidateSentence, SentencePrediction, VerificationResult


class NoOpVerifier:
    def verify(self, prediction: SentencePrediction, candidates: Sequence[CandidateSentence]) -> VerificationResult:
        selected = max(candidates, key=lambda candidate: candidate.score)
        return VerificationResult(
            text=selected.text,
            selected_source=selected.source,
            reason="NoOp verifier fallback selected the highest-scoring candidate.",
            candidates=list(candidates),
        )


class LocalLLMVerifier:
    def __init__(
            self,
            model: str,
            base_url: str = "http://127.0.0.1:8000/v1",
            api_key: str = "EMPTY",
            temperature: float = 0.0,
            timeout: int = 30,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout

    @staticmethod
    def _label(index: int) -> str:
        return string.ascii_uppercase[index]

    def build_prompt(self, prediction: SentencePrediction, candidates: Sequence[CandidateSentence]) -> List[dict]:
        candidate_lines = []
        for index, candidate in enumerate(candidates):
            label = self._label(index)
            candidate_lines.append(f"{label}. {candidate.text}")

        user_prompt = (
            "You are a verifier for Chinese spelling correction.\n"
            "Pick the best candidate under these rules:\n"
            "1. Preserve original meaning.\n"
            "2. Prefer fewer edits.\n"
            "3. Do not rewrite style.\n"
            "4. If the source sentence is already correct, keep it.\n"
            "5. Choose only from the provided candidates.\n\n"
            f"Source sentence:\n{prediction.source_text}\n\n"
            f"Candidates:\n{chr(10).join(candidate_lines)}\n\n"
            "Output only in the following format:\n"
            "Choice: <label>\n"
            "Reason: <one sentence>"
        )

        return [
            {"role": "system", "content": "You are a careful verifier for Chinese spelling correction."},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_choice(self, content: str, candidates: Sequence[CandidateSentence]) -> int:
        match = re.search(r"Choice:\s*([A-Z])", content)
        if match is None:
            return 0
        label = match.group(1)
        index = string.ascii_uppercase.index(label)
        if index >= len(candidates):
            return 0
        return index

    def _chat(self, messages: List[dict]) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            }
        ).encode("utf-8")

        http_request = request.Request(
            url=f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        with request.urlopen(http_request, timeout=self.timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
        return raw["choices"][0]["message"]["content"]

    def verify(self, prediction: SentencePrediction, candidates: Sequence[CandidateSentence]) -> VerificationResult:
        messages = self.build_prompt(prediction, candidates)
        raw_content = self._chat(messages)
        choice_index = self._parse_choice(raw_content, candidates)
        selected = candidates[choice_index]

        return VerificationResult(
            text=selected.text,
            selected_source=selected.source,
            reason=raw_content.strip(),
            candidates=list(candidates),
        )
