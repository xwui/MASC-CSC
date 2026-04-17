from typing import TYPE_CHECKING

from masc_csc.candidate_generator import MechanismAwareCandidateGenerator
from masc_csc.llm_verifier import NoOpVerifier
from masc_csc.mechanism import MechanismInferencer
from masc_csc.router import RiskAwareRouter
from masc_csc.types import (
    ErrorMechanism,
    PositionPrediction,
    SentencePrediction,
    TokenAlternative,
    VerificationResult,
)

if TYPE_CHECKING:
    from models.multimodal_frontend import MultimodalCSCFrontend


class MASCCSCPipeline:
    def __init__(
            self,
            frontend_model: "MultimodalCSCFrontend",
            candidate_generator: MechanismAwareCandidateGenerator,
            router: RiskAwareRouter,
            verifier=None,
            mechanism_inferencer: MechanismInferencer = None,
    ):
        self.frontend_model = frontend_model
        self.candidate_generator = candidate_generator
        self.router = router
        self.verifier = verifier or NoOpVerifier()
        self.mechanism_inferencer = mechanism_inferencer or MechanismInferencer()

    def _build_sentence_prediction(self, metadata: dict) -> SentencePrediction:
        positions = []
        for index, source_token in enumerate(metadata["source_tokens"]):
            alternative_tokens = metadata["topk_tokens"][index]
            alternative_ids = metadata["topk_ids"][index]
            alternative_scores = metadata["topk_probs"][index]
            alternatives = [
                TokenAlternative(token=token, token_id=token_id, score=float(score))
                for token, token_id, score in zip(alternative_tokens, alternative_ids, alternative_scores)
            ]
            mechanism = self.mechanism_inferencer.infer_from_alternatives(source_token, alternatives)
            positions.append(
                PositionPrediction(
                    index=index,
                    source_token=source_token,
                    predicted_token=metadata["predicted_tokens"][index],
                    detection_score=float(metadata["detection_scores"][index]),
                    uncertainty=float(metadata["uncertainty_scores"][index]),
                    mechanism=mechanism if isinstance(mechanism, ErrorMechanism) else ErrorMechanism.UNCERTAIN,
                    alternatives=alternatives,
                )
            )

        return SentencePrediction(
            source_text=metadata["source_text"],
            predicted_text=metadata["predicted_text"],
            positions=positions,
        )

    def analyze(self, sentence: str, top_k: int = 5) -> SentencePrediction:
        metadata = self.frontend_model.predict_with_metadata(sentence, top_k=top_k)
        return self._build_sentence_prediction(metadata)

    def correct(self, sentence: str, top_k: int = 5) -> VerificationResult:
        prediction = self.analyze(sentence, top_k=top_k)
        routing = self.router.decide(prediction)

        if routing.invoke_llm:
            candidates = self.candidate_generator.generate(prediction)
            result = self.verifier.verify(
                prediction,
                candidates,
                selected_positions=routing.supplement_positions,
            )
            result.reason = f"{result.reason}\n\nRouter: {routing.reasons}, risk={routing.risk_score:.2f}"
            return result

        return VerificationResult(
            text=prediction.predicted_text,
            selected_source="front-end-top1",
            reason=(
                f"Router skipped LLM verification and preserved frontend top1: "
                f"{routing.reasons}, risk={routing.risk_score:.2f}"
            ),
        )
