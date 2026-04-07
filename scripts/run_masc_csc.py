import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from masc_csc import (  # noqa: E402
    BaichuanLocalVerifier,
    MechanismAwareCandidateGenerator,
    MASCCSCPipeline,
    MechanismInferencer,
    NoOpVerifier,
    RiskAwareRouter,
)
from models.multimodal_frontend import MultimodalCSCFrontend  # noqa: E402


def build_model_args(device: str):
    return SimpleNamespace(
        device=device,
        hyper_params={},
    )


def load_frontend_model(checkpoint_path: str, device: str):
    args = build_model_args(device=device)
    model = MultimodalCSCFrontend(args)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        state_dict = state["state_dict"] if "state_dict" in state else state
        model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Run the MASC-CSC pipeline.")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Path to the frontend checkpoint.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sentence", type=str, required=True, help="Input sentence for correction.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--max-positions", type=int, default=2)
    parser.add_argument("--use-llm", action="store_true", help="Enable local LLM verification.")
    parser.add_argument("--llm-model-path", type=str, default="baichuan-inc/Baichuan-7B",
                        help="Local path or HuggingFace Hub ID for Baichuan-7B.")
    parser.add_argument("--llm-adapter-path", type=str, default=None,
                        help="Path to LoRA adapter (optional, for fine-tuned model).")
    return parser.parse_args()


def main():
    args = parse_args()

    frontend_model = load_frontend_model(args.ckpt_path, args.device)
    mechanism_inferencer = MechanismInferencer()
    candidate_generator = MechanismAwareCandidateGenerator(
        mechanism_inferencer=mechanism_inferencer,
        max_positions=args.max_positions,
        max_candidates=args.max_candidates,
    )
    router = RiskAwareRouter()
    verifier = (
        BaichuanLocalVerifier(
            model_path=args.llm_model_path,
            adapter_path=args.llm_adapter_path,
            device=args.device,
        )
        if args.use_llm
        else NoOpVerifier()
    )
    pipeline = MASCCSCPipeline(
        frontend_model=frontend_model,
        candidate_generator=candidate_generator,
        router=router,
        verifier=verifier,
        mechanism_inferencer=mechanism_inferencer,
    )

    result = pipeline.correct(args.sentence, top_k=args.top_k)
    print("Input :", args.sentence)
    print("Output:", result.text)
    print("Source:", result.selected_source)
    print("Reason:", result.reason)


if __name__ == "__main__":
    main()

