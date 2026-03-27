# MASC-CSC Layer

This package contains the core inference modules for the MASC-CSC project.

MASC-CSC stands for:

- Mechanism-Aware
- Selective
- Collaboration
- for Chinese Spelling Correction

The package is built on top of a small-model CSC frontend and provides:

- error mechanism inference
- mechanism-aware candidate generation
- risk-aware routing
- local LLM verification

Main files:

- `types.py`: shared data structures
- `mechanism.py`: error mechanism inference
- `candidate_generator.py`: constrained candidate generation
- `router.py`: risk-aware routing
- `llm_verifier.py`: local LLM verifier interface
- `pipeline.py`: full MASC-CSC inference pipeline
