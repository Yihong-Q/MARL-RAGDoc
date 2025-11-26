# MARL-RAGDoc

Reference implementation of the MARL-RAGDoc framework described in the design notes. The project demonstrates how to build a hierarchical multi-agent retrieval-augmented generation (RAG) pipeline that can reason over multimodal industrial documents.

## Project Layout

- `src/marl_ragdoc/data_types.py` – shared dataclasses (e.g., `Element`, `Candidate`).
- `src/marl_ragdoc/parsing/multimodal_parser.py` – document ingestion, OCR stubs, embedding hooks.
- `src/marl_ragdoc/retrieval/candidate_pool.py` – projection layers, similarity scoring, top-k pooling.
- `src/marl_ragdoc/agents/` – coordinator and modality agent policies plus neural networks.
- `src/marl_ragdoc/environment/rag_env.py` – hierarchical MARL environment that glues parsing, retrieval, and reasoning.
- `src/marl_ragdoc/reasoning/collaborative_reasoner.py` – weighted aggregation, reflection loop, LLM interface.
- `src/marl_ragdoc/training/trainer.py` – RL training harness, losses, rollout helpers.
- `src/marl_ragdoc/pipeline.py` – high-level orchestration entry point.

## Getting Started

```bash
poetry install
poetry run python -m marl_ragdoc.pipeline --config configs/demo.yaml
```

You must provision external OCR, table extraction, and LLM endpoints. The included components expose abstract hooks so you can plug in the tools used in your environment.

## Status

The code focuses on architecture clarity: state/action definitions, reward wiring, and module interfaces match the MARL-RAGDoc description. Production deployments should replace the mocked subsystems (OCR, embeddings, LLM scoring) with real services, add persistence for experience replay, and extend evaluation.

