from __future__ import annotations

import argparse
import numpy as np

from .config import SystemConfig
from .data_types import QueryContext
from .parsing.multimodal_parser import MultimodalParsingModule
from .retrieval.candidate_pool import CandidatePoolBuilder
from .environment.rag_env import RAGEnvironment
from .reasoning.collaborative_reasoner import CollaborativeReasoner
from .training.trainer import Trainer


class DummyEncoder:
    def encode(self, payload: bytes, modality):
        np.random.seed(len(payload))
        return np.random.randn(128)


def run_demo(document_path: str, query: str) -> None:
    config = SystemConfig.default()
    parser = MultimodalParsingModule(DummyEncoder())
    parsed = parser.parse(document_path)

    query_ctx = QueryContext(query=query, embedding=np.random.randn(128))
    pool_builder = CandidatePoolBuilder(config.retrieval)
    pools = pool_builder.build(parsed.elements, query_ctx)

    env = RAGEnvironment(config.retrieval)
    trainer = Trainer(config, env, state_dims={"coordinator": 384, "text": 512, "image": 512, "table": 512})
    stats = trainer.rollout(pools, query_ctx)

    reasoner = CollaborativeReasoner(config.reasoning)
    outputs = reasoner.run(query, env.state.selected, {k.name: v for k, v in env.state.weights.items()})
    print("Rollout rewards:", stats.rewards)
    print("Answer:", outputs.response.answer)
    print("Reflections:", outputs.reflections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--document", required=False, default="demo.pdf")
    parser.add_argument("--query", required=False, default="Explain safety procedures.")
    args = parser.parse_args()
    run_demo(args.document, args.query)

