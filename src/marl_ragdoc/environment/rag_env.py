from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from ..config import RetrievalConfig
from ..data_types import Candidate, CandidatePools, EvidenceSet, Modality, QueryContext


@dataclass
class EnvState:
    depth: int
    weights: Dict[Modality, float]
    selected: EvidenceSet
    pools: CandidatePools


class RAGEnvironment:
    """
    Simplified hierarchical RAG environment.
    Coordinator controls weights/depth; modality agents select candidates.
    """

    def __init__(self, retrieval_cfg: RetrievalConfig):
        self.cfg = retrieval_cfg
        self.reset()

    def reset(self) -> EnvState:
        weights = {Modality.TEXT: 1 / 3, Modality.IMAGE: 1 / 3, Modality.TABLE: 1 / 3}
        self.state = EnvState(depth=0, weights=weights, selected=EvidenceSet(), pools=None)
        return self.state

    def seed(self, pools: CandidatePools) -> None:
        self.state.pools = pools

    def coordinator_step(self, action: str) -> Tuple[EnvState, float, bool]:
        if action == "UpdateWeights":
            self._randomize_weights()
        elif action == "IncreaseDepth":
            self.state.depth += 1
        elif action == "Terminate":
            return self.state, self._terminal_reward(), True
        done = self.state.depth >= self.cfg.max_depth
        return self.state, 0.0, done

    def modality_step(self, modality: Modality, action: str, candidate: Candidate) -> Tuple[EnvState, float, bool]:
        reward = 0.0
        if action == "Select":
            self.state.selected.add(candidate.element)
            reward = candidate.similarity
        elif action == "Expand":
            self.state.selected.add(candidate.element)
            reward = candidate.similarity * 1.1
        elif action == "Stop":
            return self.state, reward, True
        return self.state, reward, False

    def _randomize_weights(self) -> None:
        values = np.random.dirichlet([1.0, 1.0, 1.0])
        for idx, modality in enumerate([Modality.TEXT, Modality.IMAGE, Modality.TABLE]):
            self.state.weights[modality] = float(values[idx])

    def _terminal_reward(self) -> float:
        stats = self.state.selected.summary()
        return stats["text"] * 0.2 + stats["images"] * 0.3 + stats["tables"] * 0.5

