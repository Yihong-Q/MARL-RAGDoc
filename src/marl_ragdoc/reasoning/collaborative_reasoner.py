from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from ..config import ReasoningConfig
from ..data_types import EvidenceSet
from ..llm.llm_client import LLMClient, LLMResponse


@dataclass
class ReasoningOutputs:
    response: LLMResponse
    reflections: int


class CollaborativeReasoner:
    def __init__(self, config: ReasoningConfig):
        self.config = config
        self.llm = LLMClient(config.llm_model)

    def run(self, query: str, evidence: EvidenceSet, weights: Dict[str, float]) -> ReasoningOutputs:
        reflections = 0
        response = self.llm.generate(query, evidence, weights)
        while response.score < self.config.reflection_threshold and reflections < self.config.max_reflection_rounds:
            reflections += 1
            response = self.llm.generate(query, evidence, weights)
        return ReasoningOutputs(response=response, reflections=reflections)

