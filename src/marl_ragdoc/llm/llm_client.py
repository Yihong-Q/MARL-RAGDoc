from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import random

from ..data_types import EvidenceSet


@dataclass
class LLMResponse:
    answer: str
    score: float


class LLMClient:
    """
    Placeholder LLM client. Replace with actual API calls.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, query: str, evidence: EvidenceSet, weights: Dict[str, float]) -> LLMResponse:
        summary = f"Answer referencing {evidence.summary()} with weights {weights}"
        score = random.uniform(0.4, 0.95)
        return LLMResponse(answer=summary, score=score)

