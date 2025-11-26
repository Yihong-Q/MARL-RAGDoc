from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple
import torch

from .networks import CoordinatorNet
from ..config import ModelConfig


class CoordinatorAction(Enum):
    UPDATE_WEIGHTS = auto()
    INCREASE_DEPTH = auto()
    TERMINATE = auto()


@dataclass
class CoordinatorState:
    query_embedding: torch.Tensor
    modality_context: torch.Tensor  # concatenated summaries
    global_meta: torch.Tensor

    def as_tensor(self) -> torch.Tensor:
        return torch.cat([self.query_embedding, self.modality_context, self.global_meta], dim=-1)


class CoordinatorAgent:
    def __init__(self, state_dim: int, config: ModelConfig):
        self.policy = CoordinatorNet(state_dim, len(CoordinatorAction), config)

    def act(self, state: CoordinatorState) -> Tuple[CoordinatorAction, torch.Tensor]:
        logits = self.policy(state.as_tensor())
        probs = torch.softmax(logits, dim=-1)
        action_idx = torch.multinomial(probs, num_samples=1).item()
        return CoordinatorAction(list(CoordinatorAction)[action_idx]), probs

