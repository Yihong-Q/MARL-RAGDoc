from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple
import torch

from .networks import ModalityNet
from ..config import ModelConfig


class ModalityAction(Enum):
    SELECT = auto()
    EXPAND = auto()
    SKIP = auto()
    STOP = auto()


@dataclass
class ModalityState:
    query_embedding: torch.Tensor
    candidate_embedding: torch.Tensor
    selection_context: torch.Tensor
    layout_features: torch.Tensor

    def as_tensor(self) -> torch.Tensor:
        return torch.cat(
            [self.query_embedding, self.candidate_embedding, self.selection_context, self.layout_features],
            dim=-1,
        )


class ModalityAgent:
    def __init__(self, state_dim: int, config: ModelConfig):
        self.policy = ModalityNet(state_dim, len(ModalityAction), config)

    def act(self, state: ModalityState) -> Tuple[ModalityAction, torch.Tensor]:
        logits = self.policy(state.as_tensor())
        probs = torch.softmax(logits, dim=-1)
        action_idx = torch.multinomial(probs, num_samples=1).item()
        return ModalityAction(list(ModalityAction)[action_idx]), probs

