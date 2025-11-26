from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, int], output_dim: int, dropout: float):
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CoordinatorNet(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, config: ModelConfig):
        super().__init__()
        self.policy_head = MLP(input_dim, config.coordinator_layers, action_dim, config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy_head(x)


class ModalityNet(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, config: ModelConfig):
        super().__init__()
        self.policy_head = MLP(input_dim, config.modality_layers, action_dim, config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy_head(x)

