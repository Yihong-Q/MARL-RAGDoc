from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class RetrievalConfig:
    shared_dim: int = 128
    text_top_k: int = 10
    table_top_k: int = 8
    image_top_k: int = 6
    max_depth: int = 4


@dataclass
class ReasoningConfig:
    reflection_threshold: float = 0.7
    max_reflection_rounds: int = 2
    llm_model: str = "gpt-4o"
    summary_tokens: int = 256


@dataclass
class ModelConfig:
    hidden_dim: int = 256
    coordinator_layers: Tuple[int, int] = (256, 256)
    modality_layers: Tuple[int, int] = (256, 256)
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gamma: float = 0.95
    batch_size: int = 8
    max_steps: int = 20000
    rollout_len: int = 32


@dataclass
class SystemConfig:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: str = "cuda"

    @classmethod
    def default(cls) -> "SystemConfig":
        return cls()

