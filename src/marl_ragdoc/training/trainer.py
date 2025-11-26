from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import torch
from torch import nn, optim

from ..config import SystemConfig
from ..environment.rag_env import RAGEnvironment
from ..agents.coordinator import CoordinatorAgent, CoordinatorAction, CoordinatorState
from ..agents.modality_agent import ModalityAction, ModalityAgent, ModalityState
from ..data_types import CandidatePools, Modality, QueryContext


@dataclass
class RolloutStats:
    rewards: List[float]
    coordinator_actions: List[CoordinatorAction]


class Trainer:
    def __init__(self, system_config: SystemConfig, env: RAGEnvironment, state_dims: Dict[str, int]):
        self.cfg = system_config
        self.env = env
        self.coordinator = CoordinatorAgent(state_dims["coordinator"], system_config.model)
        self.modality_agents = {
            Modality.TEXT: ModalityAgent(state_dims["text"], system_config.model),
            Modality.IMAGE: ModalityAgent(state_dims["image"], system_config.model),
            Modality.TABLE: ModalityAgent(state_dims["table"], system_config.model),
        }
        params = list(self.coordinator.policy.parameters())
        for agent in self.modality_agents.values():
            params.extend(agent.policy.parameters())
        self.optimizer = optim.Adam(params, lr=system_config.training.learning_rate, weight_decay=system_config.training.weight_decay)

    def rollout(self, pools: CandidatePools, query: QueryContext) -> RolloutStats:
        stats = RolloutStats(rewards=[], coordinator_actions=[])
        self.env.reset()
        self.env.seed(pools)
        done = False
        while not done:
            coord_state = self._build_coordinator_state(query)
            action, _ = self.coordinator.act(coord_state)
            stats.coordinator_actions.append(action)
            state, reward, done = self.env.coordinator_step(action.name.replace("_", " ").title().replace(" ", ""))
            stats.rewards.append(reward)
            if done:
                break
            for modality, candidates in state.pools.as_dict().items():
                for candidate in candidates:
                    mod_state = self._build_modality_state(query, candidate.element.embedding)
                    mod_action, _ = self.modality_agents[modality].act(mod_state)
                    _, mod_reward, stop = self.env.modality_step(modality, mod_action.name.title(), candidate)
                    stats.rewards.append(mod_reward)
                    if stop:
                        break
        return stats

    def _build_coordinator_state(self, query: QueryContext) -> CoordinatorState:
        query_tensor = torch.from_numpy(query.embedding).float()
        modality_context = torch.zeros_like(query_tensor)
        global_meta = torch.zeros_like(query_tensor)
        return CoordinatorState(query_tensor, modality_context, global_meta)

    def _build_modality_state(self, query: QueryContext, candidate: torch.Tensor) -> ModalityState:
        query_tensor = torch.from_numpy(query.embedding).float()
        candidate_tensor = torch.from_numpy(candidate).float()
        selection_context = torch.zeros_like(query_tensor)
        layout = torch.zeros_like(query_tensor)
        return ModalityState(query_tensor, candidate_tensor, selection_context, layout)

