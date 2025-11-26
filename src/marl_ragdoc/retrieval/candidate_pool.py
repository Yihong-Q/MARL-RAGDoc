from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np

from ..config import RetrievalConfig
from ..data_types import Candidate, Element, Modality, QueryContext


@dataclass
class CandidatePools:
    text: List[Candidate]
    images: List[Candidate]
    tables: List[Candidate]

    def as_dict(self) -> Dict[Modality, List[Candidate]]:
        return {
            Modality.TEXT: self.text,
            Modality.IMAGE: self.images,
            Modality.TABLE: self.tables,
        }


class CandidatePoolBuilder:
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.projection = {
            Modality.TEXT: np.eye(config.shared_dim),
            Modality.IMAGE: np.eye(config.shared_dim),
            Modality.TABLE: np.eye(config.shared_dim),
        }

    def build(self, elements: List[Element], query: QueryContext) -> CandidatePools:
        per_modality: Dict[Modality, List[Candidate]] = {
            Modality.TEXT: [],
            Modality.IMAGE: [],
            Modality.TABLE: [],
        }
        for element in elements:
            proj = self._project(element)
            similarity = self._cosine(query.embedding, proj)
            per_modality[element.modality].append(
                Candidate(element=element, similarity=similarity)
            )
        return CandidatePools(
            text=self._top_k(per_modality[Modality.TEXT], self.config.text_top_k),
            images=self._top_k(per_modality[Modality.IMAGE], self.config.image_top_k),
            tables=self._top_k(per_modality[Modality.TABLE], self.config.table_top_k),
        )

    def _project(self, element: Element) -> np.ndarray:
        matrix = self.projection[element.modality]
        return matrix @ element.embedding

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _top_k(candidates: List[Candidate], k: int) -> List[Candidate]:
        return sorted(candidates, key=lambda c: c.similarity, reverse=True)[:k]

