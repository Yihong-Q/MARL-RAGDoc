from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional
import numpy as np


class Modality(Enum):
    TEXT = auto()
    IMAGE = auto()
    TABLE = auto()


@dataclass
class Element:
    content: str
    modality: Modality
    embedding: np.ndarray
    metadata: Dict[str, float | int | str]


@dataclass
class Candidate:
    element: Element
    similarity: float
    depth: int = 0


@dataclass
class QueryContext:
    query: str
    embedding: np.ndarray


@dataclass
class EvidenceSet:
    text: List[Element] = field(default_factory=list)
    images: List[Element] = field(default_factory=list)
    tables: List[Element] = field(default_factory=list)

    def add(self, element: Element) -> None:
        if element.modality == Modality.TEXT:
            self.text.append(element)
        elif element.modality == Modality.IMAGE:
            self.images.append(element)
        else:
            self.tables.append(element)

    def summary(self) -> Dict[str, int]:
        return {
            "text": len(self.text),
            "images": len(self.images),
            "tables": len(self.tables),
        }

