from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol
import numpy as np

from ..data_types import Element, Modality


class Encoder(Protocol):
    def encode(self, payload: str | bytes, modality: Modality) -> np.ndarray:
        ...


@dataclass
class ParserOutputs:
    elements: List[Element]


class MultimodalParsingModule:
    """
    Lightweight orchestrator over OCR, table structure recovery, and embedding extraction.
    Real implementations should override `_detect_regions` and `_extract_content`.
    """

    def __init__(self, encoder: Encoder):
        self._encoder = encoder

    def parse(self, document_path: str | Path) -> ParserOutputs:
        regions = self._detect_regions(document_path)
        elements = []
        for region in regions:
            content, modality, metadata = self._extract_content(region)
            embedding = self._encode(content, modality, metadata)
            elements.append(Element(content, modality, embedding, metadata))
        return ParserOutputs(elements=elements)

    def _detect_regions(self, document_path: str | Path) -> List[dict]:
        # Placeholder: integrate PDF/image analysers here
        return [{"type": "text", "content": "Placeholder paragraph.", "metadata": {}}]

    def _extract_content(self, region: dict) -> tuple[str, Modality, dict]:
        modality = {
            "text": Modality.TEXT,
            "image": Modality.IMAGE,
            "table": Modality.TABLE,
        }[region["type"]]
        return region["content"], modality, region.get("metadata", {})

    def _encode(self, content: str, modality: Modality, metadata: dict) -> np.ndarray:
        return self._encoder.encode(content.encode("utf-8"), modality)

