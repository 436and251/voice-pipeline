from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class FrontendResult:
    normalized_text: str
    phones: list[str]
    phone_ids: list[int]
    word2ph: list[int] | None
    bert_features: torch.Tensor

    def __post_init__(self) -> None:
        if len(self.phones) != len(self.phone_ids):
            raise ValueError("phone and phone-id lengths differ")
        if self.bert_features.shape != (1024, len(self.phone_ids)):
            raise ValueError("BERT columns must match phone IDs")
