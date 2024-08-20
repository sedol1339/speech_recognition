from __future__ import annotations

import string
from typing import Any

import numpy as np
import torch


from .model import ModelWrapper


class DummyWrapper(ModelWrapper):
    """
    A dummy model for debugging train scripts. Mimics real model behaviour.
    """

    def __init__(self):
        super().__init__(path='', device='')
        self.n_forwards = 0

    def forward(
        self,
        batch: dict[str, Any],
        eval: bool = False,
    ) -> dict[str, torch.Tensor]:
        losses = 2 ** np.random.normal(-self.n_forwards / 1000, size=8)
        return {'loss': torch.tensor(losses, requires_grad=True)}

    def transcribe(self, batch: dict[str, Any]) -> list[str]:
        return [self._distort(text) for text in batch['transcription']]

    @property
    def sampling_rate(self) -> int:
        return 16_000

    def get_modules(self) -> torch.nn.ModuleDict:
        return torch.nn.ModuleDict({})

    def _distort(self, text: str) -> str:
        distort_prob = np.clip(0.1 + 0.1 * (self.n_forwards / 1000), 0, 1)
        chars = list(string.ascii_lowercase)
        return ''.join([
            (
                c
                if np.random.rand() > distort_prob
                else str(np.random.choice(chars, size=1))
            )
            for c in text
        ])
