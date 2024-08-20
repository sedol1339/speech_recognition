from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


DEFAULT_QUANTILES = (0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0)


class TensorTracker:
    def __init__(
        self,
        tensor: torch.Tensor,
        quantiles: tuple = DEFAULT_QUANTILES,
        n_elements: int = 10,
        elements_seed: int = 0,
    ):
        n_elements = min(n_elements, tensor.numel())
        rng = np.random.default_rng(elements_seed)
        self.tensor = tensor
        self.quantiles = quantiles
        self.indices: list[int] | list[tuple]
        if tensor.ndim == 1 and n_elements == tensor.numel():
            # all elements
            self.indices = list(range(len(tensor)))
        else:
            self.indices = [
                tuple(rng.choice(length) for length in tensor.size())
                for _ in range(n_elements)
            ]
        self.history: list[dict[str, Any]] = []

    def update(self, step: int):
        self.history.append({
            'step': step,
            'mean': float(torch.mean(self.tensor)),
            'std': float(torch.std(self.tensor)),
            'quantiles': [
                float(torch.quantile(self.tensor, q)) for q in self.quantiles
            ],
            'values': [float(self.tensor[idxs]) for idxs in self.indices],
        })

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.history)
        quantiles = pd.DataFrame(
            df.pop('quantiles').to_list(), columns=[f'q{q:g}' for q in self.quantiles]
        )
        values = pd.DataFrame(
            df.pop('values').to_list(),
            columns=[f'val{i}' for i in range(len(self.indices))],
        )
        return pd.concat([df, quantiles, values], axis='columns')

    def to_csv(self, path: str | Path, precision: int = 3):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(
            path,
            float_format=f'%.{precision}f',
            index=False,
            compression='infer',
        )


class MultiTensorTracker:
    def __init__(
        self,
        state_dict: dict[str, torch.Tensor],
        quantiles: tuple = DEFAULT_QUANTILES,
        n_elements: int = 10,
        n_elements_override: dict[str, int] | None = None,
        elements_seed: int = 0,
    ):
        n_elements_override = n_elements_override or {}
        self.watchers = {
            name: TensorTracker(
                tensor,
                quantiles=quantiles,
                n_elements=n_elements_override.get(name, n_elements),
                elements_seed=elements_seed,
            )
            for name, tensor in state_dict.items()
        }

    def update(self, step: int):
        for watcher in self.watchers.values():
            watcher.update(step)

    def to_dataframes(self) -> dict[str, pd.DataFrame]:
        return {name: watcher.to_dataframe() for name, watcher in self.watchers.items()}

    def to_csvs(self, dir: str | Path, precision: int = 3, compression: bool = True):
        ext = 'zip' if compression else 'csv'
        dir = Path(dir)
        for name, watcher in self.watchers.items():
            valid_name = ''.join(c for c in name if (c.isalnum() or c in "._- "))
            watcher.to_csv(dir / f'{valid_name}.{ext}', precision=precision)
