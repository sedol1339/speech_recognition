from pathlib import Path
from typing import Any, Literal
from abc import ABC, abstractmethod

import numpy as np
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader

from src.metrics import EvalResults, LossResults


class ModelWrapper(ABC):
    """
    A wrapper for all code related to some specific model.

    Need to make training and testing code model-agnostic. Also defines some useful functions.
    """

    device: str

    def __init__(self, path: str, device: str):
        self.device = device

        optimizer_path = Path(path) / 'optimizer.pt'
        if not optimizer_path.is_file():
            self.optimizer = None
        else:
            self.optimizer = torch.load(
                optimizer_path, map_location=torch.device(self.device)
            )

    def save(self, path: str | Path, save_optimizer: bool = True):
        """
        Saves the model, processors/tokenizers, and optimizer (if present). Can be loaded further
        with the class constructor.
        """
        if save_optimizer:
            optimizer_path = Path(path) / 'optimizer.pt'
            if self.optimizer:
                optimizer_path.parent.mkdir(exist_ok=True, parents=True)
                torch.save(self.optimizer, optimizer_path)

    @abstractmethod
    def forward(
        self,
        batch: dict[str, Any],
        eval: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        A forward function.

        Accepts an input batch with 'audio' (datasets.features.Audio) and 'transcription' (str) features.
        Returns dict with 'loss' field, any any other model-specific fields such as 'logits' and 'labels'.

        If eval, forward in eval mode (in torch this typically means model.eval()), otherwise in train mode.
        However, eval mode does not uses no-grad context, so this context should be applied externally.

        The 'loss' field is a tensor of non-reduced losses. Paddings have NaN loss values. To backprop,
        use torch.nanmean.

        We do not calculate average loss here to 1) make loss evaluation on val/test sets more consistent,
        2) to allow different loss balancings between short and large sentences, and 3) to allow the
        calculation of bootstrap confidence intervals for loss.
        """
        pass

    @abstractmethod
    def transcribe(self, batch: dict[str, Any]) -> list[str]:
        """
        Decode a batch of samples.
        """
        pass

    @property
    @abstractmethod
    def sampling_rate(self) -> int:
        """
        A required sampling rate for audio samples.
        """
        pass

    @abstractmethod
    def get_modules(self) -> torch.nn.ModuleDict:
        """
        Returns a ModuleDict of torch modules.
        """
        pass

    @property
    def loss_type(self) -> Literal['ctc_loss', 'cross_entropy'] | None:
        """
        The type of loss used, to allow custom scaling or reducing for each
        type of loss. None means loss is not specified for the model.
        """
        return None

    @property
    def n_logits(self) -> float | None:
        """
        Such N that self.forward(...)['logits'].shape == (batch_size, N), or None,
        if the model does not return logits.

        This function may not be implemented.
        """
        raise NotImplementedError()

    def loss_from_logits(
        self, batch: dict[str, Any], logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Some model implementations may return 'logits' field in the .forward() output. For these
        models, the current function allows to calculate loss from logits in a differentiable
        way. This can be used to apply IRM-based methods.

        The batch should be already processed by .forward() function (it may add new fields).

        This function may not be implemented.
        """
        raise NotImplementedError()

    @property
    def allow_transcriptions_from_forward_outputs(self) -> bool:
        """
        Whether the method `.transcription_from_forward_outputs()` is implemented.

        For example, it may be implemented for CTC models, but may not be implemented for
        models that perform sequential decoding.
        """
        return False

    def transcriptions_from_forward_outputs(
        self, forward_outputs: dict[str, torch.Tensor]
    ) -> list[str]:
        """
        Some model implementations may allow to get transcriptions from `.forward()` outputs.

        For example, it may be implemented for CTC models, but may not be implemented for
        models that perform sequential decoding.

        See also `.allow_transcription_from_forward_outputs`.
        """
        raise NotImplementedError()

    def evaluate(
        self,
        dataloader: DataLoader,
        return_loss: bool = False,
        return_transcriptions: bool = True,
        loss_scale: float = 1.0,
        loss_n_bootstraps: int = 1000,
    ) -> EvalResults:
        """
        Run the model on the whole dataloader, returns loss and/or transcriptions.
        """
        val_losses = []
        true_texts = []
        pred_texts = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='evaluating'):
                preds = self.forward(batch, eval=True)
                if return_loss:
                    val_losses.append(preds['loss'].cpu().numpy())
                if return_transcriptions:
                    if self.allow_transcriptions_from_forward_outputs:
                        pred_texts += self.transcriptions_from_forward_outputs(preds)
                    else:
                        pred_texts += self.transcribe(batch)
                true_texts += batch['transcription']

        if return_loss:
            # concatenating
            ndims = {arr.ndim for arr in val_losses}
            if ndims == {1}:
                pass
            elif ndims == {2}:
                lengths = [arr.shape[1] for arr in val_losses]
                if len(set(lengths)) > 1:
                    # need nan paddings to concatenate
                    max_len = max(lengths)
                    val_losses = [
                        np.pad(
                            arr,
                            ((0, 0), (0, max_len - arr.shape[1])),
                            constant_values=np.nan,
                        )
                        for arr in val_losses
                    ]
            losses_numpy = loss_scale * np.concatenate(val_losses)
            loss = np.nanmean(losses_numpy)

            subsample_losses = []
            for _ in range(loss_n_bootstraps):
                indices = np.random.choice(len(losses_numpy), len(losses_numpy))
                score = np.nanmean(losses_numpy[indices])
                subsample_losses.append(score)
            loss_std = np.std(subsample_losses)

            loss_results = LossResults(
                losses=losses_numpy,
                loss=loss,
                loss_std=loss_std,
            )
        else:
            loss_results = None

        return EvalResults(
            true_texts=true_texts,
            pred_texts=pred_texts if return_transcriptions else None,
            loss_results=loss_results,
        )
