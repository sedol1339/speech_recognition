from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from torch.nn import CrossEntropyLoss
from transformers import WhisperProcessor, GenerationConfig
from transformers import WhisperForConditionalGeneration

from src.datasets import get_hf_token

from .model import ModelWrapper


class WhisperWrapper(ModelWrapper):
    """
    A wrapper for all code related to Whisper.
    """

    def __init__(
        self,
        path: str = 'openai/whisper-medium',
        language: str = 'russian',
        task: str = 'transcribe',
        device: str = 'cuda',
        model_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(path, device)

        model_kwargs = model_kwargs or {}
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
        self.model = WhisperForConditionalGeneration.from_pretrained(
            path, token=get_hf_token(), **model_kwargs
        ).to(self.device)
        print(
            f'Model has {sum(p.numel() for p in self.model.parameters())//10**6}M'
            ' trainable weights'
        )

        self.processor = WhisperProcessor.from_pretrained(
            path, language=language, task=task
        )

        self.generation_config = GenerationConfig.from_pretrained(path)
        self.generation_config.forced_decoder_ids = (
            self.processor.get_decoder_prompt_ids(language=language, task=task)
        )
        self.generation_config.num_beams = 1
        self.generation_config.do_sample = False

    def save(self, path: str | Path, save_optimizer: bool = True):
        """
        Saves the model, processor, and optimizer (if present). Can be loaded further with the
        class constructor.
        """
        super().save(path, save_optimizer)
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
        self.generation_config.save_pretrained(path)

    def _get_mfcc_and_attention_mask(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mfccs = [
            self.processor.feature_extractor(
                audio['array'],
                sampling_rate=audio['sampling_rate'],
            ).input_features[0]
            for audio in batch['audio']
        ]
        results = self.processor.feature_extractor.pad(
            {'input_features': mfccs},
            return_attention_mask=True,
            return_tensors='pt',
            padding='longest',
        )
        return (
            results.input_features.to(self.device),
            results.attention_mask.to(self.device),
        )

    def _get_transcription_token_ids(self, batch: dict[str, Any]) -> torch.Tensor:
        token_ids = [
            self.processor.tokenizer(t).input_ids for t in batch['transcription']
        ]
        results = self.processor.tokenizer.pad(
            {'input_ids': token_ids},
            return_tensors='pt',
            padding='longest',
        )
        return results.input_ids.masked_fill(
            results.attention_mask == 0,
            -100,  # default ignore_index for torch.nn.CrossEntropyLoss
        ).to(self.device)

    def forward(
        self,
        batch: dict[str, Any],
        eval: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        A forward function for Whisper.

        Accepts an input batch with 'audio' (datasets.features.Audio) and 'transcription' (str) features.
        Returns dict with 'loss', 'logits' and 'labels' fields.

        If eval, forward in eval mode (in torch this typically means model.eval()), otherwise in train mode.
        However, eval mode does not uses no-grad context, so this context should be applied externally.

        The 'loss' field is a 2d-tensor of per-token losses of shape (batch_size, n_tokens). Paddings have
        NaN loss values. To backprop, use torch.nanmean.

        We do not calculate average loss here to 1) make loss evaluation on val/test sets more consistent,
        2) to allow different loss balancings between short and large sentences, and 3) to allow the
        calculation of bootstrap confidence intervals for loss.
        """
        self.model.eval() if eval else self.model.train()

        (
            batch['_input_features'],
            batch['_attention_mask'],
        ) = self._get_mfcc_and_attention_mask(batch)
        batch['_labels'] = self._get_transcription_token_ids(batch)

        outputs = self.model(
            input_features=batch['_input_features'],
            attention_mask=batch['_attention_mask'],
            labels=batch['_labels'],
            return_dict=True,
        )
        losses = torch.nn.functional.cross_entropy(
            input=torch.moveaxis(outputs.logits, 2, 1),
            target=batch['_labels'],
            reduction='none',
        )
        losses[batch['_labels'] == -100] = torch.nan
        return {
            'loss': losses,
            'logits': outputs.logits,
            'labels': batch['_labels'],
        }

    def transcribe(self, batch: dict[str, Any]) -> list[str]:
        """
        Decode a batch of samples with Whisper in no-grad mode.
        """
        input_features, attention_mask = self._get_mfcc_and_attention_mask(batch)

        with torch.no_grad():
            generated_token_ids = self.model.generate(
                input_features,
                attention_mask=attention_mask,
                generation_config=self.generation_config,
            )
        transcriptions = self.processor.tokenizer.batch_decode(
            generated_token_ids,
            num_processes=1,
            skip_special_tokens=True,
        )
        return transcriptions

    @property
    def sampling_rate(self) -> int:
        return self.processor.feature_extractor.sampling_rate

    def get_modules(self) -> torch.nn.ModuleDict:
        """
        Returns a ModuleDict of trainable torch modules, with only one module
        which is WhisperForConditionalGeneration.
        """
        return torch.nn.ModuleDict({'model': self.model})

    @property
    def loss_type(self) -> Literal['ctc_loss', 'cross_entropy'] | None:
        return 'cross_entropy'

    @property
    def n_logits(self) -> float:
        """
        Such N that self.forward(...)['logits'].shape == (batch_size, N).

        Equals vocab_size for Whisper.
        """
        return self.model.config.vocab_size

    def loss_from_logits(
        self, batch: dict[str, Any], logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates loss from logits in a differentiable way. This can be used to apply
        IRM-based methods. Follows the implementation in WhisperForConditionalGeneration.

        The batch should be already processed by .forward() function (it adds new fields).
        """
        loss_fct = CrossEntropyLoss()
        return loss_fct(
            logits.view(-1, self.model.config.vocab_size), labels.reshape(-1)
        )
