from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from src.datasets import get_hf_token

from .model import ModelWrapper


class Wav2vec2Wrapper(ModelWrapper):
    """
    A wrapper for all code related to Whisper.
    """

    def __init__(
        self,
        path: str = 'facebook/wav2vec2-base-960h',
        add_ru_vocab: bool = False,
        device: str = 'cuda',
        input_text_case: Literal['upper', 'lower'] | None = None,
        drop_unk_from_input_text: bool = True,
        pad_to_seconds: float | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ):
        """
        Note that model outputs depend on the size of zero pad. From this it follows that batching
        (batch size, samples order) inevitably affects predictions and loss. Code to reproduce:
        https://github.com/pytorch/audio/issues/2242#issuecomment-2274573650

        `pad_to_seconds` parameter is only to determine max memory usage and shouldn't be used in
        a regular setting.
        """
        super().__init__(path, device)

        model_kwargs = model_kwargs or {}
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
        self.model = Wav2Vec2ForCTC.from_pretrained(
            path, token=get_hf_token(), **model_kwargs
        ).to(device)
        self.processor = Wav2Vec2Processor.from_pretrained(path)
        self.input_text_case = input_text_case
        if add_ru_vocab:
            assert input_text_case is not None
            self.add_ru_vocab()

        self.model.config.ctc_loss_reduction = 'none'
        assert self.processor.tokenizer.pad_token_id == self.model.config.pad_token_id
        self.drop_unk_from_input_text = drop_unk_from_input_text
        self.pad_to_seconds = pad_to_seconds

    def add_ru_vocab(
        self,
        phonetic_embeddings_init: bool = True,
    ):
        """
        Checks if the loaded tokenizer contains these chars (one char = one token). If not, adds
        new tokens for Russian letters and modifies lm_head by adding new (zero-initialized)
        weights for the corresponding new logits. Note that this does not reset weights for
        the existing logits. However, the model's argmax outputs may change after this, since
        new weights are 0 and not -inf.
        """
        russan_letters = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ-'
        phonetic_transliteration = 'ABVGDE~JZIYKLMNOPRSTUFHC~~~~~~~~~~'
        if self.input_text_case == 'lower':
            russan_letters = russan_letters.lower()
            phonetic_transliteration = phonetic_transliteration.lower()
        n_added = self.processor.tokenizer.add_tokens(list(russan_letters))
        if n_added == 0:
            return

        print(f'Added {n_added} new tokens to the vocabulary')

        new_vocab_size = max(self.processor.tokenizer.get_vocab().values()) + 1

        lm_head = torch.nn.Linear(self.model.lm_head.in_features, new_vocab_size).to(
            self.device
        )
        lm_head.weight.data.fill_(0)
        lm_head.bias.data.fill_(0)
        with torch.no_grad():
            lm_head.weight[: self.model.config.vocab_size, :] = (
                self.model.lm_head.weight
            )
            lm_head.bias[: self.model.config.vocab_size] = self.model.lm_head.bias

        if phonetic_embeddings_init:
            new_vocab = self.processor.tokenizer.get_vocab()
            for ru, en in zip(russan_letters, phonetic_transliteration):
                if en == '~':
                    continue
                ru_idx = new_vocab[ru]
                en_idx = new_vocab[en]
                with torch.no_grad():
                    lm_head.weight[ru_idx, :] = lm_head.weight[en_idx, :]
                    lm_head.bias[ru_idx] = lm_head.bias[en_idx] + 0.01

        self.model.lm_head = lm_head
        self.model.config.vocab_size = new_vocab_size

    def save(self, path: str | Path, save_optimizer: bool = True):
        """
        Saves the model, processor, and optimizer (if present). Can be loaded further with the
        class constructor.
        """
        super().save(path, save_optimizer)
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)

    def _get_transcription_token_ids(self, batch: dict[str, Any]) -> torch.Tensor:
        # upper or lower case
        if self.input_text_case == 'upper':
            transcriptions = [t.upper() for t in batch['transcription']]
        elif self.input_text_case == 'lower':
            transcriptions = [t.lower() for t in batch['transcription']]
        else:
            transcriptions = batch['transcription']

        # tokenizing
        tokens_for_samples = []
        for transcription in transcriptions:
            # encoding
            tokens = self.processor.tokenizer(transcription).input_ids
            if self.drop_unk_from_input_text:
                # removing <unk> token
                tokens = [
                    t for t in tokens if t != self.processor.tokenizer.unk_token_id
                ]
            # converting to the required format
            tokens_for_samples.append({
                'input_ids': tokens,
                'attention_mask': [1] * len(tokens),
            })

        # collating tokens with paddings
        padding_results = self.processor.tokenizer.pad(
            tokens_for_samples, return_tensors='pt'
        )

        # ignoring loss on paddings by using token id -100
        transcription_token_ids = padding_results.input_ids.masked_fill(
            padding_results.attention_mask == 0, -100
        )

        return transcription_token_ids.to(self.device)

    def _get_preprocessed_waveforms(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        waveforms = [audio['array'] for audio in batch['audio']]
        if self.pad_to_seconds is not None:
            min_len = self.sampling_rate * self.pad_to_seconds
            waveforms = [
                np.concatenate([w, np.zeros(min_len - len(w))]) for w in waveforms
            ]
        # collating waveforms with paddings
        feature_extractor_results = self.processor.feature_extractor(
            waveforms,
            return_tensors='pt',
            padding='longest',
            sampling_rate=self.sampling_rate,
            return_attention_mask=True,
        )
        return (
            feature_extractor_results.input_values.to(self.device),
            feature_extractor_results.attention_mask.to(self.device),
        )

    def forward(
        self,
        batch: dict[str, Any],
        eval: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        A forward function for Wav2vec2.

        Accepts an input batch with 'audio' (datasets.features.Audio) and 'transcription' (str) features.
        Returns dict with 'loss', 'logits' and 'labels' fields.

        If eval, forward in eval mode (in torch this typically means model.eval()), otherwise in train mode.
        However, eval mode does not uses no-grad context, so this context should be applied externally.

        Wav2vec2 uses CTC loss, which is a single scalar per sample. The raw CTC loss values (which are
        typically very large) are divided by 100. No reduction is used (same as reduction='none') in
        torch.nn.functional.ctc_loss. Also, CTC loss may return infinity, such values get filled with
        zero. The recommended way is to further apply torch.mean to the 'loss' field, so longer
        samples with have a greater impact on the average value, as done in Whisper.
        """
        self.model.eval() if eval else self.model.train()
        self.model.config.ctc_loss_reduction = 'none'

        # we set ctc_zero_infinity, otherwise we may get inf on some sample, which gives NaN
        # gradients, even if we filter out inf with masked_select (bug?)
        # https://github.com/pytorch/pytorch/issues/67180
        # https://github.com/pytorch/pytorch/issues/68425
        self.model.config.ctc_zero_infinity = True

        batch['_labels'] = self._get_transcription_token_ids(batch)

        (
            batch['_waveforms'],
            batch['_attention_mask'],
        ) = self._get_preprocessed_waveforms(batch)

        # checking that waveforms are long enough
        input_non_padded_lengths = batch['_attention_mask'].sum(dim=-1)
        output_lengths = self.model.wav2vec2._get_feat_extract_output_lengths(
            input_non_padded_lengths, add_adapter=False
        )
        assert (output_lengths > 0).all(), 'The input non-padded waveform is too short'

        # forward
        outputs = self.model(
            batch['_waveforms'],
            attention_mask=batch['_attention_mask'],
            labels=batch['_labels'],
        )

        # inf to nan
        return {
            'loss': outputs.loss,
            'logits': outputs.logits,
            'labels': batch['_labels'],
        }

    @property
    def allow_transcription_from_forward_outputs(self) -> bool:
        return True

    def transcriptions_from_forward_outputs(
        self, forward_outputs: dict[str, torch.Tensor]
    ) -> list[str]:
        """
        Get transcriptions from Wav2vec2 `.forward()` outputs.
        """
        logits_argmax = torch.argmax(forward_outputs['logits'], dim=-1)
        return self.processor.batch_decode(logits_argmax, skip_special_tokens=True)

    def transcribe(self, batch: dict[str, Any]) -> list[str]:
        """
        Decode a batch of samples with Wav2vec2 in no-grad mode.
        """
        with torch.no_grad():
            preds = self.forward(batch, eval=True)
        transcriptions = self.transcriptions_from_forward_outputs(preds)
        return transcriptions

    @property
    def sampling_rate(self) -> int:
        return self.processor.feature_extractor.sampling_rate

    def get_modules(self) -> torch.nn.ModuleDict:
        """
        Returns a ModuleDict of trainable torch modules, with only one module
        which is Wav2Vec2ForCTC.
        """
        return torch.nn.ModuleDict({'model': self.model})

    @property
    def loss_type(self) -> Literal['ctc_loss', 'cross_entropy'] | None:
        return 'ctc_loss'

    @property
    def n_logits(self) -> float:
        """
        Such N that self.forward(...)['logits'].shape == (batch_size, N).

        Equals vocab_size for Wav2vec2.
        """
        return self.model.config.vocab_size

    def loss_from_logits(
        self, batch: dict[str, Any], logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates loss from logits in a differentiable way. This can be used to apply
        IRM-based methods. Follows the implementation in Wav2Vec2ForCTC.

        The batch should be already processed by .forward() function (it adds new fields).
        """
        input_lengths = self.model._get_feat_extract_output_lengths(
            batch['_attention_mask'].sum(-1)
        ).to(torch.long)

        # assuming that padded tokens are filled with -100
        # when not being attended to
        labels_mask = labels >= 0
        target_lengths = labels_mask.sum(-1)
        flattened_targets = labels.masked_select(labels_mask)

        # ctc_loss doesn't support fp16
        log_probs = torch.nn.functional.log_softmax(
            logits, dim=-1, dtype=torch.float32
        ).transpose(0, 1)

        with torch.backends.cudnn.flags(enabled=False):
            return torch.nn.functional.ctc_loss(
                log_probs,
                flattened_targets,
                input_lengths,
                target_lengths,
                blank=self.model.config.pad_token_id,
                reduction=self.model.config.ctc_loss_reduction,
                zero_infinity=self.model.config.ctc_zero_infinity,
            )
