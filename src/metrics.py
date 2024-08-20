from __future__ import annotations

from typing import Any
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import nltk
import numpy as np
import pandas as pd


@dataclass
class LossResults:
    losses: np.ndarray | None = None
    loss: float | None = None
    loss_std: float | None = None

    def clear_arrays(self):
        self.losses = None


@dataclass
class WERResults:
    wer: float | None = None
    wer_std: float | None = None
    wers: list[float] | None = None
    distances: list[int] | None = None
    lengths: list[int] | None = None

    def clear_arrays(self):
        self.wers = None
        self.distances = None
        self.lengths = None


@dataclass
class EvalResults:
    """
    Represents eval results, given some model predictions and ground truth texts.

    The results can be saved to comma-separated csv, when the first row after header
    is a special row containing metrics as json-encoded string. Example:

    [loss=1 wer=2]
    true_texts,pred_texts,wers
    a b,a,0.5
    x y,x y,0.0

    Saved results can also be loaded with pd.read_csv with discarding the first row:
    `pd.read_csv(saved_evals, skiprows=1)`.
    """

    true_texts: list[str] | None = None
    pred_texts: list[str] | None = None
    wer_results: WERResults | None = None
    loss_results: LossResults | None = None

    def save(self, path: str | Path):
        values = {}
        if self.loss_results is not None:
            values['loss'] = self.loss_results.loss
            values['loss_std'] = self.loss_results.loss_std
        if self.wer_results is not None:
            values['wer'] = self.wer_results.wer
            values['wer_std'] = self.wer_results.wer_std
        values_string = ' '.join(
            [f'{k}={v}' for k, v in values.items() if v is not None]
        )  # only floats and ints
        columns: dict[str, Any] = {}
        if self.true_texts is not None:
            columns['true_texts'] = self.true_texts
        if self.pred_texts is not None:
            columns['pred_texts'] = self.pred_texts
        if self.wer_results is not None:
            if self.wer_results.distances is not None:
                columns['distances'] = self.wer_results.distances
            if self.wer_results.lengths is not None:
                columns['lengths'] = self.wer_results.lengths
            if self.wer_results.wers is not None:
                columns['wers'] = self.wer_results.wers
        csv_string = pd.DataFrame(columns).to_csv(sep=',', header=True, index=False)
        with open(path, 'w') as h:
            h.write('[' + values_string + ']\n' + csv_string)

    @classmethod
    def load(cls, path: str | Path, load_arrays: bool = True) -> EvalResults:
        with open(path) as h:
            values_string = h.readline().strip('\n')
        values = dict([x.split('=') for x in values_string[1:-1].split()])
        results = cls()
        results.loss_results = LossResults()
        results.wer_results = WERResults()
        if 'loss' in values:
            results.loss_results.loss = float(values['loss'])
        if 'loss_std' in values:
            results.loss_results.loss_std = float(values['loss_std'])
        if 'wer' in values:
            results.wer_results.wer = float(values['wer'])
        if 'wer_std' in values:
            results.wer_results.wer_std = float(values['wer_std'])
        if load_arrays:
            df = pd.read_csv(path, sep=',', header=0, skiprows=1)
            if 'true_texts' in df:
                results.true_texts = df['true_texts'].astype(str).tolist()
            if 'pred_texts' in df:
                results.pred_texts = df['pred_texts'].astype(str).tolist()
            if 'wers' in df:
                results.wer_results.wers = df['wers'].astype(float).tolist()
            if 'distances' in df:
                results.wer_results.distances = df['distances'].astype(int).tolist()
            if 'lengths' in df:
                results.wer_results.lengths = df['lengths'].astype(int).tolist()
        return results

    def clear_arrays(self):
        self.true_texts = None
        self.pred_texts = None
        if self.loss_results is not None:
            self.loss_results.clear_arrays()
        if self.wer_results is not None:
            self.wer_results.clear_arrays()


def levenshtein(seq1: Sequence[Any], seq2: Sequence[Any]) -> int:
    """
    Calculates the Levenshtein distance for pair of sequences.

    This function will consider array elements as tokens that may be either equal
    or different. The Levenshtein distance is how much editing operations (delete,
    insert, replace) we should to do transform the first list to the second list
    (or vice versa).

    Example:
        levenshtein('abcde', 'abed')
        >>> 2
        levenshtein(['a', 'xyz', 'c', 'def', 'e'], ['a', 'xyz', 'e', 'def'])
        >>> 2

    NOTE: should actually be equal to nltk.edit_distance, TODO check.
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y), dtype=np.int32)
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y
    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    int(matrix[x - 1, y]) + 1,
                    int(matrix[x - 1, y - 1]),
                    int(matrix[x, y - 1]) + 1,
                )
            else:
                matrix[x, y] = min(
                    int(matrix[x - 1, y]) + 1,
                    int(matrix[x - 1, y - 1]) + 1,
                    int(matrix[x, y - 1]) + 1,
                )
    return int(matrix[size_x - 1, size_y - 1])


def calculate_metrics_std(
    values: np.ndarray, weights: np.ndarray, iters: int = 1000
) -> float:
    subsample_scores = []
    for _ in range(1000):
        indices = np.random.choice(len(values), len(values))
        score = np.average(values[indices], weights=weights[indices])
        subsample_scores.append(score)

    return np.std(subsample_scores)


def calculate_wer(
    *,  # no positional args to make sure not to confise true and pred
    true_texts: list[str],
    pred_texts: list[str],
    ignore_punctuation_and_capitalization: bool = True,
    clip: bool = True,
) -> WERResults:
    """
    Calculates average word error rate (WER) for list of predicted and true texts.

    Average WER is a weighted sum of WERs on all samples, where weights are lengths
    of true texts. So, long texts have larger impact on average WER than short texts.
    If all true texts are empty, weights on averaging are uniform.

    WER on sample is a word-based Levenshtein distance between predicted and true
    texts, divided by the length of true text. If true text is empty, sample's WER
    == 0 if predicted text is empty, otherwise 1.

    If clip == True, mean WER and std WER will be calculated after clipping per-sample
    WER between 0 and 1, to reduce influence of outliers. However, the returned per-sample
    WER is not clipped.

    Example:
        calculate_wer(true_texts=['Attention is all youneed'],
              pred_texts=['Attention is all you need'])['mean']
        >>> 0.5
    """
    distances_list = []
    lengths_list = []
    for pred_txt, true_txt in zip(pred_texts, true_texts):
        pred_txt_tokenized = nltk.wordpunct_tokenize(pred_txt)
        true_txt_tokenized = nltk.wordpunct_tokenize(true_txt)
        if ignore_punctuation_and_capitalization:
            pred_txt_tokenized = [x.lower() for x in pred_txt_tokenized if x.isalnum()]
            true_txt_tokenized = [x.lower() for x in true_txt_tokenized if x.isalnum()]

        distances_list.append(levenshtein(pred_txt_tokenized, true_txt_tokenized))
        lengths_list.append(len(true_txt_tokenized))

    distances = np.array(distances_list)
    lengths = np.array(lengths_list)

    wers = distances / lengths

    # empty predicted for empty truth
    wers[(distances == 0) & (lengths == 0)] = 0
    # non-empty predicted for empty truth
    wers[(distances > 0) & (lengths == 0)] = 1

    weights = lengths if (sum(lengths) > 0) else np.ones(len(wers))

    return WERResults(
        wer=np.average(np.clip(wers, 0, 1) if clip else wers, weights=weights),
        wer_std=calculate_metrics_std(
            np.clip(wers, 0, 1) if clip else wers, weights=weights
        ),
        wers=wers,
        distances=distances,
        lengths=lengths,
    )
