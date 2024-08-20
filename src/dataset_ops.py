from typing import Any
from collections.abc import Callable, Iterator
from dataclasses import dataclass
import itertools

import numpy as np
import torch
from datasets import Dataset


def collate_to_lists(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate samples from datasets.Dataset to a batch. Each field will be a list.
    """
    keys = set.union(*[set(sample) for sample in samples])
    return {key: [sample.get(key) for sample in samples] for key in keys}


class ChainTransforms:
    """
    Chain multiple transforms to use in dataset.set_transform().

    The following two examples should provide the same result, while the difference is
    that the latter is peformed on-the-fly and does not occupy more disk or memory.

    ```
    dataset = dataset.map(fn1).map(fn2)
    ```

    ```
    dataset.set_transform(ChainTransforms(fn1, fn2))
    ```

    TODO REDESIGN THIS SOLUTION. The `.set_transform` function works with batched inputs,
    and un-batch and re-batch a sample inside it is highly undesirable. It works in my case,
    but may work incorrectly in another cases, since batching may be performed in another
    way. Related issues:
    https://github.com/huggingface/datasets/issues/1776
    https://github.com/huggingface/datasets/issues/3385
    https://github.com/huggingface/datasets/issues/7095
    """

    def __init__(self, *functions):
        self.functions = functions

    def __call__(self, sample):
        """
        When using `.set_transform(fn)`, the function `fn` works with batched inputs,
        and then they are unbatched again.
        """
        # unbatching
        keys = list(sample)
        if len(keys) == 0:
            samples = [sample]
        else:
            n_samples = len(sample[keys[0]])
            samples = [{k: sample[k][i] for k in keys} for i in range(n_samples)]

        # processing
        for fn in self.functions:
            samples = [fn(s) for s in samples]

        # updating keys
        keys = list(samples[0])

        # batching
        for k in keys:
            sample[k] = [s[k] for s in samples]

        return sample


@dataclass
class FilterByDuration:
    """
    To use as a function in dataset.filter() for audio HF datasets.
    Filters audio samples by duration.
    """

    min_seconds: float | None = None
    max_seconds: float | None = None

    def __call__(self, sample: dict[str, Any]) -> bool:
        duration = len(sample['audio']['array'])
        sample_rate = sample['audio']['sampling_rate']

        if self.min_seconds is not None and duration < sample_rate * self.min_seconds:
            return False

        if self.max_seconds is not None and duration >= sample_rate * self.max_seconds:
            return False

        return True


# def expose_audio_fields(sample: dict[str, Any]) -> dict[str, Any]:
#     """
#     To use as a function in dataset.map() for audio HF datasets.
#     Adds several fields to each sample of:
#     - Adds 'path' feature (extracts it from 'audio' feature)
#     - Adds 'audio_length' feature (audio length in seconds)
#     The intended usage is to further save these fields, along with model predictions.
#     """
#     sample['path'] = sample['audio']['path']
#     sample['audio_length'] = (
#         len(sample['audio']['array']) / sample['audio']['sampling_rate']
#     )
#     return sample


@dataclass
class AddEnvInfo:
    """Add information about environment to each sample for IRM/BIRM."""

    env_name: str
    env_idx: int

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        sample['env_name'] = self.env_name
        sample['env_idx'] = self.env_idx
        return sample


def infinitely_sample_from_dataset(dataset: Dataset, seed: int | None = None):
    """An infinite sample generator that reshuffles the dataset each epoch (a pass over the whole dataset).

    Example:
        dataset = Dataset.from_dict({'field': ['a', 'b', 'c']})
        iterator = infinitely_sample_from_dataset(dataset)
        for _ in range(9):
            print(next(iterator)['field'], end=' ')
        >>> a c b c b a c a b
    """
    seed_generator = np.random.default_rng(seed)
    while True:
        new_seed = seed_generator.integers(0, 2**32 - 1)
        yield from dataset.shuffle(seed=new_seed)


def infinitely_interleave(
    sample_generators: list[Iterator], repeats: list[int] | None = None
):
    """An infinite generator that interleaves between two generators.

    If repeats is specified, will sample repeats[i] samples at once from i-th generator.

    Since this function does not use randomness, samples from all generators are more evenly
    distributed. For example, if using infinitely_interleave with 2 sample generators, and then
    infinitely_collate with batch_size=2, then there is there is a guarantee that each batch
    will contain examples from both sample generators.

    Example:
        dataset1 = Dataset.from_dict({'field': ['a', 'b', 'c']})
        dataset2 = Dataset.from_dict({'field': ['0', '1']})
        iter1 = infinitely_sample_from_dataset(dataset1)
        iter2 = infinitely_sample_from_dataset(dataset2)
        iterator = infinitely_interleave([iter1, iter2], repeats=[1, 3])
        for _ in range(12):
            print(next(iterator)['field'], end=' ')
        >>> c 1 0 1 b 0 0 1 a 1 0 0
    """
    if repeats is None:
        repeats = [1] * len(sample_generators)
    for it, n_repeats in itertools.cycle(zip(sample_generators, repeats)):
        for _ in range(n_repeats):
            yield next(it)


def infinitely_collate(
    sample_generator: Iterator, batch_size: int, collate_fn: Callable
):
    """Performs batching of infinite sample generator.

    Example:
        dataset1 = Dataset.from_dict({'field': [-1, -2, -3]})
        dataset2 = Dataset.from_dict({'field': [0, 1]})
        iter1 = infinitely_sample_from_dataset(dataset1)
        iter2 = infinitely_sample_from_dataset(dataset2)
        iterator = infinitely_interleave([iter1, iter2])
        collated = infinitely_collate(iterator, batch_size=2, collate_fn=DefaultDataCollator())
        for _ in range(4):
            print(next(collated)['field'], end=' ')
        >>> tensor([-1,  1]) tensor([-3,  0]) tensor([-2,  0]) tensor([-1,  1])
    """
    incomplete_batch = []
    for sample in sample_generator:
        incomplete_batch.append(sample)
        if len(incomplete_batch) == batch_size:
            yield collate_fn(incomplete_batch)
            incomplete_batch.clear()


def to_device(sample: dict[str, Any], device: str):
    """
    Send all fields of torch.Tensor type to device, such as 'cuda'
    """
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            sample[key] = value.to(device)
    return sample
