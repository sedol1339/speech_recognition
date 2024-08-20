import os
from pathlib import Path
from collections.abc import Callable

from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from datasets.features import Audio

from src.dataset_ops import FilterByDuration


def get_hf_token() -> str | None:
    """
    Get HF token from environment variable HF_TOKEN or file ~/hf_token.txt.
    """
    if 'HF_TOKEN' in os.environ:
        return os.environ['HF_TOKEN']
    path = Path.home() / 'hf_token.txt'
    if path.is_file():
        return open(path).read().rstrip()
    else:
        return None


# todo change this to decorators?
KNOWN_DATASETS = {
    'podlodka': lambda: load_dataset('bond005/podlodka_speech'),
    'rulibrispeech': lambda: load_dataset('bond005/rulibrispeech'),
    'taiga_speech_v2': lambda: load_dataset('bond005/taiga_speech_v2'),
    'wikipedia': lambda: load_dataset('rmndrnts/wikipedia_asr'),
    'lena_dataset': lambda: load_dataset('rmndrnts/lena_dataset'),
    'golos_farfield': lambda: load_dataset('bond005/sberdevices_golos_100h_farfield'),
    'tuberculosis': lambda: load_dataset('tmp/asr_med_ru_tuberculosis_hf'),
    'sova_ruyoutube': lambda: DatasetDict(
        {'val': load_from_disk('tmp/sova_ruyoutube_val')}
    ),
    'sova_rudevices': lambda: load_dataset('bond005/sova_rudevices'),
    'resd': (
        lambda: load_dataset('Aniemore/resd_annotated')
        .rename_column('text', 'transcription')
        .rename_column('speech', 'audio')
    ),
    'fleurs': (
        lambda: load_dataset('google/fleurs', name='ru_ru', trust_remote_code=True)
        .remove_columns('transcription')
        .rename_column('raw_transcription', 'transcription')
    ),
    'speech_massive': lambda: load_dataset(
        'FBK-MT/Speech-MASSIVE', name='ru-RU'
    ).rename_column('utt', 'transcription'),
    'speech_massive_test': lambda: load_dataset(
        'FBK-MT/Speech-MASSIVE-test', name='ru-RU', token=get_hf_token()
    ).rename_column('utt', 'transcription'),
    'common_voice_17_0': lambda: load_dataset(
        'mozilla-foundation/common_voice_17_0',
        name='ru',
        token=get_hf_token(),
        trust_remote_code=True,
    ).rename_column('sentence', 'transcription'),
}


def get_dataset(
    dataset_name: str,
    split: str,
    load_unknown: bool = False,
    sampling_rate: int | None = None,
    min_seconds: int | None = None,
    max_seconds: int | None = None,
    max_samples: int | None = None,
    log_fn: Callable = print,
) -> Dataset:
    """
    Load one of:
    1) A known dataset by it's name, as specified in KNOWN_DATASETS dict.
    2) An unknown dataset (if `load_unknown=True`) with `load_dataset(dataset_name)`.

    It is guaranteed (for known datasets only) that the 'audio' field will be Audio feature
    with specified sampling_rate, and the 'transcription' field is a text transcription.

    Depending on parameters, is able to filter by duration and subsample.
    """
    log_fn(f'Loading dataset {dataset_name}:{split}')
    # retrieving dataset dict
    if dataset_name in KNOWN_DATASETS:
        dataset_dict = KNOWN_DATASETS[dataset_name]()
    else:
        assert (
            load_unknown
        ), f'Dataset {dataset_name} is unknown, pass `load_unknown=True` to load'
        dataset_dict = load_dataset(dataset_name)
        assert isinstance(dataset_dict, DatasetDict)

    # retrieving split
    assert (
        split in dataset_dict
    ), f'cannot get split {split} for {dataset_name}={dataset_dict}'
    dataset = dataset_dict[split]

    # sampling rate
    if sampling_rate is not None:
        dataset = dataset.cast_column('audio', Audio(sampling_rate=sampling_rate))

    # filtering by duration
    if min_seconds is not None or max_seconds is not None:
        dataset = dataset.filter(
            FilterByDuration(min_seconds=min_seconds, max_seconds=max_seconds),
            desc='filtering by duration',
            num_proc=4,
            writer_batch_size=100,  # to avoid OOM
        )

    # getting subset
    if max_samples is not None and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=0).select(range(max_samples))

    return dataset
