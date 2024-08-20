from tqdm import tqdm
import numpy as np
from pysnr import snr_signal
import pandas as pd


def count_snrs_and_energies(ds):
    """
    Count snr using https://github.com/psambit9791/pysnr
    and energy by standart formula
    then averaging the results

    Parameters
    ----------
    ds : HF dataset with chosen split

    Returns
    -------
    list
        The computed SNRs
    list
        The computed average energies by sample
    """
    snrs = []
    avg_energies = []
    for i in tqdm(range(len(ds))):
        if i in [2238, 2239, 2240, 3574, 3575, 3573]:
            continue
        signal = ds[i]['audio']['array']

        snr, _ = snr_signal(signal)
        snrs.append(snr)

        squared_signal = np.square(signal)
        average_energy = np.mean(squared_signal)
        avg_energies.append(average_energy)
    return snrs, avg_energies


def avg_to_csv(snrs, nrgs, dataset, split):
    """
    Write mean values of snrs and nrgs to csv with the specified dataset name and split for column 'dataset'

    Parameters
    ----------
    snrs : list
        snrs
    nrgs : list
        average energies
    dataset : str
        dataset name
    split : str
        dataset split
    """
    name = f'{dataset}_{split}'
    data = {
        'dataset': [name],
        'average snr': [np.mean(snrs)],
        'average energy': [np.mean(nrgs)],
    }
    df = pd.DataFrame(data)
    df.to_csv(f'{name}.csv', index=False)
