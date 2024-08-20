from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
from matplotlib.scale import SymmetricalLogScale
import numpy as np

from src.metrics import EvalResults


def get_log_eval_indices(
    log_delta: float = 1.3,
    min_delta: int = 3,
    max_delta: int = 500,
    max_value: int = 10_000,
) -> list[int]:
    """
    A custom way to define validation steps. At the start of the training, we
    want valudation to perform in log scale, but delta between validations
    is no less than `min_delta` and no more than `max_delta`.
    ```
    """
    steps = [0]
    while True:
        last_step = steps[-1]
        delta = round(last_step * log_delta) - last_step
        delta = np.clip(delta, min_delta, max_delta)
        next_step = last_step + delta
        if next_step >= max_value:
            break
        steps.append(int(next_step))
    return steps

    # n_log_steps = np.ceil(np.log(max_value) / np.log(log_delta)).astype(int)
    # log_steps = np.array(sorted(set(np.round(log_delta ** np.arange(n_log_steps)).astype(int))))
    # transition_idx = np.where(np.diff(log_steps) > max_delta)[0].min()
    # uniform_steps = np.arange(
    #     start=log_steps[transition_idx] + max_delta,
    #     stop=max_value,
    #     step=max_delta,
    # )
    # return list(np.concatenate([
    #     log_steps[:transition_idx + 1],
    #     uniform_steps
    # ]))


def save_training_plot(
    plot_path: str | Path,
    history: list[dict[str, Any]],
    evals: dict[str, dict[int, EvalResults]],
    figsize: tuple[float, float] = (20, 10),
    train_sets_names: list[str] | None = None,
):
    """
    Plots the results obtained by train.py.

    `history[i]` is an array of values (such as "loss") for i-th training step. So,
    the total number of training steps is `len(history)`.
    `evals[name][step]` is evaluation results obtained at the specified step for
    the specified validation dataset name (such as "rulibrispeech:val").

    If `train_sets_name` are specified, the corresponding keys in `evals` will
    be plotted with "C0" color. Other keys will be plotted with colors from "C1"
    to "C9".
    """
    all_fields = sorted(set.union(*[set(h) for h in history]))
    if 'loss' in all_fields:  # to the first place
        all_fields.remove('loss')
        all_fields = ['loss'] + all_fields

    # n eval subplots
    ls = ['solid', 'dashed', 'dotted', 'dashdot']
    if train_sets_names is not None:
        train_sets_names = sorted([name for name in train_sets_names if name in evals])
    else:
        train_sets_names = []
    test_sets_names = sorted(set(evals) - set(train_sets_names))
    n_subplots = max(
        1,
        # no more than len(ls) train plots on a single subplot
        np.ceil(len(train_sets_names) / len(ls)).astype(int),
        # no more than 9 non-train plots on a single subplot
        np.ceil(len(test_sets_names) / 9).astype(int),
    )

    _, axs = plt.subplots(ncols=n_subplots * 2, figsize=figsize)
    axs_loss = axs[:n_subplots]
    axs_wer = axs[n_subplots:]

    # draw train
    # https://matplotlib.org/stable/api/markers_api.html
    markers = ['o', 'X', 's', 'v']
    for i, name in enumerate(all_fields):
        values = np.array([h.get(name) for h in history], dtype=float)
        steps = np.where(~np.isnan(values))[0]
        axs_loss[0].scatter(
            steps,
            values[steps],
            label=name,
            color='slateblue',
            marker=markers[i % len(markers)],
            zorder=-1,
            alpha=0.4,
            s=10,
        )
        axs_loss[0].plot(
            steps,
            values[steps],
            color='slateblue',
            zorder=-1,
            alpha=0.4,
            lw=0.5,
        )

    # eval styles
    styles = {
        # lines for train datasets are C0 with different linestyle
        **{
            name: {
                'color': 'C0',
                'linestyle': ls[i % len(ls)],
                'ax_idx': i // len(ls),
            }
            for i, name in enumerate(train_sets_names)
        },
        # lines for train datasets are C1...C9 with solid linestyle
        **{
            name: {
                'color': f'C{1 + i % 9}',
                'linestyle': 'solid',
                'ax_idx': i // 9,
            }
            for i, name in enumerate(test_sets_names)
        },
    }

    # draw eval
    for name, style in styles.items():
        evals_for_name = evals[name]
        eval_steps = list(evals_for_name.keys())

        wer = np.array([
            (res.wer_results.wer if res.wer_results else None)
            for res in evals_for_name.values()
        ]).astype(float)
        wer_std = np.array([
            (res.wer_results.wer_std if res.wer_results else None)
            for res in evals_for_name.values()
        ]).astype(float)
        axs_wer[style['ax_idx']].plot(
            eval_steps,
            wer,
            label=f'Eval WER: {name}',
            color=style['color'],
            linestyle=style['linestyle'],
        )
        axs_wer[style['ax_idx']].fill_between(
            eval_steps,
            wer - wer_std,
            wer + wer_std,
            color=style['color'],
            alpha=0.2,
        )

        loss = np.array([
            (res.loss_results.loss if res.loss_results else None)
            for res in evals_for_name.values()
        ]).astype(float)
        loss_std = np.array([
            (res.loss_results.loss_std if res.loss_results else None)
            for res in evals_for_name.values()
        ]).astype(float)
        axs_loss[style['ax_idx']].plot(
            eval_steps,
            loss,
            label=f'Eval loss: {name}',
            color=style['color'],
            linestyle=style['linestyle'],
        )
        axs_loss[style['ax_idx']].fill_between(
            eval_steps,
            loss - loss_std,
            loss + loss_std,
            color=style['color'],
            alpha=0.2,
        )

    for ax in axs:
        ax.set_xscale(SymmetricalLogScale(ax, base=10, linthresh=10))
        ax.legend()
    for ax in axs_loss:
        ax.set_yscale('log')

    plt.tight_layout()

    plt.savefig(plot_path)
    plt.close()
