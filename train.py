# ruff: noqa: E402, E731

import json
import logging
import shutil
import sys
import os
import itertools
import functools
from pathlib import Path
from typing import Any, Final

import numpy as np

import torch
from torch.utils.data import DataLoader

import transformers

from absl import app, flags
from ml_collections import config_flags

from src.dummy import DummyWrapper
from src.model import ModelWrapper
from src.tensor_tracker import MultiTensorTracker

sys.path.append('.')
from src.datasets import get_dataset
from src.whisper import WhisperWrapper
from src.wav2vec2 import Wav2vec2Wrapper
from src.metrics import EvalResults, calculate_wer
from src.optimizer import group_param_names_for_weight_decay
from src.train_plot import get_log_eval_indices, save_training_plot
from src.dataset_ops import (
    AddEnvInfo,
    infinitely_sample_from_dataset,
    infinitely_interleave,
    infinitely_collate,
    to_device,
    ChainTransforms,
    collate_to_lists,
)
from src.birm import (
    BayesianInvariantRiskMinimization_BayesFullbatch,
    BayesianInvariantRiskMinimization_BayesByVariance,
)

CONFIG = config_flags.DEFINE_config_file('config')
flags.mark_flag_as_required('config')


def main(_):
    config = CONFIG.value

    # base paths
    script_dir = Path.cwd().resolve()
    root_evals_dir = Path(config.saving.evals_dir).resolve()
    root_evals_dir.mkdir(exist_ok=True, parents=True)

    # paths for the current training run:
    name: Final[str] = (
        config.saving.name if config.saving.name != '' else f'{config.model.path}_tuned'
    )
    evals0_dir = root_evals_dir / config.model.path
    evals_dir = root_evals_dir / name
    shutil.rmtree(evals_dir, ignore_errors=True)
    evals_dir.mkdir(parents=True)
    get_evals_dir = lambda step: evals_dir / f'steps/step{step}'
    plot_path = evals_dir / 'plot.png'
    log_path = evals_dir / 'train.log'
    history_path = evals_dir / 'history.json'
    weights_tracker_dir = evals_dir / 'weights_tracker'

    # logging
    # we to this AFTER creating evals_dir to avoid errors
    logger = logging.getLogger()
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        force=True,
        handlers=[
            logging.FileHandler(str(log_path)),
            logging.StreamHandler(),
        ],
    )
    logger.info('Environment variables: ' + str(os.environ))
    logger.info('Config:\n' + str(config))
    transformers.utils.logging.set_verbosity(transformers.logging.WARNING)

    # seeds
    torch.manual_seed(config.seed)
    if config.device == 'cuda':
        torch.cuda.manual_seed_all(config.seed)

    # functions for debug copying and saving
    def debug_save(
        dir: Path,
        model_wrapper: ModelWrapper,
        batch: dict[str, Any],
        outputs: dict[str, Any],
        rng_state: torch.Tensor,
    ):
        dir.mkdir(parents=True, exist_ok=True)
        model_wrapper.save(dir / 'model', save_optimizer=True)
        torch.save(batch, dir / 'batch.pkl')
        torch.save(outputs, dir / 'outputs.pkl')
        torch.save(rng_state, dir / 'rng_state.pkl')

    def debug_save_steps(
        model_wrapper: ModelWrapper, batch: dict[str, Any], outputs: dict[str, Any]
    ):
        nonlocal config
        dir = evals_dir / 'failing_step'
        if config.analysis.on_fail_save_current_step:
            debug_save(dir, model_wrapper, batch, outputs, torch.get_rng_state())

    # model
    if (evals0_dir / 'config.json').is_file():
        os.chdir(root_evals_dir)
        logger.info(f'Loading model locally from {str(evals0_dir)}')
    else:
        logger.info('Loading model from Huggingface')

    if config.model.type == 'whisper':
        model_wrapper = WhisperWrapper(
            config.model.path,
            device=config.device,
            **config.whisper.kwargs,
        )
    elif config.model.type == 'wav2vec2':
        model_wrapper = Wav2vec2Wrapper(
            config.model.path,
            device=config.device,
            **config.wav2vec2.kwargs,
        )
    elif config.model.type == 'dummy':
        model_wrapper = DummyWrapper()
    os.chdir(script_dir)

    # datasets
    train_datasets = {
        dataset_and_split: get_dataset(
            *dataset_and_split.split(':', 1),
            load_unknown=False,
            sampling_rate=model_wrapper.sampling_rate,
            min_seconds=config.data.min_seconds,
            max_seconds=config.data.max_seconds,
            max_samples=config.data.train.max_samples,
            log_fn=logger.info,
        ).with_transform(
            ChainTransforms(AddEnvInfo(env_name=dataset_and_split, env_idx=i))
        )
        for i, dataset_and_split in enumerate(config.data.train.datasets)
    }

    if config.evaluation.enabled:
        eval_datasets = {
            dataset_and_split: get_dataset(
                *dataset_and_split.split(':', 1),
                load_unknown=False,
                sampling_rate=model_wrapper.sampling_rate,
                min_seconds=config.data.min_seconds,
                max_seconds=config.data.max_seconds,
                max_samples=config.data.eval.max_samples,
                log_fn=logger.info,
            )
            for dataset_and_split in (
                config.data.eval.datasets + config.data.train.datasets
                if config.data.eval.add_train_datasets
                else config.data.eval.datasets
            )
        }

    if config.exit_after_downloading:
        logger.info('Exiting after downloading')
        return

    # dataloaders
    train_sample_generator = infinitely_interleave(
        [infinitely_sample_from_dataset(dataset) for dataset in train_datasets.values()]
    )
    train_batch_generator = infinitely_collate(
        train_sample_generator,
        batch_size=config.data.train.batch_size,
        collate_fn=collate_to_lists,
    )
    if config.evaluation.enabled:
        eval_dataloaders = {
            dataset_name: DataLoader(
                dataset,
                batch_size=config.data.eval.batch_size,
                shuffle=False,
                collate_fn=collate_to_lists,
            )
            for dataset_name, dataset in eval_datasets.items()
        }

    # optimizer
    if (
        model_wrapper.optimizer is None  # optimizer not loaded from checkpoint
        or not config.model.load_optimizer_if_saved  # optimizer loaded but discarded
    ):
        params = (
            group_param_names_for_weight_decay(
                model_wrapper.get_modules(), config.optimizer.weight_decay
            )
            if config.optimizer.weight_decay != 0
            else model_wrapper.get_modules().parameters()
        )
        model_wrapper.optimizer = torch.optim.Adam(params, lr=config.optimizer.lr)
        logger.info('A new optimizer was created')
    else:
        logger.info(
            'The saved optimizer was loaded. Params will not be updated, except lr'
        )
        for g in model_wrapper.optimizer.param_groups:
            g['lr'] = config.optimizer.lr

    # birm
    if config.birm.enable:
        if config.birm.type == 'auto':
            if isinstance(model_wrapper, WhisperWrapper):
                birm_type = 'bayes_fullbatch'
            elif isinstance(model_wrapper, Wav2vec2Wrapper):
                birm_type = 'bayes_by_variance'
        else:
            birm_type = config.birm.type

        if birm_type == 'bayes_fullbatch':
            birm = BayesianInvariantRiskMinimization_BayesFullbatch(
                num_envs=len(train_datasets),
                num_classes=model_wrapper.n_logits,
                device=config.device,
            )
        elif birm_type == 'bayes_by_variance':
            birm = BayesianInvariantRiskMinimization_BayesByVariance(
                num_envs=len(train_datasets),
                num_classes=model_wrapper.n_logits,
                device=config.device,
            )
        birm_loss_history_for_scaling = []

    # evaluation and saving code
    history = []  # dict with losses for each training step

    if config.evaluation.enabled:
        evals = {  # dict from dataset_and_split to dict (step -> EvalResults)
            dataloader_name: {} for dataloader_name in eval_dataloaders
        }

        def evaluate_and_save(step: int, dataloader_name: str) -> EvalResults:
            dir = evals0_dir if step == 0 else get_evals_dir(step)
            dir.mkdir(parents=True, exist_ok=True)
            n_samples = config.data.eval.max_samples
            filepath = dir / f'eval_{dataloader_name}_{n_samples}samples.csv'
            if filepath.is_file():
                eval_results = EvalResults.load(filepath, load_arrays=False)
            else:
                logger.info(f'eval: {filepath}')
                dataloader = eval_dataloaders[dataloader_name]
                torch.manual_seed(0)
                eval_results = model_wrapper.evaluate(
                    dataloader,
                    return_loss=config.evaluation.calc_loss,
                    return_transcriptions=True,
                    loss_scale=config.training.loss_scale,
                )
                eval_results.wer_results = calculate_wer(
                    true_texts=eval_results.true_texts,  # type: ignore[arg-type]
                    pred_texts=eval_results.pred_texts,  # type: ignore[arg-type]
                )
                eval_results.save(filepath)
            return eval_results

        # evaluation steps
        if config.evaluation.steps.log_scale:
            eval_steps = get_log_eval_indices(
                log_delta=config.evaluation.steps.log_scale_base,
                min_delta=config.evaluation.steps.log_scale_min_delta,
                max_delta=config.evaluation.steps.every_n_steps,
                max_value=config.training.max_steps,
            )
        else:
            eval_steps = range(
                0,
                config.training.max_steps,
                config.evaluation.steps.every_n_steps,
            )
        logger.info(f'Evaluation steps: {eval_steps}')
    else:
        # steps for plotting only, since we have no evaluation
        eval_steps = range(0, config.training.max_steps, 10)

    # weights tracker
    if config.analysis.weights_tracker.enabled:
        weights_tracker = MultiTensorTracker(
            model_wrapper.get_modules().state_dict(),
            n_elements=config.analysis.weights_tracker.n_elements,
            n_elements_override={
                name.replace(':', '.'): value
                for name, value in config.analysis.weights_tracker.n_elements_override.items()
            },
        )

    # train loop
    for step in itertools.count():
        logger.info(f'step {step}')
        history.append({})

        # weight tracking
        if config.analysis.weights_tracker.enabled:
            weights_tracker.update(step)

        # evaluation
        if step in eval_steps:
            if config.evaluation.enabled:
                with open(history_path, 'w') as h:
                    json.dump(history, h)
                for dataloader_name in eval_dataloaders:
                    evals[dataloader_name][step] = evaluate_and_save(
                        step, dataloader_name
                    )
                # .clear_arrays() arrays in all but 0th step
                if config.saving.save_model and step > 0:
                    dir = get_evals_dir(step)
                    dir.mkdir(parents=True, exist_ok=True)
                    logger.info(
                        'Saving model and optimizer...'
                        if config.saving.save_optimizer
                        else 'Saving model...'
                    )
                    model_wrapper.save(dir, save_optimizer=config.saving.save_optimizer)
                    logger.info('Saved')

            # saving plot
            if step > 0:
                save_training_plot(
                    plot_path=plot_path,
                    history=history,
                    evals=evals if config.evaluation.enabled else {},
                    train_sets_names=list(train_datasets),
                )

            # saving tracked weights
            if config.analysis.weights_tracker.enabled:
                logger.info('Saving weights tracker...')
                weights_tracker.to_csvs(weights_tracker_dir)
                logger.info('Saved')

        if step == config.training.max_steps:
            break

        # forward
        batch = to_device(next(train_batch_generator), config.device)
        outputs = model_wrapper.forward(batch, eval=False)
        loss = config.training.loss_scale * torch.nanmean(outputs['loss'])
        history[-1]['loss'] = float(loss)
        logger.info(f'Prediction loss {float(loss):g}')
        total_loss = loss

        # debug checks
        if config.analysis.nan_logits:
            if torch.isnan(outputs['logits']).any():
                debug_save_steps(model_wrapper, batch, outputs)
                raise AssertionError('outputs[\'logits\'] are NaN')
        if config.analysis.nan_loss:
            if torch.isnan(loss):
                debug_save_steps(model_wrapper, batch, outputs)
                raise AssertionError('torch.nanmean(outputs[\'loss\']) is NaN')

        # birm
        if config.birm.enable and step >= config.birm.start_from_step:
            env_indices = torch.tensor(batch['env_idx']).to(config.device)
            if birm_type == 'bayes_fullbatch':
                birm_loss = birm.calc_loss(
                    logits=outputs['logits'],
                    labels=outputs['labels'],
                    env_indices=env_indices,
                    n_birm_samples=config.birm.samples_number,
                    loss_fn=functools.partial(model_wrapper.loss_from_logits, batch),
                )
            elif birm_type == 'bayes_by_variance':
                birm_loss = birm.calc_loss(
                    config.training.loss_scale * outputs['loss'],
                    env_indices,
                )
            if config.birm.scaling_type == 'relative':
                birm_loss_orig = float(birm_loss)
                birm_loss_history_for_scaling.append(birm_loss_orig)
                ratio = np.mean([h['loss'] for h in history[-10:]]) / np.mean(
                    birm_loss_history_for_scaling[-10:]
                )
                birm_loss *= ratio * config.birm.scale
                logger.info(
                    f'BIRM loss scaling ratio {ratio:g}'
                    f' ({birm_loss_orig:g} -> {float(birm_loss):g})'
                )
            elif config.birm.scaling_type == 'absolute':
                birm_loss *= config.birm.scale
            if config.analysis.nan_loss:
                if torch.isnan(birm_loss):
                    debug_save_steps(model_wrapper, batch, outputs)
                    raise AssertionError('BIRM loss is NaN')
            history[-1]['birm_loss'] = float(birm_loss)
            total_loss += birm_loss

            # debug checks
            if config.analysis.nan_loss:
                if torch.isnan(birm_loss):
                    debug_save_steps(model_wrapper, batch, outputs)
                    raise AssertionError('BIRM loss is NaN')

        # backward
        total_loss.backward()
        if config.training.gradient_clipping.enabled:
            torch.nn.utils.clip_grad_norm_(
                model_wrapper.get_modules().parameters(),
                config.training.gradient_clipping.max_norm,
            )

        # debug checks
        non_finite_grad_names = []
        if config.analysis.non_finite_grad:
            for _name, values in model_wrapper.model.named_parameters():
                if values.grad is not None:
                    if not torch.isfinite(values.grad).all():
                        non_finite_grad_names.append(_name)
            if len(non_finite_grad_names):
                msg = f'gradients contain inf or NaN: {non_finite_grad_names}'
                debug_save_steps(model_wrapper, batch, outputs)
                raise AssertionError(msg)

        # step
        model_wrapper.optimizer.step()
        model_wrapper.optimizer.zero_grad()

        logger.info(
            f'Max allocated memory: {torch.cuda.max_memory_allocated(0) / 2**30:.2f} GB'
        )


if __name__ == '__main__':
    app.run(main)
