It is not a released tool, but only a snapshot of a private repository. I'm not adding documentation yet, however everything is self-explanatory.

To install pre-commit hooks:
```
pip install pre-commit
pre-commit install
```

To fine-tune ASR models:
```
python train.py --config=configs/base_config.py
```

A lot of useful information is saved:
- Loss/WER plots
- Model predictions
- Weight dynamics
- Training logs

The file `configs/base_config.py` contains all details such as:
- The model checkpoint to use (Wav2vec2 or Whisper)
- Training and evaluation datasets
- Training hyperparameters
- Path to save results

See example run in `example training run` folder.