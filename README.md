## Repository map

- `configs/` — all settings (`config.py`) and named experiments (`experiments.json`)
- `src/` — all project code: data, model, training, experiments
- `tests/` — correctness tests, run with `pytest -q`
- `notebooks/run_project.ipynb` — runs everything on Colab (only calls `main.py`)
- `main.py` — one entry point for all commands
- `results/` — small output files from experiments (see below)
- `docs/figures/` — figures made by our code + architecture diagram (used in report and slides)
- `docs/presentation/` — slides for final presentation
- `docs/report/` — final report (LaTeX)
- `docs/proposal/` — submitted proposal

## Date preprocess

The channel-wise mean and standard deviation were computed over
all pixels of the 50,000 original training images after scaling
pixel values to [0, 1].

- Mean (RGB): (0.4914, 0.4822, 0.4465)
- Std (RGB): (0.2470, 0.2435, 0.2616)

The values were verified using NumPy.

Dataset source: [Official CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

## Results folder

All small output files from our experiments are saved here. They are part of the
submission, so please commit them (but no checkpoints or datasets here!).
Don't edit these files by hand — they are always created by code.

### `results/ablation/`
Created by `python main.py ablate` (code: `src/experiments/ablation.py`).
- `<experiment>_seed<seed>.json` — one file per training run, for example
  `baseline_p4_seed0.json`. Inside: config, environment info, training history
  per epoch (train/val loss and accuracy), final test accuracy, number of params, train time.
- `summary.csv` — one row per experiment: mean and std of test accuracy over seeds,
  train time, number of params. This is the main table for the report.

If Colab disconnects, just run `ablate` again — runs that already have JSON file are skipped.

### `results/profiling/`
Created by `python main.py profile` (code: `src/experiments/profiling.py`).
- `profile.csv` — speed (ms per batch, images/s), peak GPU memory, params and FLOPs,
  for both patch sizes, for inference and for training step. Also hardware and batch size.

### `results/sanity/`
Created by `python main.py sanity` and `python main.py trace --experiment baseline_p4`.
- `overfit_curve.json` — loss per step when model overfits 32 images (shows training pipeline works).
- `shape_trace.csv` — tensor shape after each part of the model. Used for shape table in report.

Figures made from these files go to `docs/figures/`.