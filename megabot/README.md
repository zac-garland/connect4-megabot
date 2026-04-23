# Connect 4 Project Workspace

## First Step

Run `notebooks/01_import_models.ipynb` before continuing any other work.

That notebook uses the shared model hub in `src/connect4_model_hub.py` to:
- pull or copy the runnable Connect 4 models into one place
- standardize them all as full `.keras` models
- write `models/manifest.csv` so later notebooks know what exists and how to load it

Do not start new benchmarking or training work until the model hub import step has been run.

## Main Workflow

1. Run `notebooks/01_import_models.ipynb`.
2. Confirm the standardized models exist in `models/`.
3. Run `notebooks/02_round_robin.ipynb` if you want the 100-game round robin.
4. Use the standardized `.keras` files in `models/` for future training notebooks.

## Structure

- `notebooks/01_import_models.ipynb`
  Imports the runnable models and standardizes them into `models/`.
- `notebooks/02_round_robin.ipynb`
  Loads the standardized models and runs the 100-game round robin.
- `models/`
  Standardized full `.keras` models plus `manifest.csv`.
- `data/`
  Benchmark outputs.
- `src/connect4_model_hub.py`
  Shared loading, encoding, and benchmark logic.
- `docs/project 3 overview - connect 4 new.pdf`
  Preserved project overview PDF.

## Important Note On Model Format

The models are stored as full saved Keras models, not just weights.
That is intentional: weights alone are not enough unless the exact architecture and custom layers are also preserved.

## Current Best Model

`dean_cnn` is the current best benchmark model and should be treated as `M1` for the policy-gradient phase.
