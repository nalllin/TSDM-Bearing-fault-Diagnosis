# TSDM-Bearing-fault-Diagnosis

Diffusion-based time-series modeling for bearing fault diagnosis. The project provides a 1D U-Net diffusion model, a dataset loader for bearing vibration signals stored in MATLAB `.mat` files, training utilities, and a small analysis script to compare real and synthetic signals.

## Project layout

- `models.py` — 1D U-Net (with residual and attention blocks) used for diffusion modeling. 
- `dataset.py` — `BearingSignalDataset` for loading/segmenting/normalizing `.mat` vibration data. 
- `train.py` — training loop, forward diffusion, and reverse diffusion sampler. 
- `analysis.py` — example script that loads a trained model and plots real vs. synthetic signals and envelope spectra. 
- `utils.py` — helpers for loading `.mat` files, segmentation, normalization, and plotting.

## Requirements

Install dependencies (example with pip):

```bash
pip install torch numpy scipy matplotlib
```

## Data expectations

The dataset loader expects a folder of `.mat` files containing bearing vibration signals. The loader looks for keys that include `_DE_time` (drive end signal); if not found, it falls back to the first non-metadata key.

Label assignment is based on filename patterns:

- `Normal` → Healthy
- `IR` → Inner race fault
- `OR` → Outer race fault
- `B` → Ball fault

The dataset selects files that contain `007` for faults and `Normal` for healthy signals (configurable via `use_normal`).

## Training

Update the `data_dir` in `train.py` to point to your `.mat` files, then run:

```bash
python train.py
```

Artifacts are saved to the `output/` directory (trained model weights and loss history). The default settings train for 250 epochs with a 1D U-Net and a diffusion schedule of 3000 steps.

## Sampling / analysis

The example analysis script loads a trained model and compares real vs. synthetic signals:

```bash
python analysis.py
```

Update the `model_path` and `data_dir` variables inside `analysis.py` as needed. The script saves plots (`real_vs_synthetic_*.png`, `env_spectrum_*.png`) for each class and visualizes time-domain signals and envelope spectra.

## Notes

- This repository uses flat files in the top-level directory; imports inside the scripts currently reference a `models` package. If you encounter import errors, run scripts as a module within a package layout or adjust imports to match your local structure.
- GPU acceleration is used automatically when CUDA is available.
