# Image-GS heuristic

Predicts the number of Gaussians $N$ and training iterations for the
[Image-GS](https://github.com/NYU-ICL/image-gs) method, then trains it with
those parameters.

## Install

1. Install Image-GS and make it importable (place these files in the Image-GS
   repo root, or run with `PYTHONPATH` pointing at it):
   <https://github.com/NYU-ICL/image-gs>.
2. Install the heuristic dependencies:
   ```
   pip install opencv-python scikit-learn numpy joblib
   ```
3. Download a model next to `predict.py` (other variants:
   `heuristic_common.pkl`, `heuristic_full.pkl` in the same
   [release](https://github.com/michalzajac8/image-gs-heuristic/releases/tag/models-v1)):
   ```
   curl -L -O https://github.com/michalzajac8/image-gs-heuristic/releases/download/models-v1/heuristic_sota.pkl
   ```

## Run

```
python predict.py path/to/image.png
python predict.py path/to/image.png --target-mode B --target-value 35 --device cuda:0
```

Writes `<input_stem>.pt`: the trained Gaussian parameters, final PSNR / SSIM,
predicted $N$ and steps, and image features.

## Parameters

| Flag | Default | Values |
|---|---|---|
| `input` | required | path to a single image (PNG, JPG, etc.) |
| `--target-mode` | `B` | `A` (relative tolerance) or `B` (absolute PSNR) |
| `--target-value` | `35` | `B`: `25, 28, 30, 32, 35, 38` dB; `A`: `1, 2, 3, 5` dB |
| `--no-pilot` | off | disable the pilot training step |
| `--output` | `<input_stem>.pt` | output checkpoint path |
| `--models` | auto | path to a model pickle; smallest present variant if omitted |
| `--device` | `cuda:0` | torch device |
| `--dry-run` | off | predict only, do not train |
