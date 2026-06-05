# Image-GS heuristic

Predicts the number of Gaussians $N$ and training iterations for the
[Image-GS](https://github.com/NYU-ICL/image-gs) method, then trains it with
those parameters. Supports a pilot trial training and two target modes
(absolute PSNR threshold or relative tolerance vs per-image best PSNR).

## Models

The trained Random Forest pickles are not stored in the repository. Download
the variant you need from the
[`models-v1` release](https://github.com/michalzajac8/image-gs-heuristic/releases/tag/models-v1)
and place it next to `predict.py`:

```
# smallest, default (B_35dB, pilot only)
curl -L -O https://github.com/michalzajac8/image-gs-heuristic/releases/download/models-v1/heuristic_sota.pkl

# 30/35 dB, pilot + no-pilot
curl -L -O https://github.com/michalzajac8/image-gs-heuristic/releases/download/models-v1/heuristic_common.pkl

# all targets
curl -L -O https://github.com/michalzajac8/image-gs-heuristic/releases/download/models-v1/heuristic_full.pkl
```

Pick the smallest variant that contains the target you need:

| variant | size | classifiers inside | use when |
|---|---|---|---|
| `heuristic_sota.pkl` | 45 MB | `B_35dB_pilot` + `iter_model_pilot` | you only need the single best-performing config (35 dB target, pilot on) |
| `heuristic_common.pkl` | 112 MB | `B_30dB_*`, `B_35dB_*` (pilot + no-pilot) + both iter regressors | 30 / 35 dB targets with or without pilot |
| `heuristic_full.pkl` | 278 MB | all 20 N-classifiers + both iter regressors | any target A (1/2/3/5 dB) or B (25/28/30/32/35/38 dB) |

Variants are produced by lossless compression and subsetting of a single
uncompressed source model. Predictions on the kept (target, pilot) entries are
**bit-for-bit identical** across variants.

## What's here

```
predict.py                  CLI: extract features, predict, train
README.md                   this file
tools/build_variants.py     rebuilds the variants from the uncompressed source
```

## Install

1. Install Image-GS (provides the `model.py` and CUDA renderer used to train).
   Follow the upstream instructions:
   <https://github.com/NYU-ICL/image-gs>.
2. Make sure the Image-GS repo is importable. The simplest setup is to place
   these files inside the Image-GS repo root, or run `predict.py` from the
   repo root with `PYTHONPATH=.`
3. Extra Python deps used only by the heuristic:
   ```
   pip install opencv-python scikit-learn numpy joblib
   ```
4. Download at least one model pickle (see [Models](#models)).

## Run

```
python predict.py path/to/image.png
```

`predict.py` auto-picks the smallest matching pickle present
(`heuristic_sota.pkl` → `heuristic_common.pkl` → `heuristic_full.pkl`).
Override with `--models`:

```
python predict.py path/to/image.png --target-mode B --target-value 35 --device cuda:0
python predict.py path/to/image.png --target-mode A --target-value 2 --no-pilot --models heuristic_full.pkl
```

Output (default): `<input_stem>.pt`, a torch checkpoint containing the trained
Gaussian parameters, final PSNR / SSIM, predicted $N$ and steps, and image
features.

## Parameters

| Flag | Default | Values |
|---|---|---|
| `input` | required | path to a single image (PNG, JPG, etc.) |
| `--target-mode` | `B` | `A` (relative tolerance) or `B` (absolute PSNR) |
| `--target-value` | `35` | for `B`: `25, 28, 30, 32, 35, 38` dB; for `A`: `1, 2, 3, 5` dB |
| `--no-pilot` | off | disables the pilot step |
| `--output` | `<input_stem>.pt` | output checkpoint path |
| `--models` | auto | path to a heuristic pickle; auto-discovery if omitted |
| `--device` | `cuda:0` | torch device |
| `--dry-run` | off | predict only, do not train the final model |

## Rebuilding variants

```
python tools/build_variants.py --source heuristic_models.pkl
```

Reads the uncompressed source model (`heuristic_models.pkl`, not distributed
here) and emits `heuristic_full.pkl`, `heuristic_common.pkl`,
`heuristic_sota.pkl`. Pass `--compress 9` for a slightly smaller but
slower-to-load output. The published release assets were produced this way.

## Notes

- Pilot is a short trial training ($N=1000$, 400 steps, ~2-2.5 s on RTX 4000 Ada)
  whose final PSNR is the strongest single predictor of the optimal $N$.
- Predicted $N$ is one of 8 trained classes: `250, 500, 1000, 2000, 5000,
  10000, 20000, 40000`.
- Predicted steps are rounded to multiples of 100 and clamped to `[500, 10000]`.
- The models were trained on 6,564 images (DTD, DIV2K, Kodak) with 5-fold
  cross-validation on nearly 53,000 Image-GS runs.
- `predict.py` first tries `joblib.load` (handles the compressed variants),
  falls back to `pickle.load` (handles the uncompressed source).
</content>
</invoke>
