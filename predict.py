#!/usr/bin/env python3
"""
Image-GS heuristic: predict N and number of steps for an input image,
then train Image-GS with the predicted parameters.

Modes:
  - Target B (default): absolute PSNR threshold. Available: 25, 28, 30, 32, 35, 38 dB.
  - Target A: relative tolerance vs the per-image best PSNR. Available: 1, 2, 3, 5 dB.

Pilot (default ON): runs a short trial training (N=1000, 400 steps, ~2-2.5 s on RTX 4000 Ada)
and uses its PSNR as an additional feature. Disable with --no-pilot.
"""

import argparse
import os
import pickle
import sys
import time
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent

# Smallest variant first; predict.py auto-picks the first one present.
VARIANT_FILES = [
    "heuristic_sota.pkl",
    "heuristic_common.pkl",
    "heuristic_full.pkl",
    "heuristic_models.pkl",
]


def find_default_models():
    for name in VARIANT_FILES:
        p = SCRIPT_DIR / name
        if p.exists():
            return p
    return SCRIPT_DIR / VARIANT_FILES[0]

N_OPTIONS = [250, 500, 1000, 2000, 5000, 10000, 20000, 40000]
FEATURES = [
    "width", "height", "num_pixels", "file_size_bytes",
    "shannon_entropy", "mean_gradient", "hf_energy_ratio",
    "edge_density", "laplacian_var", "color_entropy",
]
TARGET_B_AVAILABLE = [25, 28, 30, 32, 35, 38]
TARGET_A_AVAILABLE = [1, 2, 3, 5]

PILOT_N = 1000
PILOT_STEPS = 400


# ---------------------------------------------------------------------------
# Image features
# ---------------------------------------------------------------------------

def _shannon_entropy(img_gray):
    hist, _ = np.histogram(img_gray, bins=256, range=(0, 256))
    hist = hist[hist > 0]
    p = hist / hist.sum()
    return float(-np.sum(p * np.log2(p)))


def _mean_gradient(img_gray):
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.sqrt(gx ** 2 + gy ** 2).mean())


def _hf_energy_ratio(img_gray, cutoff_fraction=0.25):
    f = np.fft.fft2(img_gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift) ** 2
    rows, cols = img_gray.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt((y - crow) ** 2 + (x - ccol) ** 2)
    cutoff = cutoff_fraction * np.sqrt(crow ** 2 + ccol ** 2)
    hf = magnitude[dist > cutoff].sum()
    total = magnitude.sum()
    return float(hf / total) if total > 0 else 0.0


def _edge_density(img_gray):
    edges = cv2.Canny(img_gray, 100, 200)
    return float((edges > 0).sum() / edges.size)


def _laplacian_var(img_gray):
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())


def _color_entropy(img_bgr):
    nc = min(3, img_bgr.shape[2]) if len(img_bgr.shape) == 3 else 1
    entropies = []
    for c in range(nc):
        ch = img_bgr[:, :, c] if len(img_bgr.shape) == 3 else img_bgr
        hist, _ = np.histogram(ch, bins=256, range=(0, 256))
        hist = hist[hist > 0]
        p = hist / hist.sum()
        entropies.append(float(-np.sum(p * np.log2(p))))
    return float(np.mean(entropies))


def extract_features(image_path):
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")
    if len(img_bgr.shape) == 2:
        img_gray = img_bgr
        img_bgr = np.expand_dims(img_bgr, axis=2)
    else:
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    metrics = {
        "width": w, "height": h, "num_pixels": h * w,
        "file_size_bytes": os.path.getsize(str(image_path)),
        "shannon_entropy": _shannon_entropy(img_gray),
        "mean_gradient": _mean_gradient(img_gray),
        "hf_energy_ratio": _hf_energy_ratio(img_gray),
        "edge_density": _edge_density(img_gray),
        "laplacian_var": _laplacian_var(img_gray),
        "color_entropy": _color_entropy(img_bgr),
    }
    return metrics, np.array([metrics[f] for f in FEATURES])


# ---------------------------------------------------------------------------
# Image-GS args
# ---------------------------------------------------------------------------

def _make_args(image_path, n, max_steps, device):
    return Namespace(
        batch_mode=True, eval=False, metric_models=None, data_root=".",
        input_path=str(image_path), device=device, seed=123,
        num_gaussians=n,
        init_scale=5.0, topk=10, disable_topk_norm=False,
        disable_inverse_scale=False, disable_color_init=False,
        init_mode="gradient", init_random_ratio=0.3, smap_filter_size=20,
        quantize=False, pos_bits=32, scale_bits=32, rot_bits=32, feat_bits=32,
        l1_loss_ratio=1.0, l2_loss_ratio=0.0, ssim_loss_ratio=0.1,
        disable_tiles=False, max_steps=max_steps,
        pos_lr=5e-4, scale_lr=2e-3, rot_lr=2e-3, feat_lr=5e-3,
        disable_lr_schedule=False, decay_ratio=10.0, check_decay_steps=1000,
        max_decay_times=1, decay_threshold=1e-3,
        disable_prog_optim=True, initial_ratio=1.0,
        add_steps=500, add_times=0, post_min_steps=0,
        log_root="results", log_dir="/tmp/image_gs_predict_logs",
        log_level="WARNING", save_image_format="png", save_plot_format="png",
        vis_gaussians=False, eval_steps=100,
        save_image_steps=999999, save_ckpt_steps=999999,
        ckpt_file=None, downsample=False, downsample_ratio=1.0,
        gamma=1.0, compute_complexity=False,
    )


def _train(image_path, n, max_steps, device):
    from model import GaussianSplatting2D
    args = _make_args(image_path, n, max_steps, device)
    model = GaussianSplatting2D(args)
    result = model.optimize_batch()
    state = {
        "step": result["final_step"],
        "psnr": result["final_psnr"],
        "ssim": result["final_ssim"],
        "lpips": result["final_lpips"],
        "flip": result["final_flip"],
        "msssim": result["final_msssim"],
        "bytes": result["num_bytes"],
        "time": result["total_time"],
        "num_gaussians": result["final_num_gaussians"],
        "state_dict": model.state_dict(),
    }
    del model
    torch.cuda.empty_cache()
    return result, state


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _model_key(target_mode, target_value, pilot):
    suffix = "pilot" if pilot else "no_pilot"
    return f"{target_mode}_{int(target_value)}dB_{suffix}"


def _round_steps(pred):
    return int(max(500, min(10000, round(pred / 100) * 100)))


def predict_n_and_steps(models, fv, target_mode, target_value, pilot, pilot_psnr=None):
    key = _model_key(target_mode, target_value, pilot)
    if key not in models["n_models"]:
        raise ValueError(
            f"No model '{key}' in this pickle. "
            f"Available: {sorted(models['n_models'].keys())}.\n"
            f"Use a larger variant (heuristic_full.pkl) or change --target-mode/--target-value."
        )
    iter_key = "iter_model_pilot" if pilot else "iter_model_no_pilot"
    if iter_key not in models:
        raise ValueError(
            f"This pickle does not contain '{iter_key}'. "
            f"Use a variant that includes it (heuristic_full.pkl or heuristic_common.pkl), "
            f"or flip --no-pilot accordingly."
        )

    if pilot:
        X_n = np.append(fv, pilot_psnr).reshape(1, -1)
    else:
        X_n = fv.reshape(1, -1)
    n = int(models["n_models"][key].predict(X_n)[0])

    if pilot:
        X_iter = np.append(fv, [n, pilot_psnr]).reshape(1, -1)
    else:
        X_iter = np.append(fv, n).reshape(1, -1)
    steps = _round_steps(models[iter_key].predict(X_iter)[0])
    return n, steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Image-GS heuristic: pick N and steps, then train.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", help="Path to the input image.")
    p.add_argument("--target-mode", choices=["A", "B"], default="B",
                   help="A: relative tolerance vs per-image best PSNR. B: absolute PSNR threshold. (default: B)")
    p.add_argument("--target-value", type=float, default=35.0,
                   help="For B: target PSNR in dB (25, 28, 30, 32, 35, 38). For A: tolerance in dB (1, 2, 3, 5). (default: 35)")
    p.add_argument("--no-pilot", action="store_true",
                   help="Disable the pilot training step. Pilot is on by default.")
    p.add_argument("--output", default=None,
                   help="Output file path. Default: <input_stem>.pt next to this script.")
    p.add_argument("--models", default=None,
                   help="Path to a heuristic pickle (joblib-compressed or raw). "
                        "Default: auto-pick the smallest present file from "
                        "heuristic_sota.pkl / heuristic_common.pkl / heuristic_full.pkl / heuristic_models.pkl.")
    p.add_argument("--device", default="cuda:0",
                   help="Torch device (default: cuda:0).")
    p.add_argument("--dry-run", action="store_true",
                   help="Predict N and steps only, do not train.")
    return p.parse_args()


def main():
    args = parse_args()

    pilot = not args.no_pilot
    target_mode = args.target_mode
    target_value = args.target_value

    available = TARGET_B_AVAILABLE if target_mode == "B" else TARGET_A_AVAILABLE
    if int(target_value) not in available:
        closest = min(available, key=lambda v: abs(v - target_value))
        print(f"Target {target_mode}={target_value} not trained. Using closest: {closest}.")
        target_value = closest

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        sys.exit(f"Input not found: {input_path}")

    models_path = Path(args.models).resolve() if args.models else find_default_models()
    if not models_path.exists():
        sys.exit(f"Models file not found: {models_path}\n"
                 f"Run tools/build_variants.py to generate the variant pickles, "
                 f"or pass --models with an explicit path.")

    if args.output is None:
        output_path = SCRIPT_DIR / f"{input_path.stem}.pt"
    else:
        output_path = Path(args.output).resolve()

    print(f"Input:   {input_path}")
    print(f"Target:  {target_mode}={target_value} {'dB' if target_mode == 'B' else 'dB tolerance'}")
    print(f"Pilot:   {'on' if pilot else 'off'}")
    print(f"Device:  {args.device}")

    will_train = pilot or not args.dry_run
    if will_train:
        try:
            import model  # noqa: F401
        except ImportError as e:
            sys.exit(f"Cannot import image-gs 'model' module: {e}\n"
                     f"Install image-gs from https://github.com/NYU-ICL/image-gs and make sure it is on PYTHONPATH.")

    print(f"Loading heuristic models from {models_path.name} ...")
    t0 = time.perf_counter()
    try:
        import joblib
        hmodels = joblib.load(models_path)
    except Exception:
        with open(models_path, "rb") as f:
            hmodels = pickle.load(f)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    print("Extracting image features ...")
    metrics, fv = extract_features(input_path)
    print(f"  laplacian_var={metrics['laplacian_var']:.1f}, "
          f"edge_density={metrics['edge_density']:.3f}, "
          f"shannon_entropy={metrics['shannon_entropy']:.2f}")

    pilot_psnr = None
    if pilot:
        print(f"Pilot training: N={PILOT_N}, {PILOT_STEPS} steps ...")
        t0 = time.perf_counter()
        pilot_res, _ = _train(str(input_path), PILOT_N, PILOT_STEPS, args.device)
        pilot_psnr = pilot_res["final_psnr"]
        print(f"  pilot PSNR={pilot_psnr:.2f} dB, {time.perf_counter() - t0:.1f}s")

    n, steps = predict_n_and_steps(hmodels, fv, target_mode, target_value,
                                   pilot=pilot, pilot_psnr=pilot_psnr)
    print(f"Prediction: N={n}, steps={steps}")

    if args.dry_run:
        return

    print(f"Training Image-GS: N={n}, max {steps} steps ...")
    t0 = time.perf_counter()
    result, state = _train(str(input_path), n, steps, args.device)
    print(f"  final PSNR={result['final_psnr']:.2f} dB, "
          f"SSIM={result['final_ssim']:.4f}, "
          f"final step={result['final_step']}, "
          f"size={result['num_bytes']/1024:.1f} KB, "
          f"time={time.perf_counter() - t0:.1f}s")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        **state,
        "input_path": str(input_path),
        "target_mode": target_mode,
        "target_value": target_value,
        "pilot": pilot,
        "pilot_psnr": pilot_psnr,
        "predicted_n": n,
        "predicted_steps": steps,
        "image_metrics": metrics,
    }
    torch.save(bundle, output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
