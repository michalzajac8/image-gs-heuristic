#!/usr/bin/env python3

import argparse
import importlib.util
import sys
import time
from argparse import Namespace
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent

VARIANT_FILES = ["heuristic_sota.pkl", "heuristic_common.pkl", "heuristic_full.pkl"]

FEATURES = [
    "width", "height", "num_pixels", "file_size_bytes",
    "shannon_entropy", "mean_gradient", "hf_energy_ratio",
    "edge_density", "laplacian_var", "color_entropy",
]

TARGET_B = [25, 28, 30, 32, 35, 38]
TARGET_A = [1, 2, 3, 5]

PILOT_N = 1000
PILOT_STEPS = 400


def find_default_models():
    for name in VARIANT_FILES:
        path = SCRIPT_DIR / name
        if path.exists():
            return path
    return SCRIPT_DIR / VARIANT_FILES[0]


# Image features

def shannon_entropy(gray):
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist[hist > 0]
    p = hist / hist.sum()
    return float(-np.sum(p * np.log2(p)))


def mean_gradient(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.sqrt(gx ** 2 + gy ** 2).mean())


def hf_energy_ratio(gray, cutoff_fraction=0.25):
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray.astype(np.float64)))) ** 2
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt((y - crow) ** 2 + (x - ccol) ** 2)
    cutoff = cutoff_fraction * np.sqrt(crow ** 2 + ccol ** 2)
    total = spectrum.sum()
    if total == 0:
        return 0.0
    return float(spectrum[dist > cutoff].sum() / total)


def edge_density(gray):
    edges = cv2.Canny(gray, 100, 200)
    return float((edges > 0).sum() / edges.size)


def laplacian_var(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def color_entropy(bgr):
    channels = bgr.shape[2] if bgr.ndim == 3 else 1
    entropies = []
    for c in range(channels):
        ch = bgr[:, :, c] if bgr.ndim == 3 else bgr
        hist, _ = np.histogram(ch, bins=256, range=(0, 256))
        hist = hist[hist > 0]
        p = hist / hist.sum()
        entropies.append(-np.sum(p * np.log2(p)))
    return float(np.mean(entropies))


def extract_features(image_path):
    bgr = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")
    if bgr.ndim == 2:
        gray = bgr
        bgr = bgr[:, :, None]
    else:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    metrics = {
        "width": w, "height": h, "num_pixels": h * w,
        "file_size_bytes": image_path.stat().st_size,
        "shannon_entropy": shannon_entropy(gray),
        "mean_gradient": mean_gradient(gray),
        "hf_energy_ratio": hf_energy_ratio(gray),
        "edge_density": edge_density(gray),
        "laplacian_var": laplacian_var(gray),
        "color_entropy": color_entropy(bgr),
    }
    return metrics, np.array([metrics[f] for f in FEATURES])


# Image-GS training

def make_args(image_path, n, max_steps, device):
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


def train(image_path, n, max_steps, device):
    from model import GaussianSplatting2D
    model = GaussianSplatting2D(make_args(image_path, n, max_steps, device))
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


# Prediction

def model_key(target_mode, target_value, pilot):
    suffix = "pilot" if pilot else "no_pilot"
    return f"{target_mode}_{int(target_value)}dB_{suffix}"


def round_steps(value):
    return int(max(500, min(10000, round(value / 100) * 100)))


def predict_n_and_steps(models, features, target_mode, target_value, pilot, pilot_psnr=None):
    key = model_key(target_mode, target_value, pilot)
    if key not in models["n_models"]:
        raise ValueError(f"No N-model '{key}' in this pickle. "
                         f"Available: {sorted(models['n_models'])}.")
    iter_key = "iter_model_pilot" if pilot else "iter_model_no_pilot"
    if iter_key not in models:
        raise ValueError(f"No iteration model '{iter_key}' in this pickle.")

    n_input = np.append(features, pilot_psnr) if pilot else features
    n = int(models["n_models"][key].predict(n_input.reshape(1, -1))[0])

    iter_input = np.append(features, [n, pilot_psnr]) if pilot else np.append(features, n)
    steps = round_steps(models[iter_key].predict(iter_input.reshape(1, -1))[0])
    return n, steps


# Main

def parse_args():
    p = argparse.ArgumentParser(description="Image-GS heuristic: pick N and steps, then train.")
    p.add_argument("input", help="Path to the input image.")
    p.add_argument("--target-mode", choices=["A", "B"], default="B",
                   help="A: relative tolerance vs per-image best PSNR. B: absolute PSNR threshold.")
    p.add_argument("--target-value", type=float, default=35.0,
                   help="B: target PSNR in dB (25, 28, 30, 32, 35, 38). A: tolerance in dB (1, 2, 3, 5).")
    p.add_argument("--no-pilot", action="store_true",
                   help="Disable the pilot training step (on by default).")
    p.add_argument("--output", default=None,
                   help="Output path. Default: <input_stem>.pt next to this script.")
    p.add_argument("--models", default=None,
                   help="Path to a heuristic pickle. Default: smallest present variant.")
    p.add_argument("--device", default="cuda:0", help="Torch device.")
    p.add_argument("--dry-run", action="store_true", help="Predict only, do not train.")
    return p.parse_args()


def main():
    args = parse_args()
    pilot = not args.no_pilot
    target_mode = args.target_mode
    target_value = args.target_value

    available = TARGET_B if target_mode == "B" else TARGET_A
    if int(target_value) not in available:
        target_value = min(available, key=lambda v: abs(v - target_value))
        print(f"Target {target_mode}={args.target_value} not trained. Using closest: {target_value}.")

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        sys.exit(f"Input not found: {input_path}")

    models_path = Path(args.models).resolve() if args.models else find_default_models()
    if not models_path.exists():
        sys.exit(f"Models file not found: {models_path}\n"
                 f"Download a heuristic pickle from the models-v1 release and place it next to "
                 f"predict.py, or pass --models with an explicit path.")

    output_path = Path(args.output).resolve() if args.output else SCRIPT_DIR / f"{input_path.stem}.pt"

    print(f"Input:  {input_path}")
    print(f"Target: {target_mode}={target_value} dB")
    print(f"Pilot:  {'on' if pilot else 'off'}")
    print(f"Device: {args.device}")

    if (pilot or not args.dry_run) and importlib.util.find_spec("model") is None:
        sys.exit("image-gs 'model' module not found.\n"
                 "Install image-gs from https://github.com/NYU-ICL/image-gs and put it on PYTHONPATH.")

    print(f"Loading {models_path.name} ...")
    models = joblib.load(models_path)

    metrics, features = extract_features(input_path)

    pilot_psnr = None
    if pilot:
        print(f"Pilot training: N={PILOT_N}, {PILOT_STEPS} steps ...")
        pilot_result, _ = train(input_path, PILOT_N, PILOT_STEPS, args.device)
        pilot_psnr = pilot_result["final_psnr"]
        print(f"  pilot PSNR={pilot_psnr:.2f} dB")

    n, steps = predict_n_and_steps(models, features, target_mode, target_value, pilot, pilot_psnr)
    print(f"Prediction: N={n}, steps={steps}")

    if args.dry_run:
        return

    print(f"Training Image-GS: N={n}, max {steps} steps ...")
    t0 = time.perf_counter()
    result, state = train(input_path, n, steps, args.device)
    print(f"  PSNR={result['final_psnr']:.2f} dB, SSIM={result['final_ssim']:.4f}, "
          f"step={result['final_step']}, size={result['num_bytes'] / 1024:.1f} KB, "
          f"time={time.perf_counter() - t0:.1f}s")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        **state,
        "input_path": str(input_path),
        "target_mode": target_mode,
        "target_value": target_value,
        "pilot": pilot,
        "pilot_psnr": pilot_psnr,
        "predicted_n": n,
        "predicted_steps": steps,
        "image_metrics": metrics,
    }, output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
