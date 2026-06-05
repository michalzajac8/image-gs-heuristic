#!/usr/bin/env python3
"""
Build size-reduced variants of heuristic_models.pkl.

Lossless variants only — every kept (target, pilot) entry predicts identically
to the original. Reduction comes from compression and from dropping unused
target configurations.

Usage:
  python tools/build_variants.py [--source heuristic_models.pkl] [--out-dir .]
"""

import argparse
import pickle
import shutil
import time
from pathlib import Path

import joblib

SCRIPT_DIR = Path(__file__).resolve().parent
FINAL_MODEL_DIR = SCRIPT_DIR.parent
DEFAULT_SOURCE = FINAL_MODEL_DIR / "heuristic_models.pkl"

# Subsets of "n_models" keys to keep
SUBSETS = {
    "full":   None,  # keep all
    "common": ["B_30dB_pilot", "B_30dB_no_pilot",
               "B_35dB_pilot", "B_35dB_no_pilot"],
    "sota":   ["B_35dB_pilot"],
}


def subset_models(models_pkl, keep_keys, keep_iter_pilot=True, keep_iter_no_pilot=True):
    out = dict(models_pkl)
    out["n_models"] = {k: v for k, v in models_pkl["n_models"].items() if k in keep_keys}
    if not keep_iter_pilot:
        out.pop("iter_model_pilot", None)
    if not keep_iter_no_pilot:
        out.pop("iter_model_no_pilot", None)
    return out


def fmt_size(nbytes):
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(DEFAULT_SOURCE))
    ap.add_argument("--out-dir", default=str(FINAL_MODEL_DIR))
    ap.add_argument("--compress", type=int, default=3,
                    help="joblib compress level (0-9). Default 3.")
    args = ap.parse_args()

    src = Path(args.source).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source: {src}  ({fmt_size(src.stat().st_size)})")
    print(f"Out:    {out_dir}")
    print(f"Compress level: {args.compress}\n")

    t0 = time.perf_counter()
    with open(src, "rb") as f:
        full = pickle.load(f)
    print(f"Loaded source in {time.perf_counter() - t0:.1f}s")
    print(f"  n_models keys: {len(full['n_models'])}")

    rows = []
    for name, keep in SUBSETS.items():
        # Decide whether to drop iter models (sota only needs pilot iter regressor)
        if name == "sota":
            sub = subset_models(full, set(keep), keep_iter_pilot=True, keep_iter_no_pilot=False)
        elif name == "full":
            sub = full
        else:
            sub = subset_models(full, set(keep), keep_iter_pilot=True, keep_iter_no_pilot=True)

        out_path = out_dir / f"heuristic_{name}.pkl"
        t0 = time.perf_counter()
        joblib.dump(sub, out_path, compress=args.compress)
        dt = time.perf_counter() - t0
        size = out_path.stat().st_size
        n_keys = len(sub["n_models"])
        rows.append((name, n_keys, size, dt))
        print(f"  heuristic_{name}.pkl: {n_keys} N-classifiers, "
              f"{fmt_size(size)}, wrote in {dt:.1f}s")

    print("\nSummary")
    print(f"  {'variant':<10} {'n_models':>10} {'size':>12} {'write':>8}")
    for name, n, size, dt in rows:
        print(f"  {name:<10} {n:>10} {fmt_size(size):>12} {dt:>7.1f}s")


if __name__ == "__main__":
    main()
