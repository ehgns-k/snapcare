"""Evaluate a trained checkpoint on the official RGB test split.

Writes predictions.csv and groundtruth_test.csv in the exact format the
upstream scripts/compute_eval_statistics.py expects
(dish_id, calories, mass, fat, carb, protein), prints per-field MAE / MAE%,
and tells you how to run the official scorer for a cross-check.

Usage (from nutrition_manager/):
    python eval.py                          # uses checkpoints/best.pt
    python eval.py --checkpoint path/to.pt
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, resolve  # noqa: E402
from data.dataset import Nutrition5kDataset, build_transforms  # noqa: E402
from models.model import build_model  # noqa: E402


@torch.no_grad()
def predict_split(model, loader, device, mean_t, std_t):
    model.eval()
    ids, preds, gts = [], [], []
    for imgs, y_raw, dish_ids in loader:
        imgs = imgs.to(device, non_blocking=True)
        pred_raw = (model(imgs).float().cpu() * std_t + mean_t).clamp_min(0)
        preds.append(pred_raw.numpy())
        gts.append(y_raw.numpy())
        ids.extend(list(dish_ids))
    return ids, np.concatenate(preds), np.concatenate(gts)


def write_csv(path, ids, values, targets):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for dish_id, row in zip(ids, values):
            w.writerow([dish_id, *[f"{v:.6f}" for v in row]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path(args.checkpoint) if args.checkpoint else resolve(
        cfg["train"]["checkpoint_dir"], "best.pt"
    )
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    targets = ckpt["targets"]
    mean_t = torch.tensor(ckpt["target_mean"], dtype=torch.float32)
    std_t = torch.tensor(ckpt["target_std"], dtype=torch.float32)

    model = build_model(ckpt["backbone"], num_outputs=len(targets), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])

    manifest = resolve(cfg.get("manifest_csv", "dataset_processed/manifest.csv"))
    if manifest.is_file():
        df = pd.read_csv(manifest)
    else:
        df = pd.read_csv(resolve(cfg["labels_csv"]))
        df["view"] = "overhead"
        df["image_path"] = cfg["dataset_root"] + "/" + df["rgb_path"]
    # Evaluate on the overhead test frames only (one per dish) — comparable to v1.
    df_test = df[(df["split"] == "test") & (df["view"] == "overhead")].reset_index(drop=True)
    ds = Nutrition5kDataset(df_test, targets, build_transforms(ckpt["img_size"], False))
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], num_workers=cfg["train"]["num_workers"])

    ids, preds, gts = predict_split(model, loader, device, mean_t, std_t)

    abs_err = np.abs(preds - gts)
    mae = abs_err.mean(axis=0)
    mae_pct = 100.0 * abs_err.sum(axis=0) / np.maximum(gts.sum(axis=0), 1e-6)
    print(f"Test dishes: {len(ids)}\n")
    print(f"{'field':<10}{'MAE':>12}{'MAE %':>10}")
    for t, m, p in zip(targets, mae, mae_pct):
        print(f"{t:<10}{m:>12.2f}{p:>9.1f}%")
    print(f"{'mean':<10}{mae.mean():>12.2f}{mae_pct.mean():>9.1f}%")

    pred_path = resolve("predictions.csv")
    gt_path = resolve("groundtruth_test.csv")
    write_csv(pred_path, ids, preds, targets)
    write_csv(gt_path, ids, gts, targets)
    print(f"\nWrote {pred_path}\nWrote {gt_path}")
    official = resolve(cfg["dataset_root"], "scripts/compute_eval_statistics.py")
    print(
        "\nCross-check with the official scorer:\n"
        f"    python {official} {gt_path} {pred_path} eval_stats.json"
    )


if __name__ == "__main__":
    main()
