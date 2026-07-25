"""Train the RGB-only Nutrition5k regression model (v1).

Usage (run from nutrition_manager/, with the `snapcare` env active):
    python data/build_labels.py     # once, to create data/labels.csv
    python train.py                 # trains, writes checkpoints/best.pt

Selection metric is the mean per-field MAE% on a held-out validation slice of
the official train split (lower is better).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, resolve  # noqa: E402
from data.dataset import Nutrition5kDataset, build_transforms  # noqa: E402
from models.model import build_model  # noqa: E402


def load_dataframe(cfg):
    """Prefer the multi-view manifest; fall back to the overhead-only labels.csv.

    Both are normalized to have `view` and project-relative `image_path` columns.
    """
    manifest = resolve(cfg.get("manifest_csv", "dataset_processed/manifest.csv"))
    if manifest.is_file():
        print(f"Using manifest: {manifest}")
        return pd.read_csv(manifest)
    labels = resolve(cfg["labels_csv"])
    if not labels.is_file():
        raise SystemExit(
            f"Neither {manifest} nor {labels} found. "
            "Run: python data/build_labels.py  (and optionally python build_dataset.py)"
        )
    print(f"Manifest not found; using overhead-only labels: {labels}")
    df = pd.read_csv(labels)
    df["view"] = "overhead"
    df["image_path"] = cfg["dataset_root"] + "/" + df["rgb_path"]
    return df


def compute_target_stats(df, targets, clip_percentile):
    y = df[targets].to_numpy(dtype=np.float64)
    if clip_percentile:
        caps = np.percentile(y, clip_percentile, axis=0)
        y = np.minimum(y, caps)
    mean = y.mean(axis=0)
    std = y.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


@torch.no_grad()
def evaluate(model, loader, device, mean_t, std_t, targets):
    model.eval()
    abs_err = np.zeros(len(targets))
    gt_sum = np.zeros(len(targets))
    n = 0
    for imgs, y_raw, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        pred_norm = model(imgs).float().cpu()
        pred_raw = pred_norm * std_t + mean_t
        err = (pred_raw - y_raw).abs().sum(dim=0).numpy()
        abs_err += err
        gt_sum += y_raw.sum(dim=0).numpy()
        n += y_raw.shape[0]
    mae = abs_err / max(n, 1)
    mae_pct = 100.0 * abs_err / np.maximum(gt_sum, 1e-6)
    return mae, mae_pct


def main():
    cfg = load_config()
    tcfg = cfg["train"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])
    targets = cfg["targets"]

    df = load_dataframe(cfg)

    df_train_all = df[df["split"] == "train"].reset_index(drop=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)

    # Val split is carved out BY DISH so a dish's side-angle frames can never
    # leak into training while its overhead frame sits in validation. Validation
    # itself is overhead-only, matching the (overhead-only) test protocol.
    rng = np.random.default_rng(tcfg["seed"])
    # Only dishes that have an overhead frame are eligible for validation (val is
    # overhead-only); this keeps side-angle-only dishes entirely in training.
    overhead_dishes = np.sort(
        df_train_all.loc[df_train_all["view"] == "overhead", "dish_id"].unique()
    )
    rng.shuffle(overhead_dishes)
    n_val = int(round(len(overhead_dishes) * tcfg["val_frac"]))
    val_dishes = set(overhead_dishes[:n_val].tolist())

    is_val_dish = df_train_all["dish_id"].isin(val_dishes)
    df_train = df_train_all[~is_val_dish].reset_index(drop=True)
    df_val = df_train_all[is_val_dish & (df_train_all["view"] == "overhead")].reset_index(drop=True)
    n_side = int((df_train["view"] == "side").sum())
    print(f"train={len(df_train)} ({n_side} side + {len(df_train)-n_side} overhead)  "
          f"val={len(df_val)} overhead  test={len(df_test)} overhead")

    mean, std = compute_target_stats(df_train, targets, tcfg["clip_percentile"])
    mean_t = torch.tensor(mean, dtype=torch.float32)
    std_t = torch.tensor(std, dtype=torch.float32)
    print("target mean:", dict(zip(targets, mean.round(2))))
    print("target std :", dict(zip(targets, std.round(2))))

    img_size = cfg["model"]["img_size"]
    train_ds = Nutrition5kDataset(df_train, targets, build_transforms(img_size, True))
    val_ds = Nutrition5kDataset(df_val, targets, build_transforms(img_size, False))

    loader_kwargs = dict(
        batch_size=tcfg["batch_size"],
        num_workers=tcfg["num_workers"],
        pin_memory=(device == "cuda"),
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = build_model(
        cfg["model"]["backbone"], num_outputs=len(targets), dropout=cfg["model"]["dropout"]
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg["epochs"])
    loss_fn = nn.SmoothL1Loss()
    use_amp = bool(tcfg["amp"]) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    mean_dev, std_dev = mean_t.to(device), std_t.to(device)
    ckpt_dir = resolve(tcfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_metric = float("inf")

    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{tcfg['epochs']}")
        for imgs, y_raw, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)
            y_norm = ((y_raw.to(device) - mean_dev) / std_dev)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(imgs)
                loss = loss_fn(pred, y_norm)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        sched.step()

        mae, mae_pct = evaluate(model, val_loader, device, mean_t, std_t, targets)
        mean_pct = float(mae_pct.mean())
        print(f"  val MAE%: " + "  ".join(f"{t}={p:.1f}%" for t, p in zip(targets, mae_pct)))
        print(f"  val mean MAE% = {mean_pct:.2f}  (train loss {running/len(train_loader):.4f})")

        if mean_pct < best_metric:
            best_metric = mean_pct
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "backbone": cfg["model"]["backbone"],
                    "img_size": img_size,
                    "targets": targets,
                    "target_mean": mean.tolist(),
                    "target_std": std.tolist(),
                    "val_mean_mae_pct": mean_pct,
                    "epoch": epoch,
                },
                ckpt_dir / "best.pt",
            )
            print(f"  ** saved best.pt (mean MAE% {mean_pct:.2f})")

    print(f"Done. Best val mean MAE% = {best_metric:.2f}")


if __name__ == "__main__":
    main()
