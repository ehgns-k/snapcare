"""Single-image nutrition inference — the entry point for the SnapCare app.

Usage (from nutrition_manager/):
    python predict.py path/to/meal.jpg
    python predict.py path/to/meal.jpg --checkpoint checkpoints/best.pt

Note: v1 is trained on overhead RealSense frames, so it works best on a
roughly top-down photo of a single plated dish.
"""
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import resolve  # noqa: E402
from data.dataset import build_transforms  # noqa: E402
from models.model import build_model  # noqa: E402

UNITS = {"calories": "kcal", "mass": "g", "fat": "g", "carb": "g", "protein": "g"}


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(ckpt["backbone"], num_outputs=len(ckpt["targets"]), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--checkpoint", default=str(resolve("checkpoints", "best.pt")))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not Path(args.checkpoint).is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    model, ckpt = load_model(args.checkpoint, device)

    mean = torch.tensor(ckpt["target_mean"])
    std = torch.tensor(ckpt["target_std"])
    tf = build_transforms(ckpt["img_size"], train=False)

    img = Image.open(args.image).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = (model(x).float().cpu().squeeze(0) * std + mean).clamp_min(0)

    print(f"Prediction for {args.image}:")
    for name, value in zip(ckpt["targets"], pred.tolist()):
        print(f"  {name:<9}{value:8.1f} {UNITS.get(name, '')}")


if __name__ == "__main__":
    main()
