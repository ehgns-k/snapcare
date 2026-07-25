"""Torch Dataset + image transforms for the overhead RGB nutrition model."""
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PROJECT  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size: int, train: bool):
    if train:
        # Vertical flip + wider rotation make the model orientation-robust, which
        # matters because the side-angle training frames come from cafe1 videos
        # that are mounted upside-down and sweep through many azimuths. Nutrition
        # totals are orientation-invariant, so this is safe.
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomApply([transforms.RandomRotation(25)], p=0.5),
                transforms.ColorJitter(0.15, 0.15, 0.15, 0.03),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    resize = int(round(img_size * 1.15))
    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class Nutrition5kDataset(Dataset):
    """Yields (image_tensor, raw_target_vector, dish_id).

    Targets are returned in *raw physical units*; standardization happens in the
    training loop so the exact same checkpoint can de-normalize at eval time.
    """

    def __init__(self, df, targets, transform, path_col: str = "image_path"):
        self.df = df.reset_index(drop=True)
        self.targets = list(targets)
        self.transform = transform
        self.path_col = path_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = PROJECT / row[self.path_col]  # image_path is project-relative
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        target = torch.tensor([float(row[c]) for c in self.targets], dtype=torch.float32)
        return img, target, row["dish_id"]
