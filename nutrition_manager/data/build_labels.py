"""Build a flat labels CSV from the Nutrition5k metadata + overhead imagery.

For each dish id in the official RGB train/test split lists, we keep the dish
only if (a) it has dish-level ground truth in the metadata CSVs and (b) an
overhead ``rgb.png`` actually exists on disk. Many ids in the RGB split are
side-angle-only and have no overhead frame; those are dropped for this
RGB-only v1 model.

Output columns: dish_id, split, calories, mass, fat, carb, protein, rgb_path
(rgb_path is relative to the dataset root).

Run from anywhere:
    python data/build_labels.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_config, resolve  # noqa: E402

FIRST_FIELDS = ["dish_id", "calories", "mass", "fat", "carb", "protein"]


def parse_metadata(csv_paths):
    """dish_id -> {calories, mass, fat, carb, protein} using only the first 6 cols."""
    labels = {}
    for path in csv_paths:
        with open(path, newline="") as f:
            for row in csv.reader(f):
                if len(row) < 6 or not row[0]:
                    continue
                try:
                    vals = [float(x) for x in row[1:6]]
                except ValueError:
                    continue
                labels[row[0]] = dict(zip(FIRST_FIELDS[1:], vals))
    return labels


def read_ids(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def main():
    cfg = load_config()
    dataset_root = cfg["dataset_root"]
    imagery_subdir = cfg["imagery_subdir"]
    rgb_filename = cfg["rgb_filename"]

    meta_paths = [resolve(dataset_root, p) for p in cfg["metadata_csvs"]]
    labels = parse_metadata(meta_paths)
    print(f"Parsed dish-level labels for {len(labels)} dishes.")

    rows = []
    for split, id_file in cfg["splits"].items():
        ids = read_ids(resolve(dataset_root, id_file))
        kept = no_label = no_image = 0
        for dish_id in ids:
            if dish_id not in labels:
                no_label += 1
                continue
            rgb_rel = f"{imagery_subdir}/{dish_id}/{rgb_filename}"
            if not resolve(dataset_root, rgb_rel).is_file():
                no_image += 1
                continue
            lab = labels[dish_id]
            rows.append(
                {
                    "dish_id": dish_id,
                    "split": split,
                    **{k: lab[k] for k in FIRST_FIELDS[1:]},
                    "rgb_path": rgb_rel,
                }
            )
            kept += 1
        print(
            f"[{split}] {len(ids)} ids -> kept {kept} "
            f"(dropped {no_label} missing-label, {no_image} missing-image)"
        )

    out_path = resolve(cfg["labels_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dish_id", "split", *FIRST_FIELDS[1:], "rgb_path"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
