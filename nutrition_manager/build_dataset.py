"""Build the processed multi-view dataset (overhead + side-angle frames).

Creates ``dataset_processed/`` (never touches the read-only original):

    dataset_processed/
      frames/<dish_id>/cam<C>_s<NN>.jpg   # sampled side-angle frames (480p)
      manifest.csv                        # one row per training/eval image

Design:
  * Overhead RGB rows come straight from data/labels.csv (run build_labels.py
    first) and keep the official train/test split.
  * Side-angle frames are extracted ONLY for train-split dishes and added as
    extra *training* views. The test set stays overhead-only so the metric
    remains directly comparable to the overhead-only v1 model.

Side-angle quirks (observed by DH) are recorded in the ``notes`` column rather
than filtered out:
  * cafe1 videos tend to be mounted UPSIDE-DOWN and pause briefly before the
    camera starts rotating around the plate (45-degree sweep).
  * a few dishes are off-centered or (rarely) shot in near-total darkness.
The whole rgb_train split happens to be cafe1, so every side frame here carries
the cafe1 note.

Run (from nutrition_manager/, snapcare env):
    python build_dataset.py            # extract + build manifest
    python build_dataset.py --manifest-only   # rebuild manifest from existing frames
"""
import argparse
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PROJECT, load_config, resolve  # noqa: E402
from data.build_labels import parse_metadata, read_ids  # noqa: E402

# Sampling / encoding knobs. cafe1 side clips are short (median ~35 frames,
# range ~30-300), so a small step keeps ~4-6 frames across their length.
STEP = 8            # keep every STEP-th decoded frame
CAP = 6             # hard cap on frames kept per camera
SHORT_SIDE = 480    # output height in px (width auto, keeps aspect)
QSCALE = 4          # ffmpeg mjpeg quality (2=best..31=worst)
CAMERAS = ["A", "B", "C", "D"]
WORKERS = 16

FIRST_FIELDS = ["calories", "mass", "fat", "carb", "protein"]
NOTE_CAFE1 = "cafe1:mounted-upside-down;pre-rotation-pause"
NOTE_CAFE2 = "cafe2:short-clip"
FFMPEG = str(Path(sys.executable).parent / "ffmpeg")


def load_id_set(path):
    with open(path) as f:
        return {ln.strip() for ln in f if ln.strip()}


def cafe_of(dish_id, cafe1, cafe2):
    if dish_id in cafe1:
        return "cafe1"
    if dish_id in cafe2:
        return "cafe2"
    return "unknown"


def extract_one(video: Path, out_dir: Path, cam: str):
    """Extract up to CAP evenly-stepped frames from one camera video. Idempotent."""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob(f"cam{cam}_s*.jpg"))
    if existing:
        return existing
    pattern = str(out_dir / f"cam{cam}_s%02d.jpg")
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(video),
        "-vf", f"select='not(mod(n\\,{STEP}))',scale=-2:{SHORT_SIDE}",
        "-vsync", "0", "-frames:v", str(CAP), "-q:v", str(QSCALE), pattern,
    ]
    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob(f"cam{cam}_s*.jpg"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-only", action="store_true",
                    help="skip extraction; rebuild manifest from frames already on disk")
    args = ap.parse_args()

    cfg = load_config()
    dataset_root = cfg["dataset_root"]
    targets = cfg["targets"]

    labels_path = resolve(cfg["labels_csv"])
    if not labels_path.is_file():
        raise SystemExit(f"{labels_path} not found. Run: python data/build_labels.py")

    # Overhead rows keyed by dish_id (carry split + labels + rgb path).
    overhead = {}
    with open(labels_path, newline="") as f:
        for row in csv.DictReader(f):
            overhead[row["dish_id"]] = row

    cafe1 = load_id_set(resolve(dataset_root, "dish_ids/dish_ids_cafe1.txt"))
    cafe2 = load_id_set(resolve(dataset_root, "dish_ids/dish_ids_cafe2.txt"))

    # Dish-level labels for ALL dishes (so side-angle-only dishes — those in the
    # RGB train split with no overhead frame — can still be trained on).
    meta_labels = parse_metadata([resolve(dataset_root, p) for p in cfg["metadata_csvs"]])

    side_src = resolve(dataset_root, "imagery/side_angles")
    frames_root = resolve("dataset_processed", "frames")

    # Side frames are added for every RGB *train* dish that has side video and a
    # label — including side-only dishes v1 could not use. (Test stays overhead.)
    rgb_train = read_ids(resolve(dataset_root, cfg["splits"]["train"]))
    train_ids = [d for d in rgb_train
                 if d in meta_labels and (side_src / d).is_dir()]

    # --- extract side frames for train dishes (parallel ffmpeg) ---
    if not args.manifest_only:
        tasks = []
        for dish_id in train_ids:
            for cam in CAMERAS:
                video = side_src / dish_id / f"camera_{cam}.h264"
                if video.is_file():
                    tasks.append((video, frames_root / dish_id, cam))
        print(f"Extracting side frames: {len(tasks)} camera-videos "
              f"({len(train_ids)} train dishes) with {WORKERS} workers...")
        done = fail = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(extract_one, v, o, c): (v, c) for v, o, c in tasks}
            for fut in as_completed(futs):
                try:
                    fut.result()
                    done += 1
                except Exception as e:  # keep going; record nothing, note in stderr
                    fail += 1
                    print(f"  ffmpeg failed on {futs[fut][0]}: {e}", file=sys.stderr)
                if done % 2000 == 0:
                    print(f"  ...{done}/{len(tasks)} done")
        print(f"Extraction complete: {done} ok, {fail} failed.")

    # --- build manifest: overhead (all splits) + side (train dishes) ---
    header = ["dish_id", "split", "view", "camera", "seq", "cafe",
              *FIRST_FIELDS, "image_path", "notes"]
    rows = []

    for dish_id, r in overhead.items():
        rows.append({
            "dish_id": dish_id, "split": r["split"], "view": "overhead",
            "camera": "", "seq": 0, "cafe": cafe_of(dish_id, cafe1, cafe2),
            **{k: r[k] for k in FIRST_FIELDS},
            "image_path": f"{dataset_root}/{r['rgb_path']}", "notes": "",
        })

    for dish_id in train_ids:
        lab = meta_labels[dish_id]                    # works for side-only dishes too
        cafe = cafe_of(dish_id, cafe1, cafe2)
        note = NOTE_CAFE1 if cafe == "cafe1" else (NOTE_CAFE2 if cafe == "cafe2" else "")
        for frame in sorted((frames_root / dish_id).glob("cam*_s*.jpg")):
            cam = frame.name[3]                       # "camA_s01.jpg" -> "A"
            seq = int(frame.stem.rsplit("_s", 1)[1])
            rows.append({
                "dish_id": dish_id, "split": "train", "view": "side",
                "camera": cam, "seq": seq, "cafe": cafe,
                **{k: lab[k] for k in FIRST_FIELDS},
                "image_path": str(frame.relative_to(PROJECT)), "notes": note,
            })

    out_path = resolve("dataset_processed", "manifest.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

    n_over = sum(1 for r in rows if r["view"] == "overhead")
    n_side = sum(1 for r in rows if r["view"] == "side")
    n_train = sum(1 for r in rows if r["split"] == "train")
    n_test = sum(1 for r in rows if r["split"] == "test")
    print(f"Wrote {len(rows)} rows -> {out_path}")
    print(f"  overhead={n_over}  side={n_side}  |  train={n_train}  test={n_test}")


if __name__ == "__main__":
    main()
