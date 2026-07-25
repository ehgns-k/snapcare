# SnapCare — nutrition estimation model

RGB-only nutrition regression on the [Nutrition5k](https://github.com/google-research-datasets/Nutrition5k)
dataset. Given an overhead photo of a plated dish, the model predicts five
dish-level totals: **calories, mass, fat, carbohydrate, protein** — the exact
targets scored by the official `compute_eval_statistics.py`.

This is **v1**: a single ImageNet-pretrained CNN backbone (`efficientnet_b0` by
default) with a 5-way regression head, trained on the overhead RealSense RGB
frames. Depth (RGB-D) and side-angle frames are planned follow-ups.

## Layout

```
nutrition_manager/
  nutrition5k_dataset/     # third-party dataset (read-only, gitignored)
  configs/config.yaml      # all paths + hyperparameters
  common.py                # config loading + project-relative paths
  data/build_labels.py     # metadata + overhead imagery -> data/labels.csv
  build_dataset.py         # + side-angle frames -> dataset_processed/manifest.csv
  data/dataset.py          # torch Dataset + transforms
  models/model.py          # backbone + regression head (timm)
  train.py                 # training loop -> checkpoints/best.pt
  eval.py                  # test-split scoring + predictions.csv
  predict.py               # single-image inference (app entry point)
  environment.yml          # conda env `snapcare`
```

## Processed multi-view dataset (`dataset_processed/`)

`build_dataset.py` creates a **separate** processed dataset — the original
`nutrition5k_dataset/` is never modified:

```
dataset_processed/
  frames/<dish_id>/cam<A-D>_s<NN>.jpg   # sampled side-angle frames (480p JPEG)
  manifest.csv                          # one row per image (train/eval unit)
```

`manifest.csv` columns: `dish_id, split, view, camera, seq, cafe,
calories, mass, fat, carb, protein, image_path, notes` — `image_path` is
project-relative and may point either into the read-only original (overhead
frames) or into `dataset_processed/` (side frames).

**How side frames are used.** The overhead `realsense_overhead/rgb.png` frames
keep the official split. Side-angle frames (four cameras sweeping the plate at
~45°) are sampled (~4/camera) and added **only to the train split**; the test
set stays overhead-only so metrics remain comparable to the overhead-only
baseline. The validation slice is held out **by dish**, so a dish's side frames
never leak into training against its own overhead val frame.

**Side-angle quirks (recorded in `notes`, not filtered).** Per DH's inspection:

- **cafe1** videos tend to be **mounted upside-down** and **pause briefly**
  before the camera starts rotating. The entire `rgb_train` split is cafe1, so
  every side training frame carries this note. We do not de-rotate them — instead
  training augmentation adds vertical flips + wider rotation so the model is
  orientation-invariant (nutrition totals don't depend on orientation).
- A few dishes are **off-centered**, and (rarely) one is shot in **near-total
  darkness**. These are kept as-is; they are realistic noise.

Note the dataset's ground-truth id lists only distinguish **cafe1 (4768)** and
**cafe2 (238)** — there is no cafe3 label to key on.

## Setup

```bash
conda env create -f environment.yml
conda activate snapcare
```

## Usage

Run everything **from this `nutrition_manager/` directory**:

```bash
python data/build_labels.py   # build data/labels.csv (splits intersected with available RGB)
python build_dataset.py       # extract side frames -> dataset_processed/manifest.csv (needs ffmpeg)
python train.py               # train; saves checkpoints/best.pt
python eval.py                # score the test split; writes predictions.csv
python predict.py meal.jpg    # single-image inference
```

`build_dataset.py` needs `ffmpeg` (installed into the `snapcare` env:
`conda install -n snapcare -c conda-forge ffmpeg`). If you skip it, `train.py`
falls back to the overhead-only `labels.csv`.

## Notes / design decisions

- **Split fidelity.** Uses the official `rgb_train_ids` / `rgb_test_ids` lists.
  Roughly a third of those ids are side-angle-only (no overhead frame) and are
  dropped by `build_labels.py`, leaving ~2.6k train / ~450 test overhead images.
- **Targets** are standardized (per-field mean/std from the train slice) for a
  balanced loss; the checkpoint stores the stats so eval/inference de-normalize
  identically. Optional outlier clipping via `train.clip_percentile`.
- **Metric** mirrors the upstream scorer (per-field MAE and MAE%). `eval.py`
  also emits `predictions.csv` + `groundtruth_test.csv` so you can run
  `nutrition5k_dataset/scripts/compute_eval_statistics.py` for a cross-check.
- The dataset directory is treated as **read-only**.

## Roadmap

- v2: add the `depth_raw` channel (RGB-D 4-channel input).
- Side-angle frames via `nutrition5k_dataset/scripts/extract_frames_sampled.sh`.
- Optional per-ingredient auxiliary head (the metadata carries ingredient breakdowns).
```
