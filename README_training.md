# Knee X-ray Keypoint Baseline

This baseline trains a small heatmap-regression model from the annotation JSON files.

## Data Flow

1. Build a manifest that maps each annotation JSON to the local raw X-ray image.
2. Validate that all points, lines, raw images, and angle calculations are usable.
3. Train a small keypoint model that predicts 12 heatmaps:
   - 8 anatomical points
   - 4 manual joint-line endpoints

Do not train on `*_point.jpg`, `*_line.jpg`, or `*_combined.jpg`; those files contain labels drawn on the image.

## Commands

```bash
python build_dataset_manifest.py --output outputs/knee_dataset_manifest.csv
python validate_knee_dataset.py --manifest outputs/knee_dataset_manifest.csv
python train_keypoint_baseline.py --manifest outputs/knee_dataset_manifest.csv
```

Build the integrated legacy + 2026-07-06 dataset. This skips annotations whose
original raw X-ray image is not present locally and writes a report for them:

```bash
python build_dataset_manifest.py \
  --annotation-dir images/Knee_Xray_annotations images/20260706new \
  --raw-root images/line_point images/raw_line_point images/point_only images/20260706new "images/20260706new/001-027 raw" \
  --output outputs/knee_dataset_manifest_combined.csv \
  --inventory-output outputs/knee_dataset_inventory_combined.csv \
  --missing-output outputs/knee_dataset_missing_raw_combined.csv \
  --skip-missing-raw \
  --skip-invalid

python process_annotation_dataset.py \
  --manifest outputs/knee_dataset_manifest_combined.csv \
  --output-dir images/annotation_processed_combined \
  --no-overlays

python organize_implant_dataset.py \
  --manifest images/annotation_processed_combined/processed_manifest.csv \
  --output-root images/annotation_dataset_by_implant

python validate_knee_dataset.py \
  --manifest images/annotation_processed_combined/processed_manifest.csv

python validate_knee_dataset.py \
  --dataset-dir "images/annotation_dataset_by_implant/未加入人工關節"

python validate_knee_dataset.py \
  --dataset-dir "images/annotation_dataset_by_implant/加入人工關節"
```

Add `--clean` to `organize_implant_dataset.py` when rebuilding the organized
folders from scratch.

The organized review/training folders are:

- `images/annotation_dataset_by_implant/未加入人工關節`
- `images/annotation_dataset_by_implant/加入人工關節`

Each sample is stored under `samples/<sample_id>/` with:

- `raw.jpg`
- `annotation.json`
- `point.jpg`
- `line.jpg`
- `combined.jpg`

The processing step also writes focused manifests for model experiments:

- `images/annotation_processed_combined/processed_manifest_bone.csv`
- `images/annotation_processed_combined/processed_manifest_tka.csv`
- `images/annotation_processed_combined/processed_manifest_unknown.csv`
- `images/annotation_processed_combined/processed_manifest_bone_or_legacy_unknown.csv`

## Single-leg v2 preprocessing contract

All training and validation inputs must contain only the annotated target leg.
Do not train directly from a bilateral raster, even when all labels happen to be
on one side. The current reviewed rebuild is produced with:

```bash
python process_annotation_dataset.py \
  --manifest outputs/retraining_20260803/manifests/all_clean.csv \
  --output-dir outputs/retraining_single_leg_v2_20260803/processed \
  --no-overlays

python organize_implant_dataset.py \
  --manifest outputs/retraining_single_leg_v2_20260803/processed/processed_manifest.csv \
  --output-root images/annotation_dataset_single_leg_v2 \
  --clean
```

The crop selection rules are fail-closed and ordered:

1. Use `analysis.inference_roi` from a Measurement App export only when it is
   explicitly confirmed, uses `source_image_pixels`, is in bounds, covers all
   8 landmarks and 4 joint-line endpoints, and—for a declared bilateral
   source—is narrower than 80% of the source image.
2. Preserve a previous substantive target-leg crop rather than cropping it a
   second time only when its provenance uses exact integer coordinates, matches
   the current crop dimensions, and is genuinely non-identity. A previous
   full-image identity decision is re-evaluated when the image is broad;
   fractional or dimension-inconsistent provenance is rejected. None of these
   records can silently mark a broad bilateral image as already split.
3. When left/right annotations share one source raster, use their paired
   positions to split the image while retaining padding toward the divider.
4. For any other declared bilateral, `RL`, or broad image (`width / height >=
   0.60`), make a wide annotation-safe target-leg crop. Reject annotations whose
   horizontal span is too wide or whose points cannot be retained safely.
5. Keep a genuinely narrow single-leg image unchanged.

Coordinates are shifted into the processed crop, while the JSON and manifests
retain the ultimate source paths and SHA-256 values, the absolute source-image
crop box, crop method, confirmation state, and composed `processing_steps`.
Review `processed/crop_summary.csv`, then validate both organized cohorts:

```bash
python validate_knee_dataset.py \
  --dataset-dir "images/annotation_dataset_single_leg_v2/未加入人工關節"

python validate_knee_dataset.py \
  --dataset-dir "images/annotation_dataset_single_leg_v2/加入人工關節"
```

For paired comparisons with the 2026-08-03 baseline, the v2 manifests preserve
the same `sample_id`, anatomical side, `case_id`/`source_case_id`, patient groups,
and 5-fold assignment. Train Bone, TKA, and Mixed from:

- `outputs/retraining_single_leg_v2_20260803/manifests/bone_confirmed.csv`
- `outputs/retraining_single_leg_v2_20260803/manifests/tka.csv`
- `outputs/retraining_single_leg_v2_20260803/manifests/bone_tka.csv`

The resumable fixed-recipe run is:

```bash
bash run_retraining_single_leg_v2_20260803.sh
```

It performs patient-group 5-fold OOF training/evaluation before training each
delivery candidate on all clean rows. Do not replace bundled release weights
until the new OOF summaries and extreme-error cases have been reviewed.

For the integrated non-TKA knee model, use the `未加入人工關節` folder.
It contains confirmed new `bone` annotations plus the legacy dataset:

```bash
python train_keypoint_baseline.py \
  --dataset-dir "images/annotation_dataset_by_implant/未加入人工關節" \
  --output-dir outputs/knee_keypoint_bone_baseline
```

For the TKA model, use:

```bash
python train_keypoint_baseline.py \
  --dataset-dir "images/annotation_dataset_by_implant/加入人工關節" \
  --output-dir outputs/knee_keypoint_tka_baseline
```

If you need a stricter experiment with only confirmed new `bone` annotations, use:

```bash
python train_keypoint_baseline.py \
  --manifest images/annotation_processed_combined/processed_manifest_bone.csv \
  --output-dir outputs/knee_keypoint_bone_only_baseline
```

Visualize predictions from a trained checkpoint:

```bash
python visualize_keypoint_predictions.py \
  --dataset-dir "images/annotation_dataset_by_implant/未加入人工關節" \
  --checkpoint outputs/knee_keypoint_baseline/best.pt \
  --output-dir outputs/knee_keypoint_visualizations \
  --split val \
  --max-images 12
```

The visualization overlays ground truth and predicted keypoints on the raw X-ray:

- `G*` / yellow: ground truth
- `P*` / red: prediction

It also writes `prediction_metrics.csv` with point error and angle error.

For a quick smoke test:

```bash
python train_keypoint_baseline.py \
  --manifest outputs/knee_dataset_manifest.csv \
  --output-dir outputs/knee_keypoint_baseline_smoke \
  --epochs 1 \
  --image-width 128 \
  --image-height 160 \
  --batch-size 2 \
  --max-samples 10 \
  --device cpu
```

The checkpoint and training log are written under `outputs/`, which is ignored by git.

## Clean 5-fold retraining workflow

For model selection or release candidates, first build duplicate-audited manifests
from the reviewed implant folders. This deliberately keeps confirmed bone, TKA
cohort, and legacy `unknown` data separate:

```bash
python prepare_retraining_manifests.py \
  --output-dir outputs/retraining/manifests \
  --num-folds 5 \
  --seed 42
```

Train one case-level validation fold with the selected weighted-heatmap recipe:

```bash
python train_keypoint_baseline.py \
  --manifest outputs/retraining/manifests/bone_confirmed.csv \
  --output-dir outputs/retraining/bone_fold0 \
  --fold 0 \
  --num-folds 5 \
  --seed 42 \
  --epochs 80 \
  --batch-size 4 \
  --heatmap-peak-weight 20 \
  --decoder-id local_centroid_3x3_residual_v1 \
  --cache-dataset

python evaluate_keypoint_checkpoint.py \
  --checkpoint outputs/retraining/bone_fold0/best.pt \
  --manifest outputs/retraining/manifests/bone_confirmed.csv \
  --output-dir outputs/retraining/evaluations/bone_fold0 \
  --split val \
  --fold 0 \
  --num-folds 5 \
  --seed 42
```

Repeat training and evaluation for folds 1 through 4, then combine the five
evaluation directories:

```bash
python summarize_keypoint_cv.py \
  outputs/retraining/evaluations/bone_fold0 \
  outputs/retraining/evaluations/bone_fold1 \
  outputs/retraining/evaluations/bone_fold2 \
  outputs/retraining/evaluations/bone_fold3 \
  outputs/retraining/evaluations/bone_fold4 \
  --output-dir outputs/retraining/evaluations/bone_cv
```

Only after the five-fold summary is complete, train the delivery checkpoint on
all clean samples. Its `val_metrics` are copied from the matching OOF summary,
not measured on its own training rows:

```bash
python train_keypoint_baseline.py \
  --manifest outputs/retraining/manifests/bone_confirmed.csv \
  --output-dir outputs/retraining/final_bone \
  --train-all \
  --cv-summary outputs/retraining/evaluations/bone_cv/cv_summary.json \
  --epochs 50 \
  --batch-size 4 \
  --heatmap-peak-weight 20 \
  --decoder-id local_centroid_3x3_residual_v1 \
  --model-scope confirmed-bone \
  --cache-dataset
```

Use `tka.csv` in the same workflow for the separate TKA-cohort model. Do not
merge `legacy_unknown.csv` into a release candidate until its cohort labels are
reviewed.

Use `bone_tka.csv` for a single mixed Bone+TKA model. This manifest excludes
legacy `unknown` rows and recomputes the case-level folds across both cohorts,
so preoperative Bone and postoperative TKA images from the same case remain in
the same fold.

## Safe incremental physician-annotation import

Use the incremental importer for a newly reviewed annotation folder. It maps
`TKA` to `加入人工關節` and `bone` to `未加入人工關節`, stages every sample,
deduplicates it against the existing dataset, and validates the standard
five-file sample layout before changing the destination.

Override maps can contain source-image identifiers and study dates. Keep them
under the git-ignored `private/` directory and never commit them to the public
repository. The commands below assume that local-only location.

Run a dry run first (the default):

```bash
python import_annotation_batch.py \
  --batch-dir images/20260720 \
  --dataset-root images/annotation_dataset_by_implant \
  --report-dir outputs/annotation_import_20260720 \
  --overrides-json private/annotation_import_overrides_20260720.json
```

Review `decisions.csv` and `summary.json` in the report directory, then apply
the same reviewed batch:

```bash
python import_annotation_batch.py \
  --batch-dir images/20260720 \
  --dataset-root images/annotation_dataset_by_implant \
  --report-dir outputs/annotation_import_20260720 \
  --overrides-json private/annotation_import_overrides_20260720.json \
  --apply
```

The importer never silently overwrites an existing sample. Reviewed corrections
belong in the overrides JSON; replaced samples are moved into the report
directory so they remain recoverable.

Measurement App `*_measurement.json` exports are supported directly. During
import, the declared source filename, dimensions, side, and SHA-256 are checked
against the source raster before a sample can be staged. A valid confirmed
`analysis.inference_roi` is carried into preprocessing so a bilateral source is
cropped to the same doctor-approved target leg. Unconfirmed, out-of-bounds, or
label-excluding ROIs are never trusted; preprocessing must produce a safe
fallback crop or reject the sample. For reviewed corrections, an override may
name both `replace_existing_sample_id` and `replace_existing_sample_dir`;
raw-image replacement additionally requires the explicit
`allow_raw_replacement` flag.

The GUI names bilateral exports `<raw-stem>_L_measurement.*` and
`<raw-stem>_R_measurement.*`, so both legs from one raster can coexist. For an
ordinary new import, two annotations sharing one raw SHA-256 are accepted only
as an exact verified pair: exactly one L and one R, the same case number,
`case_id`, and study phase, valid confirmed ROIs, and opposite screen-side
crops. Any missing, duplicate, same-side, same-screen-side, unconfirmed, or
mismatched pair is quarantined rather than guessed. The only exception is an
explicitly reviewed replacement transaction.

When filenames contain a stable patient token (for example the text before
`_CR_`), build retraining manifests with patient grouping enabled. This joins
different case numbers and Bone/TKA timepoints from the same patient before
fold assignment, preventing patient leakage across cross-validation folds:

```bash
python prepare_retraining_manifests.py \
  --output-dir outputs/retraining/manifests \
  --num-folds 5 \
  --seed 42 \
  --group-by-patient
```
