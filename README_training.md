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
