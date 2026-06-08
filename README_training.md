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

Visualize predictions from a trained checkpoint:

```bash
python visualize_keypoint_predictions.py \
  --manifest outputs/knee_dataset_manifest.csv \
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
