# Model handoff directory

The v0.5.1 measurement application versions exactly three reviewed
`small_heatmap_v1` checkpoints:

| Mode | File | Checkpoint version | SHA-256 |
|---|---|---|---|
| Bone | `models/bone.pt` | `20260720-bone-final-v1` | `24481410c3fd2ce2222eed422f4d519f72da95ba232570ee31a4827d45201cfd` |
| TKA | `models/tka.pt` | `20260720-tka-final-v1` | `87027887ec091068a9b91b01a881092400fed58eb8d3eeaaeddb10e8be398e5f` |
| Mixed | `models/mixed.pt` | `20260720-bone-tka-mixed-final-v1` | `f0cfa67f34691f3d81da0e10f0d6ff753dcf71f5aafd278ddb6bf146efc6ba45` |

All three use checkpoint schema 1, 256 x 320 input, stride 4,
`grayscale_resize_percentile_1_99_v1` preprocessing, and the
`local_centroid_3x3_residual_v1` decoder. Legacy checkpoints are disabled in
the release configuration.

Run the complete configured-model validation after replacing any weight:

```bash
python knee_measurement_app.py --validate-models
```

The release audit requires these exact filenames and approved hashes. Candidate
training outputs remain excluded from Git and must never be copied wholesale
into a doctor-facing package.
