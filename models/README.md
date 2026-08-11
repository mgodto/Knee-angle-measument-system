# Model handoff directory

The v0.6.0 Internal Research Candidate contains exactly three tail-QA-curated v4
`small_heatmap_v1` checkpoints. They have not been formally promoted and are
not for clinical use:

| Mode | File | Checkpoint version | SHA-256 |
|---|---|---|---|
| Bone | `models/bone.pt` | `20260811-single-leg-v4-curated-tailqa-9e9a1de4fe98-bone-final-v1` | `36e8fee67c7c6bad8071a5a7ff8dbc713d76e28482c2a26798abd15ba3334862` |
| TKA | `models/tka.pt` | `20260811-single-leg-v4-curated-tailqa-9e9a1de4fe98-tka-final-v1` | `23a416f8c376156b3fa323298d3e45ae00060d619e0215754a46f2a2254a1669` |
| Mixed | `models/mixed.pt` | `20260811-single-leg-v4-curated-tailqa-9e9a1de4fe98-bone-tka-mixed-final-v1` | `5e2f5087a433aa6bc9c58d792e132d88f1098952df446512375818a779d1254e` |

All three use checkpoint schema 1, 256 x 320 input, stride 4,
`grayscale_resize_percentile_1_99_v1` preprocessing, and the
`local_centroid_3x3_residual_v1` decoder. Legacy checkpoints are disabled in
the release configuration.

Run the complete configured-model validation after replacing any weight:

```bash
python knee_measurement_app.py --validate-models \
  --expected-model-version bone=20260811-single-leg-v4-curated-tailqa-9e9a1de4fe98-bone-final-v1 \
  --expected-model-version tka=20260811-single-leg-v4-curated-tailqa-9e9a1de4fe98-tka-final-v1 \
  --expected-model-version mixed=20260811-single-leg-v4-curated-tailqa-9e9a1de4fe98-bone-tka-mixed-final-v1
```

The candidate archive audit requires these exact filenames, versions, and
hashes. Training outputs remain excluded from Git; only final checkpoints that
pass manifest/CV provenance, CPU runtime, OOF integrity, and tail-review gates
may be copied into this isolated candidate bundle.
