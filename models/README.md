# Model handoff directory

Place the validated `small_heatmap_v1` checkpoint at:

```text
models/current.pt
```

The binary is intentionally excluded from Git. The current research checkpoint
used during development has:

```text
SHA-256  f4551e22eb9a7eb38e116d229a1fdfbf345633c60346536012ec53f92d70ad8e
epoch    78
input    256 x 320
stride   4
```

Run `python validate_app_model.py` after replacing it. Compatible state-dict
weights must keep the same architecture, preprocessing contract, and 12-point
schema. The bundled legacy checkpoint is allowed only through the reviewed
release configuration. External replacements must use schema 1 with the explicit
checkpoint/architecture/preprocessing manifest produced by the current
`train_keypoint_baseline.py`.
