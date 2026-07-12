# Model handoff directory

Place the validated `small_heatmap_v1` checkpoint at:

```text
models/current.pt
```

`models/current.pt` is the only model binary versioned with the application;
training outputs and candidate weights remain excluded from Git. The bundled
default is the confirmed-bone research checkpoint for images without an
artificial joint:

```text
version  20260712-bone-final-v1
scope    confirmed-bone
SHA-256  000a4d09b61f64106a285b4d9fd8211236f0127462d23d6a9376d421002cdbab
epoch    50
input    256 x 320
stride   4
decoder  local_centroid_3x3_residual_v1
schema   1
```

Run `python validate_app_model.py` after replacing it. Compatible state-dict
weights must keep the same architecture, preprocessing contract, and 12-point
schema. Legacy checkpoints are disabled in the release configuration. External
replacements must use schema 1 with the explicit checkpoint, architecture,
preprocessing, and decoder manifest produced by the current training pipeline.
