# Project structure

This repository separates canonical code from launchers, orchestration, local
medical data, and generated artifacts.

## Canonical layout

| Path | Purpose |
| --- | --- |
| `knee_xray/core/` | Shared angle, geometry, and image primitives |
| `knee_xray/inference/` | Checkpoint loading, runtime validation, and model selection |
| `knee_xray/ml/` | Model architectures, preprocessing, and decoding |
| `knee_xray/ui/` | Annotation and measurement GUI implementations and widgets |
| `knee_xray/data/` | Dataset manifests, validation, processing, import, and organization |
| `knee_xray/training/` | Reusable training, evaluation, visualization, CV, and OOF code |
| `knee_xray/release/` | Release fixtures, archive audits, and generated-entrypoint helpers |
| `scripts/training/` | Long-running retraining shell orchestration |
| `scripts/build/` | macOS and Windows build entrypoints |
| `packaging/` | PyInstaller specifications and packaging metadata |
| `config/` | Runtime application configuration |
| `requirements/` | Dependency sets for the app, training, and annotation-tool packaging |
| `tests/` | Automated tests, including the layout contract |
| `models/` | Approved model handoff files and their metadata |
| `docs/` | Operator, training, and architecture documentation |

Generated build and delivery directories are intentionally not part of a clean
checkout. PyInstaller stages applications under `build/pyinstaller-dist/`.
Successful ZIPs are written to one of these ignored delivery directories:

```text
deliverables/
├── annotation/
│   ├── macos/
│   └── windows/
└── measurement/
    ├── macos/
    └── windows/
```

Each product/platform directory keeps only its latest three ZIPs. Build scripts
first validate a candidate under `build/deliverable-stage/`, atomically promote
it to `deliverables/`, and then apply the retention policy through
`python -m knee_xray.release.retain_deliverables`. Dated filenames determine
release order; legacy undated ZIPs fall back to file modification time.

## Root compatibility contract

Two historical commands remain supported:

```bash
python annotate_gui.py
python knee_measurement_app.py
```

Those two files are thin launchers for `knee_xray.ui.annotate_gui` and
`knee_xray.ui.knee_measurement_app`. They must not become independent copies of
the GUI implementations. A Windows build may temporarily generate the ignored
`knee_measurement_app_windows.py`; it is never canonical source or a third
tracked launcher.

No tracked shell script, batch file, PyInstaller spec, or other Python module
belongs at the repository root. Build entrypoints live in `scripts/build/`,
training runners in `scripts/training/`, and specs in `packaging/`.

## Placement and cleanup rules

- Importable behavior belongs in the closest `knee_xray` subpackage. Operator
  commands should call a package `main()` rather than own another implementation.
- Generated files, local medical images, private overrides, reports, and build
  products stay in their existing ignored trees. Production code must never
  import from `outputs/`, `private/`, `build/`, or `deliverables/`.
- Repository-level `dist/` is retired: build scripts must stage under
  `build/pyinstaller-dist/` and place final archives under `deliverables/`.
  An annotation delivery may still contain a legacy `dist/` directory inside
  the ZIP for recipient compatibility.
- An immutable experiment directory may retain a run-specific code snapshot for
  provenance. Reusable fixes must be made in `knee_xray/`; the snapshot is not a
  source module and must not be imported by production code.
- Exact duplicate models or manifests can represent handoff and provenance
  boundaries. Confirm which copy is authoritative before deleting either one.
- Moving a file requires updating imports, tests, documentation, workflows,
  build scripts, specs, and configuration paths together.

Verify a structural change with:

```bash
python -m unittest tests.test_project_layout -v
python -m unittest discover -s tests -v
```

Then run the affected model validation, release archive audit, or retraining
`--preflight-only` command documented for that workflow.
