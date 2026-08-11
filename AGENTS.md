# Repository organization

Keep the repository layout predictable and preserve the existing command-line
contracts while making changes.

- Put canonical production Python code under `knee_xray/`. Use the `core`,
  `inference`, `ml`, `ui`, `data`, `training`, and `release` subpackages.
- The only tracked Python files allowed at the repository root are
  `annotate_gui.py` and `knee_measurement_app.py`. They are thin compatibility
  launchers that delegate to `knee_xray.ui`; do not add application logic there.
- A generated, git-ignored `knee_measurement_app_windows.py` may exist during a
  Windows build. It is an artifact, not a third canonical launcher.
- Put training shell orchestration in `scripts/training/` and platform build
  shell or batch files in `scripts/build/`. Keep PyInstaller specs in
  `packaging/`, runtime JSON configuration in `config/`, and dependency sets in
  `requirements/`.
- Keep operator and architecture documentation in `docs/`. Update
  `docs/PROJECT_STRUCTURE.md` whenever an intentional layout exception is added.
- Treat `images/`, `outputs/`, `private/`, `build/`, `deliverables/`, `releases/`,
  `projects/`, and `automation_monthly_reports/` as data, private state, generated
  artifacts, or immutable run snapshots. Canonical code must not depend on code
  that exists only in one of these directories.
- Use `build/pyinstaller-dist/` only for PyInstaller staging. Final ZIPs belong in
  `deliverables/{measurement|annotation}/{macos|windows}/`; after a successful
  archive validation, atomically promote the candidate from
  `build/deliverable-stage/`, then run `knee_xray.release.retain_deliverables`
  so each product/platform directory keeps only its latest three ZIPs.
- The repository-level `dist/` directory is retired and must not be used for
  staging or final output. A delivery archive may retain an internal `dist/`
  layout when required for compatibility; that does not make repository `dist/`
  a valid build path.
- Do not delete medical data, private reconciliation metadata, model weights,
  manifests, or experiment snapshots merely because files look duplicated or
  are not imported by the application. Preserve provenance unless deletion is
  explicitly approved.
- When moving code, update imports, tests, README commands, workflows, build
  scripts, specs, and configuration paths in the same change. Avoid keeping a
  second implementation at the old location.
- Run `python -m unittest discover -s tests -v` after structural changes, plus
  the relevant model validation, release audit, or retraining preflight.
- Keep `tests/test_project_layout.py` aligned with this file. Do not weaken it to
  accommodate a temporary misplaced module.
