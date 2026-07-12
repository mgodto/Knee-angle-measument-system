#!/bin/zsh

set -euo pipefail

cd "$(dirname "$0")"
unset PYTHONPATH PYTHONHOME

if [[ ! -f models/current.pt ]]; then
  echo "Missing models/current.pt" >&2
  echo "Copy the validated production-compatible checkpoint there before building." >&2
  exit 1
fi

python_bin="${PYTHON_BIN:-python3}"
"$python_bin" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13), "Python 3.10-3.12 is required"'
"$python_bin" -c 'import tkinter; print("Tk", tkinter.TkVersion)'
"$python_bin" -m venv --clear .venv-measurement-build
source .venv-measurement-build/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-app.txt
python -m pip check
python -m unittest discover -s tests -v
python validate_app_model.py
python knee_measurement_app.py --smoke-test-image images/annotation_processed_combined/001L_raw.jpg --side L
export PYINSTALLER_CONFIG_DIR="$PWD/.pyinstaller-cache"
export KNEE_TARGET_ARCH="arm64"
pyinstaller --noconfirm --clean knee_measurement_app.spec
dist/KneeXrayMeasurement.app/Contents/MacOS/KneeXrayMeasurement --validate-model
smoke_home="$(mktemp -d)"
trap 'rm -rf "$smoke_home"' EXIT
env -i HOME="$smoke_home" PATH="/usr/bin:/bin" TMPDIR="${TMPDIR:-/tmp}" \
  "$PWD/dist/KneeXrayMeasurement.app/Contents/MacOS/KneeXrayMeasurement" \
  --smoke-test-image "$PWD/images/annotation_processed_combined/001L_raw.jpg" --side L
codesign --verify --deep --strict dist/KneeXrayMeasurement.app
release_zip="dist/KneeXrayMeasurement-macOS-arm64.zip"
release_zip_temp="dist/.KneeXrayMeasurement-macOS-arm64.$$.tmp.zip"
ditto -c -k --sequesterRsrc --keepParent \
  dist/KneeXrayMeasurement.app "$release_zip_temp"
mv -f "$release_zip_temp" "$release_zip"

echo
echo "Build finished: dist/KneeXrayMeasurement.app"
echo "Distribution archive: dist/KneeXrayMeasurement-macOS-arm64.zip"
echo "Run a non-PHI smoke test on this Mac before distribution."
