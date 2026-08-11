#!/bin/zsh

set -euo pipefail

cd "$(dirname "$0")"
unset PYTHONPATH PYTHONHOME

app_version="0.6.0"
release_date="20260811"
expected_bone_version="20260811-single-leg-v4-curated-tailqa-9e9a1de4fe98-bone-final-v1"
expected_tka_version="20260811-single-leg-v4-curated-tailqa-9e9a1de4fe98-tka-final-v1"
expected_mixed_version="20260811-single-leg-v4-curated-tailqa-9e9a1de4fe98-bone-tka-mixed-final-v1"
expected_bone_sha="36e8fee67c7c6bad8071a5a7ff8dbc713d76e28482c2a26798abd15ba3334862"
expected_tka_sha="23a416f8c376156b3fa323298d3e45ae00060d619e0215754a46f2a2254a1669"
expected_mixed_sha="5e2f5087a433aa6bc9c58d792e132d88f1098952df446512375818a779d1254e"
release_root="KneeXrayMeasurement-ResearchCandidate-macOS-arm64-v${app_version}-${release_date}"
release_zip="$PWD/dist/${release_root}.zip"

for model_path in models/bone.pt models/tka.pt models/mixed.pt; do
  if [[ ! -f "$model_path" ]]; then
    echo "Missing $model_path" >&2
    exit 1
  fi
done

python_bin="${PYTHON_BIN:-python3}"
"$python_bin" -c 'import platform,struct,sys; assert (3, 10) <= sys.version_info[:2] < (3, 13); assert struct.calcsize("P") == 8 and platform.machine() == "arm64", "Apple Silicon Python is required"'
"$python_bin" -c 'import tkinter; print("Tk", tkinter.TkVersion)'
"$python_bin" -m venv --clear .venv-measurement-build
source .venv-measurement-build/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-app.txt
python -m pip check
python -m unittest discover -s tests -v
python -c 'from knee_measurement_app import APP_RELEASE_CHANNEL, APP_VERSION; assert APP_VERSION == "0.6.0", APP_VERSION; assert APP_RELEASE_CHANNEL == "INTERNAL RESEARCH CANDIDATE - NOT FOR CLINICAL USE", APP_RELEASE_CHANNEL'
python knee_measurement_app.py --validate-models \
  --expected-model-version "bone=$expected_bone_version" \
  --expected-model-version "tka=$expected_tka_version" \
  --expected-model-version "mixed=$expected_mixed_version"

smoke_base="$(mktemp "${TMPDIR:-/tmp}/knee-xray-smoke.XXXXXX")"
smoke_image="${smoke_base}-unknown.png"
smoke_bone_image="${smoke_base}-bone.png"
smoke_tka_image="${smoke_base}-TKA.png"
rm -f "$smoke_base"
smoke_home="$(mktemp -d "${TMPDIR:-/tmp}/knee-xray-home.XXXXXX")"
package_stage="$(mktemp -d "${TMPDIR:-/tmp}/knee-xray-package.XXXXXX")"
extracted_stage="$(mktemp -d "${TMPDIR:-/tmp}/knee-xray-extracted.XXXXXX")"
trap 'rm -f "$smoke_base" "$smoke_image" "$smoke_bone_image" "$smoke_tka_image"; rm -rf "$smoke_home" "$package_stage" "$extracted_stage"' EXIT

python create_release_smoke_fixture.py "$smoke_image"
cp "$smoke_image" "$smoke_bone_image"
cp "$smoke_image" "$smoke_tka_image"
python knee_measurement_app.py --smoke-test-image "$smoke_image" --side R --model-mode mixed

export PYINSTALLER_CONFIG_DIR="$PWD/.pyinstaller-cache"
export KNEE_TARGET_ARCH="arm64"
rm -rf "$PWD/dist/KneeXrayMeasurement" "$PWD/dist/KneeXrayMeasurement.app" "$PWD/build/knee_measurement_app"
pyinstaller --noconfirm --clean knee_measurement_app.spec

mkdir -p "$package_stage/$release_root"
rsync -a --exclude '.DS_Store' --exclude '._*' --exclude '__MACOSX' \
  "$PWD/dist/KneeXrayMeasurement.app" "$package_stage/$release_root/"
cp README_DOCTOR_JA.txt "$package_stage/$release_root/README_DOCTOR_JA.txt"

rm -f "$release_zip"
(
  cd "$package_stage"
  COPYFILE_DISABLE=1 zip -qry -y -X "$release_zip" "$release_root"
)
python audit_measurement_archive.py "$release_zip" \
  --platform macos \
  --expected-root "$release_root" \
  --expected-model "bone.pt=$expected_bone_sha" \
  --expected-model "tka.pt=$expected_tka_sha" \
  --expected-model "mixed.pt=$expected_mixed_sha" \
  --expected-app-version "$app_version"

unzip -q "$release_zip" -d "$extracted_stage"
extracted_app="$extracted_stage/$release_root/KneeXrayMeasurement.app"
codesign --verify --deep --strict "$extracted_app"
if find -L "$extracted_app" -type l -print -quit | grep -q .; then
  echo "Extracted app contains a broken symlink." >&2
  exit 1
fi
env -i HOME="$smoke_home" PATH="/usr/bin:/bin" TMPDIR="${TMPDIR:-/tmp}" \
  "$extracted_app/Contents/MacOS/KneeXrayMeasurement" --validate-models \
  --expected-model-version "bone=$expected_bone_version" \
  --expected-model-version "tka=$expected_tka_version" \
  --expected-model-version "mixed=$expected_mixed_version"
env -i HOME="$smoke_home" PATH="/usr/bin:/bin" TMPDIR="${TMPDIR:-/tmp}" \
  "$extracted_app/Contents/MacOS/KneeXrayMeasurement" \
  --smoke-test-image "$smoke_image" --side R --model-mode bone
env -i HOME="$smoke_home" PATH="/usr/bin:/bin" TMPDIR="${TMPDIR:-/tmp}" \
  "$extracted_app/Contents/MacOS/KneeXrayMeasurement" \
  --smoke-test-image "$smoke_image" --side R --model-mode tka
env -i HOME="$smoke_home" PATH="/usr/bin:/bin" TMPDIR="${TMPDIR:-/tmp}" \
  "$extracted_app/Contents/MacOS/KneeXrayMeasurement" \
  --smoke-test-image "$smoke_image" --side R --model-mode mixed
env -i HOME="$smoke_home" PATH="/usr/bin:/bin" TMPDIR="${TMPDIR:-/tmp}" \
  "$extracted_app/Contents/MacOS/KneeXrayMeasurement" \
  --smoke-test-image "$smoke_bone_image" --side R --model-mode auto
env -i HOME="$smoke_home" PATH="/usr/bin:/bin" TMPDIR="${TMPDIR:-/tmp}" \
  "$extracted_app/Contents/MacOS/KneeXrayMeasurement" \
  --smoke-test-image "$smoke_tka_image" --side R --model-mode auto
env -i HOME="$smoke_home" PATH="/usr/bin:/bin" TMPDIR="${TMPDIR:-/tmp}" \
  "$extracted_app/Contents/MacOS/KneeXrayMeasurement" \
  --smoke-test-image "$smoke_image" --side R --model-mode auto
env -i HOME="$smoke_home" PATH="/usr/bin:/bin" TMPDIR="${TMPDIR:-/tmp}" \
  "$extracted_app/Contents/MacOS/KneeXrayMeasurement" \
  --smoke-test-image "$smoke_tka_image" --side R --model-mode bone

echo
echo "Build finished: $release_zip"
echo "This build is Apple Silicon arm64 and ad-hoc signed."
