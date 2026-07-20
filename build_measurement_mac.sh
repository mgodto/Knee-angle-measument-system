#!/bin/zsh

set -euo pipefail

cd "$(dirname "$0")"
unset PYTHONPATH PYTHONHOME

app_version="0.3.0"
release_date="20260720"
expected_bone_sha="24481410c3fd2ce2222eed422f4d519f72da95ba232570ee31a4827d45201cfd"
expected_tka_sha="87027887ec091068a9b91b01a881092400fed58eb8d3eeaaeddb10e8be398e5f"
expected_mixed_sha="f0cfa67f34691f3d81da0e10f0d6ff753dcf71f5aafd278ddb6bf146efc6ba45"
release_root="KneeXrayMeasurement-macOS-arm64-v${app_version}-${release_date}"
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
python -c 'from knee_measurement_app import APP_VERSION; assert APP_VERSION == "0.3.0", APP_VERSION'
python knee_measurement_app.py --validate-models

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
  "$extracted_app/Contents/MacOS/KneeXrayMeasurement" --validate-models
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
