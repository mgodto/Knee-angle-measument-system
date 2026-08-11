#!/bin/zsh

set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
cd "$project_root"

remove_macos_metadata() {
  local target="$1"
  [[ -e "$target" ]] || return 0
  find "$target" \( -name .DS_Store -o -name '._*' -o -name __MACOSX \) -exec rm -rf {} +
}

python_bin="${PYTHON_BIN:-python3}"
annotation_version="1.2"
annotation_build="3"
build_timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
pyinstaller_dist="$PWD/build/pyinstaller-dist"
pyinstaller_work="$PWD/build/pyinstaller-work"
delivery_dir="$PWD/deliverables/annotation/macos"
delivery_stage="$PWD/build/deliverable-stage/annotation/macos"
release_name="KneeAnnotationTool-macOS-v${annotation_version}-build${annotation_build}-${build_timestamp}.zip"
candidate_zip="$delivery_stage/$release_name"
release_zip="$delivery_dir/$release_name"

"$python_bin" -m venv --clear .venv-build
source .venv-build/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/requirements-packaging.txt
export PYINSTALLER_CONFIG_DIR="$PWD/.pyinstaller-cache"
rm -rf "$pyinstaller_dist/KneeAnnotationTool" "$pyinstaller_dist/KneeAnnotationTool.app" \
  "$pyinstaller_work/knee_annotation_tool"
rm -f "$candidate_zip"
mkdir -p "$delivery_dir" "$delivery_stage"
pyinstaller --noconfirm --clean \
  --distpath "$pyinstaller_dist" \
  --workpath "$pyinstaller_work" \
  packaging/knee_annotation_tool.spec

if [[ -d "$pyinstaller_dist/KneeAnnotationTool.app" ]]; then
  remove_macos_metadata "$pyinstaller_dist"
  package_root="$(mktemp -d "${TMPDIR:-/tmp}/knee-annotation-macos.XXXXXX")"
  trap 'rm -rf "$package_root"' EXIT
  mkdir -p "$package_root/dist"
  rsync -a --exclude .DS_Store --exclude '._*' --exclude __MACOSX \
    "$pyinstaller_dist/KneeAnnotationTool.app" "$package_root/dist/"
  if [[ -d "$pyinstaller_dist/KneeAnnotationTool" ]]; then
    rsync -a --exclude .DS_Store --exclude '._*' --exclude __MACOSX \
      "$pyinstaller_dist/KneeAnnotationTool" "$package_root/dist/"
  fi
  remove_macos_metadata "$package_root"
  (
    cd "$package_root"
    COPYFILE_DISABLE=1 zip -qry -X "$candidate_zip" dist
  )
  unzip -tq "$candidate_zip"
  if zipinfo -1 "$candidate_zip" | grep -E '(^|/)\._|(^|/)__MACOSX(/|$)' >/dev/null; then
    echo "Error: $candidate_zip contains AppleDouble metadata." >&2
    exit 1
  fi
  mv -f "$candidate_zip" "$release_zip"
  python -m knee_xray.release.retain_deliverables annotation macos \
    --repo-root "$project_root" --current "$release_zip"
else
  echo "Error: expected app was not created: $pyinstaller_dist/KneeAnnotationTool.app" >&2
  exit 1
fi

echo
echo "Build finished."
echo "Intermediate app: $pyinstaller_dist/KneeAnnotationTool.app"
echo "Send $release_zip to the doctor. It contains the same dist/ layout as the previous working package."
