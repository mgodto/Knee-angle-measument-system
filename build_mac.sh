#!/bin/zsh

set -euo pipefail

cd "$(dirname "$0")"
project_root="$PWD"

remove_macos_metadata() {
  local target="$1"
  [[ -e "$target" ]] || return 0
  find "$target" \( -name .DS_Store -o -name '._*' -o -name __MACOSX \) -exec rm -rf {} +
}

python3 -m venv .venv-build
source .venv-build/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-packaging.txt
export PYINSTALLER_CONFIG_DIR="$PWD/.pyinstaller-cache"
pyinstaller --noconfirm --clean knee_annotation_tool.spec

if [[ -d "dist/KneeAnnotationTool.app" ]]; then
  remove_macos_metadata dist
  rm -f KneeAnnotationTool-macOS.zip
  package_root="$(mktemp -d "${TMPDIR:-/tmp}/knee-annotation-macos.XXXXXX")"
  trap 'rm -rf "$package_root"' EXIT
  mkdir -p "$package_root/dist"
  rsync -a --exclude .DS_Store --exclude '._*' --exclude __MACOSX dist/KneeAnnotationTool.app "$package_root/dist/"
  if [[ -d "dist/KneeAnnotationTool" ]]; then
    rsync -a --exclude .DS_Store --exclude '._*' --exclude __MACOSX dist/KneeAnnotationTool "$package_root/dist/"
  fi
  remove_macos_metadata "$package_root"
  (
    cd "$package_root"
    COPYFILE_DISABLE=1 zip -qry -X "$project_root/KneeAnnotationTool-macOS.zip" dist
  )
  if zipinfo -1 KneeAnnotationTool-macOS.zip | grep -E '(^|/)\._|(^|/)__MACOSX(/|$)' >/dev/null; then
    echo "Error: KneeAnnotationTool-macOS.zip contains AppleDouble metadata." >&2
    exit 1
  fi
fi

echo
echo "Build finished."
if [[ -d "dist/KneeAnnotationTool.app" ]]; then
  echo "Output app: dist/KneeAnnotationTool.app"
  echo "Send KneeAnnotationTool-macOS.zip to the doctor. It contains the same dist/ layout as the previous working package."
else
  echo "Output folder: dist/KneeAnnotationTool"
fi
