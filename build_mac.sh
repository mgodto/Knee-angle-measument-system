#!/bin/zsh

set -euo pipefail

cd "$(dirname "$0")"

python3 -m venv .venv-build
source .venv-build/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-packaging.txt
pyinstaller --noconfirm --clean knee_annotation_tool.spec

echo
echo "Build finished."
if [[ -d "dist/KneeAnnotationTool.app" ]]; then
  echo "Output app: dist/KneeAnnotationTool.app"
  echo "Send dist/KneeAnnotationTool.app to the doctor. Do not send the development launcher at project root."
else
  echo "Output folder: dist/KneeAnnotationTool"
fi
