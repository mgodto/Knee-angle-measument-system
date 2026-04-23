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
echo "Output folder: dist/KneeAnnotationTool"
