@echo off
setlocal

cd /d %~dp0

where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher ^(py^) was not found. Please install Python 3 for Windows first.
  exit /b 1
)

if not exist .venv\Scripts\python.exe (
  py -3 -m venv .venv
  if errorlevel 1 exit /b 1
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1

python -m pip install -r requirements-packaging.txt
if errorlevel 1 exit /b 1

pyinstaller --noconfirm --clean knee_annotation_tool.spec
if errorlevel 1 exit /b 1

echo.
echo Build finished.
echo Send the whole folder:
echo   dist\KneeAnnotationTool
echo.
echo The doctor should run:
echo   dist\KneeAnnotationTool\KneeAnnotationTool.exe
