@echo off
setlocal

cd /d %~dp0
set PYTHONPATH=
set PYTHONHOME=

if not exist models\current.pt (
  echo Missing models\current.pt
  echo Copy the validated production-compatible checkpoint there before building.
  exit /b 1
)

where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher ^(py^) was not found. Please install Python 3 first.
  exit /b 1
)

py -3 -c "import sys; assert sys.version_info[:2] in ((3, 10), (3, 11), (3, 12)), 'Python 3.10-3.12 is required'"
if errorlevel 1 exit /b 1
py -3 -c "import tkinter; print('Tk', tkinter.TkVersion)"
if errorlevel 1 exit /b 1
py -3 -m venv --clear .venv-measurement
if errorlevel 1 exit /b 1

call .venv-measurement\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1
python -m pip install -r requirements-app.txt
if errorlevel 1 exit /b 1
python -m pip check
if errorlevel 1 exit /b 1
python -m unittest discover -s tests -v
if errorlevel 1 exit /b 1
python validate_app_model.py
if errorlevel 1 exit /b 1
python knee_measurement_app.py --smoke-test-image images\annotation_processed_combined\015R_pre_bone_raw.jpg --side R
if errorlevel 1 exit /b 1
pyinstaller --noconfirm --clean knee_measurement_app.spec
if errorlevel 1 exit /b 1
dist\KneeXrayMeasurement\KneeXrayMeasurement.exe --validate-model
if errorlevel 1 exit /b 1
dist\KneeXrayMeasurement\KneeXrayMeasurement.exe --smoke-test-image images\annotation_processed_combined\015R_pre_bone_raw.jpg --side R
if errorlevel 1 exit /b 1
powershell -NoProfile -Command "Compress-Archive -Force -Path 'dist\KneeXrayMeasurement' -DestinationPath 'dist\KneeXrayMeasurement-Windows-x64.zip'"
if errorlevel 1 exit /b 1

echo.
echo Build finished: dist\KneeXrayMeasurement
echo Distribution archive: dist\KneeXrayMeasurement-Windows-x64.zip
echo Run KneeXrayMeasurement.exe with a non-PHI image before distribution.
