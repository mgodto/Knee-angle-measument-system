@echo off
setlocal EnableExtensions

cd /d %~dp0
set PYTHONPATH=
set PYTHONHOME=

set "APP_VERSION=0.2.4"
set "RELEASE_DATE=20260720"
set "EXPECTED_MODEL_SHA=f0cfa67f34691f3d81da0e10f0d6ff753dcf71f5aafd278ddb6bf146efc6ba45"
set "RELEASE_ROOT=KneeXrayMeasurement-Windows-x64-v%APP_VERSION%-%RELEASE_DATE%"
set "RELEASE_ZIP=dist\%RELEASE_ROOT%.zip"
set "RELEASE_STAGE=release stage\%RELEASE_ROOT%"
set "SMOKE_IMAGE=%TEMP%\KneeXrayMeasurement-non-clinical-smoke-%RELEASE_DATE%.png"

if not exist models\current.pt (
  echo Missing models\current.pt
  exit /b 1
)

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install x64 Python 3.10-3.12 first.
  exit /b 1
)

python -c "import platform,struct,sys; assert (3,10) <= sys.version_info[:2] < (3,13), 'Python 3.10-3.12 is required'; assert struct.calcsize('P') == 8 and platform.machine().upper() in {'AMD64','X86_64'}, 'Windows x64 Python is required'"
if errorlevel 1 exit /b 1
python -c "import tkinter; print('Tk', tkinter.TkVersion)"
if errorlevel 1 exit /b 1
python -m venv --clear .venv-measurement
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
python create_release_smoke_fixture.py "%SMOKE_IMAGE%"
if errorlevel 1 exit /b 1
python knee_measurement_app.py --smoke-test-image "%SMOKE_IMAGE%" --side R
if errorlevel 1 exit /b 1

if exist dist\KneeXrayMeasurement rmdir /s /q dist\KneeXrayMeasurement
if exist "%RELEASE_ZIP%" del /f /q "%RELEASE_ZIP%"
if exist "release stage" rmdir /s /q "release stage"

pyinstaller --noconfirm --clean knee_measurement_app.spec
if errorlevel 1 exit /b 1
dist\KneeXrayMeasurement\KneeXrayMeasurement.exe --validate-model
if errorlevel 1 exit /b 1
dist\KneeXrayMeasurement\KneeXrayMeasurement.exe --smoke-test-image "%SMOKE_IMAGE%" --side R
if errorlevel 1 exit /b 1

mkdir "%RELEASE_STAGE%"
if errorlevel 1 exit /b 1
xcopy /E /I /H /Y dist\KneeXrayMeasurement\* "%RELEASE_STAGE%\" >nul
if errorlevel 1 exit /b 1
copy /Y README_DOCTOR_JA.txt "%RELEASE_STAGE%\README_DOCTOR_JA.txt" >nul
if errorlevel 1 exit /b 1

"%RELEASE_STAGE%\KneeXrayMeasurement.exe" --validate-model
if errorlevel 1 exit /b 1
"%RELEASE_STAGE%\KneeXrayMeasurement.exe" --smoke-test-image "%SMOKE_IMAGE%" --side R
if errorlevel 1 exit /b 1

powershell -NoProfile -Command "Compress-Archive -CompressionLevel Optimal -Path 'release stage\%RELEASE_ROOT%' -DestinationPath '%RELEASE_ZIP%'"
if errorlevel 1 exit /b 1
python audit_measurement_archive.py "%RELEASE_ZIP%" --platform windows --expected-root "%RELEASE_ROOT%" --expected-model-sha256 "%EXPECTED_MODEL_SHA%" --expected-app-version "%APP_VERSION%"
if errorlevel 1 exit /b 1

echo.
echo Build finished: %RELEASE_ZIP%
echo Keep the extracted KneeXrayMeasurement.exe and _internal folder together.
