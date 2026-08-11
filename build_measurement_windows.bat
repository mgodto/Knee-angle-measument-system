@echo off
setlocal EnableExtensions

cd /d %~dp0
set PYTHONPATH=
set PYTHONHOME=
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
chcp 65001 >nul

set "APP_VERSION=0.6.0"
set "RELEASE_DATE=20260811"
set "EXPECTED_BONE_VERSION=20260811-single-leg-v4-curated-tailqa-9e9a1de4fe98-bone-final-v1"
set "EXPECTED_TKA_VERSION=20260811-single-leg-v4-curated-tailqa-9e9a1de4fe98-tka-final-v1"
set "EXPECTED_MIXED_VERSION=20260811-single-leg-v4-curated-tailqa-9e9a1de4fe98-bone-tka-mixed-final-v1"
set "EXPECTED_BONE_SHA=36e8fee67c7c6bad8071a5a7ff8dbc713d76e28482c2a26798abd15ba3334862"
set "EXPECTED_TKA_SHA=23a416f8c376156b3fa323298d3e45ae00060d619e0215754a46f2a2254a1669"
set "EXPECTED_MIXED_SHA=5e2f5087a433aa6bc9c58d792e132d88f1098952df446512375818a779d1254e"
set "RELEASE_ROOT=KneeXrayMeasurement-ResearchCandidate-Windows-x64-v%APP_VERSION%-%RELEASE_DATE%"
set "RELEASE_ZIP=dist\%RELEASE_ROOT%.zip"
set "RELEASE_STAGE_ROOT=release-stage-measurement"
set "RELEASE_STAGE=%RELEASE_STAGE_ROOT%\%RELEASE_ROOT%"
set "WINDOWS_ENTRYPOINT=knee_measurement_app_windows.py"
set "SMOKE_IMAGE=%TEMP%\KneeXrayMeasurement-non-clinical-smoke-unknown-%RELEASE_DATE%.png"
set "SMOKE_BONE_IMAGE=%TEMP%\KneeXrayMeasurement-non-clinical-smoke-bone-%RELEASE_DATE%.png"
set "SMOKE_TKA_IMAGE=%TEMP%\KneeXrayMeasurement-non-clinical-smoke-TKA-%RELEASE_DATE%.png"

if not exist models\bone.pt (
  echo Missing models\bone.pt
  exit /b 1
)
if not exist models\tka.pt (
  echo Missing models\tka.pt
  exit /b 1
)
if not exist models\mixed.pt (
  echo Missing models\mixed.pt
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
python generate_windows_english_entrypoint.py --source knee_measurement_app.py --output "%WINDOWS_ENTRYPOINT%"
if errorlevel 1 exit /b 1
python -m py_compile "%WINDOWS_ENTRYPOINT%"
if errorlevel 1 exit /b 1
python -c "from knee_measurement_app_windows import APP_RELEASE_CHANNEL, APP_VERSION, WINDOWS_ENGLISH_BUILD; assert APP_VERSION == '0.6.0', APP_VERSION; assert APP_RELEASE_CHANNEL == 'INTERNAL RESEARCH CANDIDATE - NOT FOR CLINICAL USE', APP_RELEASE_CHANNEL; assert WINDOWS_ENGLISH_BUILD"
if errorlevel 1 exit /b 1
python "%WINDOWS_ENTRYPOINT%" --validate-models --expected-model-version bone=%EXPECTED_BONE_VERSION% --expected-model-version tka=%EXPECTED_TKA_VERSION% --expected-model-version mixed=%EXPECTED_MIXED_VERSION%
if errorlevel 1 exit /b 1
python create_release_smoke_fixture.py "%SMOKE_IMAGE%"
if errorlevel 1 exit /b 1
copy /Y "%SMOKE_IMAGE%" "%SMOKE_BONE_IMAGE%" >nul
if errorlevel 1 exit /b 1
copy /Y "%SMOKE_IMAGE%" "%SMOKE_TKA_IMAGE%" >nul
if errorlevel 1 exit /b 1
python "%WINDOWS_ENTRYPOINT%" --smoke-test-image "%SMOKE_IMAGE%" --side R --model-mode mixed
if errorlevel 1 exit /b 1

if exist dist\KneeXrayMeasurement rmdir /s /q dist\KneeXrayMeasurement
if exist "%RELEASE_ZIP%" del /f /q "%RELEASE_ZIP%"
if exist "%RELEASE_STAGE_ROOT%" rmdir /s /q "%RELEASE_STAGE_ROOT%"

pyinstaller --noconfirm --clean knee_measurement_app.spec
if errorlevel 1 exit /b 1
dist\KneeXrayMeasurement\KneeXrayMeasurement.exe --validate-models --expected-model-version bone=%EXPECTED_BONE_VERSION% --expected-model-version tka=%EXPECTED_TKA_VERSION% --expected-model-version mixed=%EXPECTED_MIXED_VERSION%
if errorlevel 1 exit /b 1

mkdir "%RELEASE_STAGE%"
if errorlevel 1 exit /b 1
xcopy /E /I /Y dist\KneeXrayMeasurement\* "%RELEASE_STAGE%\" >nul
if errorlevel 1 exit /b 1
copy /Y README_DOCTOR_EN.txt "%RELEASE_STAGE%\README_DOCTOR_EN.txt" >nul
if errorlevel 1 exit /b 1

"%RELEASE_STAGE%\KneeXrayMeasurement.exe" --validate-models --expected-model-version bone=%EXPECTED_BONE_VERSION% --expected-model-version tka=%EXPECTED_TKA_VERSION% --expected-model-version mixed=%EXPECTED_MIXED_VERSION%
if errorlevel 1 exit /b 1
"%RELEASE_STAGE%\KneeXrayMeasurement.exe" --smoke-test-image "%SMOKE_IMAGE%" --side R --model-mode bone
if errorlevel 1 exit /b 1
"%RELEASE_STAGE%\KneeXrayMeasurement.exe" --smoke-test-image "%SMOKE_IMAGE%" --side R --model-mode tka
if errorlevel 1 exit /b 1
"%RELEASE_STAGE%\KneeXrayMeasurement.exe" --smoke-test-image "%SMOKE_IMAGE%" --side R --model-mode mixed
if errorlevel 1 exit /b 1
"%RELEASE_STAGE%\KneeXrayMeasurement.exe" --smoke-test-image "%SMOKE_BONE_IMAGE%" --side R --model-mode auto
if errorlevel 1 exit /b 1
"%RELEASE_STAGE%\KneeXrayMeasurement.exe" --smoke-test-image "%SMOKE_TKA_IMAGE%" --side R --model-mode auto
if errorlevel 1 exit /b 1
"%RELEASE_STAGE%\KneeXrayMeasurement.exe" --smoke-test-image "%SMOKE_IMAGE%" --side R --model-mode auto
if errorlevel 1 exit /b 1
"%RELEASE_STAGE%\KneeXrayMeasurement.exe" --smoke-test-image "%SMOKE_TKA_IMAGE%" --side R --model-mode bone
if errorlevel 1 exit /b 1

powershell -NoProfile -Command "Compress-Archive -CompressionLevel Optimal -Path '%RELEASE_STAGE%' -DestinationPath '%RELEASE_ZIP%'"
if errorlevel 1 exit /b 1
python audit_measurement_archive.py "%RELEASE_ZIP%" --platform windows --expected-root "%RELEASE_ROOT%" --expected-model "bone.pt=%EXPECTED_BONE_SHA%" --expected-model "tka.pt=%EXPECTED_TKA_SHA%" --expected-model "mixed.pt=%EXPECTED_MIXED_SHA%" --expected-app-version "%APP_VERSION%"
if errorlevel 1 exit /b 1

echo.
echo Build finished: %RELEASE_ZIP%
echo Keep the extracted KneeXrayMeasurement.exe and _internal folder together.
