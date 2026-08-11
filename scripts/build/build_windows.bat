@echo off
setlocal

cd /d "%~dp0\..\.."

set "ANNOTATION_VERSION=1.2"
set "ANNOTATION_BUILD=3"
set "PYINSTALLER_DIST=build\pyinstaller-dist"
set "PYINSTALLER_WORK=build\pyinstaller-work"
set "DELIVERY_DIR=deliverables\annotation\windows"
set "DELIVERY_STAGE=build\deliverable-stage\annotation\windows"
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Date ([DateTime]::UtcNow) -Format yyyyMMddTHHmmssZ"') do set "BUILD_TIMESTAMP=%%I"
if not defined BUILD_TIMESTAMP (
  echo Could not create a UTC build timestamp.
  exit /b 1
)
set "RELEASE_NAME=KneeAnnotationTool-Windows-x64-v%ANNOTATION_VERSION%-build%ANNOTATION_BUILD%-%BUILD_TIMESTAMP%.zip"
set "CANDIDATE_ZIP=%DELIVERY_STAGE%\%RELEASE_NAME%"
set "RELEASE_ZIP=%DELIVERY_DIR%\%RELEASE_NAME%"

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

python -m pip install -r requirements\requirements-packaging.txt
if errorlevel 1 exit /b 1

if exist "%PYINSTALLER_DIST%\KneeAnnotationTool" rmdir /s /q "%PYINSTALLER_DIST%\KneeAnnotationTool"
if exist "%PYINSTALLER_WORK%\knee_annotation_tool" rmdir /s /q "%PYINSTALLER_WORK%\knee_annotation_tool"
if exist "%CANDIDATE_ZIP%" del /f /q "%CANDIDATE_ZIP%"
if not exist "%DELIVERY_DIR%" mkdir "%DELIVERY_DIR%"
if errorlevel 1 exit /b 1
if not exist "%DELIVERY_STAGE%" mkdir "%DELIVERY_STAGE%"
if errorlevel 1 exit /b 1

pyinstaller --noconfirm --clean --distpath "%PYINSTALLER_DIST%" --workpath "%PYINSTALLER_WORK%" packaging\knee_annotation_tool.spec
if errorlevel 1 exit /b 1

powershell -NoProfile -ExecutionPolicy Bypass -Command "Compress-Archive -Path '%PYINSTALLER_DIST%\KneeAnnotationTool\*' -DestinationPath '%CANDIDATE_ZIP%' -Force"
if errorlevel 1 exit /b 1
powershell -NoProfile -Command "Add-Type -AssemblyName System.IO.Compression.FileSystem; $archive = [IO.Compression.ZipFile]::OpenRead('%CANDIDATE_ZIP%'); try { if ($archive.Entries.Count -eq 0) { throw 'Archive is empty.' } } finally { $archive.Dispose() }"
if errorlevel 1 exit /b 1
move /Y "%CANDIDATE_ZIP%" "%RELEASE_ZIP%" >nul
if errorlevel 1 exit /b 1
python -m knee_xray.release.retain_deliverables annotation windows --repo-root "%CD%" --current "%RELEASE_ZIP%"
if errorlevel 1 exit /b 1

echo.
echo Build finished.
echo Send this zip file:
echo   %RELEASE_ZIP%
echo.
echo Intermediate folder:
echo   %PYINSTALLER_DIST%\KneeAnnotationTool
echo.
echo The doctor should run:
echo   KneeAnnotationTool.exe
