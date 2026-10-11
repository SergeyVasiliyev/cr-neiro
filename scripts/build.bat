@echo off
rem Configures and builds the project with MSVC + Ninja (Release) into .\build
setlocal
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
rem Prefer a stable Visual Studio; fall back to a preview/Insiders build.
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSPATH=%%i"
if not defined VSPATH for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSPATH=%%i"
if not defined VSPATH (
  echo Visual Studio with C++ tools not found
  exit /b 1
)
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

cd /d "%~dp0.."
rem Build the Python module against the project venv when it exists.
if exist ".venv\Scripts\python.exe" (
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release "-DPython_EXECUTABLE=%CD%\.venv\Scripts\python.exe" || exit /b 1
) else (
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release || exit /b 1
)
cmake --build build || exit /b 1
