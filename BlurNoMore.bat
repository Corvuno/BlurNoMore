@echo off
setlocal
cd /d "%~dp0"
title BlurNoMore

set "PY="
py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY python --version >nul 2>&1 && set "PY=python"
if not defined PY goto nopython
if not exist "%~dp0blurnomore.py" goto noscript

echo Checking Python packages (first time takes a minute)...
%PY% -m pip install --quiet --disable-pip-version-check pillow numpy img2pdf
if errorlevel 1 goto piperr
rem Optional: the Web photo model (DAT) and face restoration. If this fails the program still starts.
%PY% -m pip install --quiet --disable-pip-version-check onnxruntime-directml opencv-python-headless
if errorlevel 1 echo Note: the optional packages onnxruntime-directml and opencv-python-headless were not installed. The Web photo model and face restoration are off; everything else works.

echo Opening BlurNoMore...
%PY% "%~dp0blurnomore.py"
if errorlevel 1 (
  echo.
  echo The program stopped with an error. Scroll up and copy the text above.
  pause
)
exit /b

:nopython
echo Python was not found. Install it from python.org and tick "Add python.exe to PATH".
pause
exit /b 1

:noscript
echo blurnomore.py must be in the same folder as this .bat file.
pause
exit /b 1

:piperr
echo Installing the Python packages failed. Scroll up and copy the text above.
pause
exit /b 1
