@echo off
rem Start TintaPhoto from the source tree, without typing anything.
rem Double-click it in Explorer, or run it from a terminal: either way it
rem works from whatever folder it is called in.
setlocal
pushd "%~dp0"

set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" goto :no_venv

"%PYTHON%" main.py
if errorlevel 1 goto :failed

popd
endlocal
exit /b 0

:no_venv
echo.
echo There is no virtual environment in this folder, so the program cannot
echo start. Create it once, and this file will work from then on:
echo.
echo     py -3.12 -m venv .venv
echo     .venv\Scripts\python.exe -m pip install -e ".[dev]"
echo.
pause
popd
endlocal
exit /b 1

:failed
echo.
echo TintaPhoto stopped with an error. What happened is written down in
echo     %%APPDATA%%\TintaPhoto\logs
echo.
pause
popd
endlocal
exit /b 1
