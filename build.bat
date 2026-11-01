@echo off
rem Build the release, without typing anything: the icon, the program, the
rem installer and the zip, in the only order that works. Everything lands
rem in dist\, and the window stays open at the end so that what it made can
rem be read. It takes about a minute and a half.
setlocal
pushd "%~dp0"

set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" goto :no_venv

"%PYTHON%" tools\build_release.py
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" (
    echo.
    echo The build stopped with an error. Nothing in dist\ can be trusted
    echo until it has run through to the end.
)
echo.
pause
popd
endlocal & exit /b %RESULT%

:no_venv
echo.
echo There is no virtual environment in this folder, so there is nothing to
echo build with. Create it once, and this file will work from then on:
echo.
echo     py -3.12 -m venv .venv
echo     .venv\Scripts\python.exe -m pip install -e ".[dev]"
echo.
pause
popd
endlocal
exit /b 1
