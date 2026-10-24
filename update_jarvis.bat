@echo off
cd /d "%~dp0"
echo Updating Jarvis from GitHub...
git pull
if errorlevel 1 (
  echo.
  echo Update failed. Make sure Git is installed and you have no unsaved edits to the code.
  pause
  exit /b 1
)
echo.
echo Installing any new packages...
python -m pip install -r requirements.txt
echo.
echo Jarvis is up to date!
pause
