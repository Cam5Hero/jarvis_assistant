@echo off
rem Undo what setup.bat and the Jarvis app installed. Never touches Python itself or LM Studio.
setlocal EnableExtensions
cd /d "%~dp0"
title Jarvis uninstall
set "JV_DIR=%~dp0"
set "JV_DIR=%JV_DIR:~0,-1%"

rem Hugging Face model cache (where Kokoro and Whisper voice/speech models are downloaded)
if defined HF_HUB_CACHE (set "HUB=%HF_HUB_CACHE%") else if defined HF_HOME (set "HUB=%HF_HOME%\hub") else set "HUB=%USERPROFILE%\.cache\huggingface\hub"

echo ============================================
echo   Jarvis uninstall
echo ============================================
echo.
echo This will remove:
echo   - Jarvis if it is running - only the copy started from this folder
echo   - the "Jarvis" shortcut on your Desktop and in your Startup folder
echo     - only if it points to this folder
echo   - the .venv folder: all packages setup.bat installed, plus the wake word models
echo   - the .cache folder: app list and saved app settings, and __pycache__
echo   - the log file jarvis_gui.log
echo   - the downloaded Kokoro and Whisper models in:
echo     %HUB%
echo     - only the Jarvis model folders, nothing else in there
echo.
echo It will ask separately about your .env file, which holds your settings and API keys.
echo Python and LM Studio are NOT removed.
echo.
set "ANS="
set /p "ANS=Type Y and press Enter to continue, or just press Enter to cancel: "
if /i not "%ANS%"=="Y" (
    echo Cancelled. Nothing was removed.
    pause
    exit /b 0
)
echo.

rem ---- 1. Stop Jarvis (only Python processes from this folder's .venv, and their child processes) ----
echo Stopping Jarvis if it is running...
rem Waits up to 10 seconds for Jarvis to close by itself, e.g. after the in-app Uninstall button.
rem Only python.exe/pythonw.exe are matched, so this cmd window is never closed.
powershell -NoProfile -Command "$v = (Join-Path $env:JV_DIR '.venv') + '\'; function Find-Jarvis { $all = @(Get-CimInstance Win32_Process -Filter \"Name='python.exe' or Name='pythonw.exe'\"); $ids = @($all | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($v, [StringComparison]::OrdinalIgnoreCase) } | ForEach-Object { $_.ProcessId }); $kids = @($all | Where-Object { $ids -contains $_.ParentProcessId } | ForEach-Object { $_.ProcessId }); @($ids + $kids | Select-Object -Unique) }; $t = 0; while ((Find-Jarvis).Count -gt 0 -and $t -lt 10) { Start-Sleep -Seconds 1; $t++ }; $left = Find-Jarvis; if ($left.Count -eq 0) { Write-Host '  Jarvis is not running.' }; foreach ($p in $left) { Stop-Process -Id $p -Force -ErrorAction SilentlyContinue; Write-Host ('  stopped process ' + $p) }"
timeout /t 2 /nobreak >nul

rem ---- 2. Shortcuts (Desktop + Startup), only if they point into this folder ----
echo Removing shortcuts...
powershell -NoProfile -Command "$sh = New-Object -ComObject WScript.Shell; $dirs = @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Startup')); foreach ($d in $dirs) { $lnk = Join-Path $d 'Jarvis.lnk'; if (Test-Path $lnk) { $s = $sh.CreateShortcut($lnk); if (($s.TargetPath + ' ' + $s.Arguments + ' ' + $s.WorkingDirectory) -like ('*' + $env:JV_DIR + '*')) { Remove-Item $lnk -Force; Write-Host ('  removed ' + $lnk) } else { Write-Host ('  kept ' + $lnk + ' - it belongs to a different Jarvis folder') } } }"

rem ---- 3. Files inside this folder ----
echo Removing installed packages and cache files in this folder...
if exist ".venv" rmdir /s /q ".venv"
if exist ".venv" echo   Could not fully delete .venv - close Jarvis and run uninstall.bat again.
if exist ".cache" rmdir /s /q ".cache"
if exist "__pycache__" rmdir /s /q "__pycache__"
if exist "jarvis_gui.log" del /q "jarvis_gui.log"
if exist "kokoro_install.log" del /q "kokoro_install.log"
if exist "setup.log" del /q "setup.log"

rem ---- 4. Jarvis model downloads (exact folder names only) ----
echo Removing downloaded Jarvis models...
set "MODELS=models--hexgrad--Kokoro-82M"
for %%S in (tiny tiny.en base base.en small small.en medium medium.en large-v1 large-v2 large-v3) do call set "MODELS=%%MODELS%% models--Systran--faster-whisper-%%S"
for %%S in (small.en medium.en large-v2 large-v3) do call set "MODELS=%%MODELS%% models--Systran--faster-distil-whisper-%%S"
set "MODELS=%MODELS% models--mobiuslabsgmbh--faster-whisper-large-v3-turbo models--deepdml--faster-whisper-large-v3-turbo-ct2"
for %%M in (%MODELS%) do (
    if exist "%HUB%\%%M" (
        rmdir /s /q "%HUB%\%%M"
        echo   removed %%M
    )
    if exist "%HUB%\.locks\%%M" rmdir /s /q "%HUB%\.locks\%%M"
)
echo.

rem ---- 5. .env (settings + API keys) - default keep ----
if exist ".env" (
    choice /C YN /N /M "Also delete your .env file with your settings and API keys? [Y/N, recommended N]: "
    if errorlevel 2 (
        echo   Kept .env
    ) else (
        del /q ".env"
        if exist ".env.bak" del /q ".env.bak"
        echo   Deleted .env
    )
)

echo.
echo ============================================
echo   Done. Jarvis has been uninstalled.
echo ============================================
echo Python and LM Studio were NOT removed. If you don't need them any more,
echo uninstall them from Windows Settings, then Apps, then Installed apps.
echo LM Studio's downloaded AI models are kept by LM Studio - remove them in LM Studio first if you want the space back.
echo.

rem ---- 6. Offer to delete this folder too (sent to the Recycle Bin, so it can be restored) ----
powershell -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms; $m = 'Delete the Jarvis folder too?' + [char]10 + [char]10 + 'This removes everything in:' + [char]10 + $env:JV_DIR + [char]10 + [char]10 + 'including your .env keys if you kept them. The folder goes to the Recycle Bin, so you can still restore it from there.'; $r = [System.Windows.Forms.MessageBox]::Show($m, 'Jarvis uninstall', 'YesNo', 'Warning', 'Button2'); if ($r -eq 'Yes') { exit 7 } else { exit 0 }"
if not errorlevel 7 goto :keep_folder

rem Safety checks: a real folder (not a drive root) that contains this uninstall.bat and jarvis_gui.py.
if "%JV_DIR%"=="" goto :keep_folder
if "%JV_DIR:~-1%"==":" goto :keep_folder
if not exist "%JV_DIR%\uninstall.bat" goto :keep_folder
if not exist "%JV_DIR%\jarvis_gui.py" goto :keep_folder

echo Moving the Jarvis folder to the Recycle Bin in a few seconds. You can close this window.
cd /d "%TEMP%"
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 3; $d = $env:JV_DIR; if ($d -and (Split-Path $d -Parent) -and (Test-Path -LiteralPath (Join-Path $d 'jarvis_gui.py')) -and (Test-Path -LiteralPath (Join-Path $d 'uninstall.bat'))) { Add-Type -AssemblyName Microsoft.VisualBasic; [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory($d, 'OnlyErrorDialogs', 'SendToRecycleBin') }"
endlocal
exit

:keep_folder
echo Kept the Jarvis folder. You can delete it yourself any time:
echo   %JV_DIR%
pause
endlocal
