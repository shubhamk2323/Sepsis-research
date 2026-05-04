@echo off
echo ==============================================
echo       GitHub Auto-Update Script
echo ==============================================
echo.

cd /d "%~dp0"

echo Checking for changes...
git status -s > temp_status.txt
set /p CHANGES=<temp_status.txt
del temp_status.txt

if "%CHANGES%"=="" (
    echo.
    echo No new changes found to upload. You are up to date!
) else (
    echo.
    echo New changes found. Uploading to GitHub...
    git add .
    git commit -m "Auto-update from local machine: %date% %time%"
    git push origin main
    echo.
    echo Upload Successful!
)

echo.
pause
