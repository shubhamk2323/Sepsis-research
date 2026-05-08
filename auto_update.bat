@echo off
echo ==============================================
echo       GitHub Auto-Update Script
echo ==============================================
echo.

cd /d "%~dp0"

echo Uploading to GitHub...
git add .
git commit -m "Auto-update from local machine: %date% %time%"
git push origin main

echo.
echo Upload Process Complete!
echo.
pause
