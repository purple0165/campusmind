@echo off
cd /d "%~dp0"

set PYTHONPATH=%PYTHONPATH%;%~dp0

echo Starting CampusMind Web...
echo.

python backend/main.py

pause