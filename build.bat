@echo off
REM ============================================================
REM  build.bat  -  Run this ONCE to produce LiveTranslator.exe
REM  Requirements: Python 3.11+ installed, pip available
REM ============================================================

echo [1/4] Installing Python dependencies...
pip install -r requirements.txt
pip install pyinstaller pystray Pillow

echo [2/4] Building LiveTranslator.exe ...
pyinstaller --noconfirm --onefile --windowed --name "LiveTranslator" --add-data ".env;." launcher.py

echo [3/4] Copying .env into dist folder...
if exist .env (
    copy .env dist\.env
) else (
    echo WARNING: .env file not found. Copy it manually into the dist folder.
)

echo [4/4] Done!
echo.
echo Your executable is at:  dist\LiveTranslator.exe
echo.
echo Share the entire dist\ folder with your team.
echo Each team member just double-clicks LiveTranslator.exe
echo.
pause
