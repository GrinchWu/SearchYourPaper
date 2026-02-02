@echo off
echo Building AI Academic Assistant...
cd /d %~dp0
pyinstaller --noconfirm --onedir --windowed --name "AI学术助手" --add-data "src;src" src/main.py
echo Build complete! Check dist folder.
pause
