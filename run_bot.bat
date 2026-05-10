@echo off
title Crypto TAS-Fib Bot
cd /d "%~dp0"

echo [1/3] Checking Virtual Environment...
if not exist ".venv\Scripts\activate" (
    echo ERROR: Virtual environment not found in .venv folder!
    pause
    exit /b
)

echo [2/3] Activating environment...
call .venv\Scripts\activate

echo [3/3] Starting Telegram Bot...
echo Press Ctrl+C to stop the bot.
python impulse_fib_trader\telegram_bot.py

pause
