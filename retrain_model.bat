@echo off
title Retrain SUPER-MODEL
cd /d "%~dp0"
call .venv\Scripts\activate
echo Training started...
python impulse_fib_trader\main_combined_train.py
echo Training finished. New model saved.
pause
