@echo off
title Review Forensics - Fake Product Review Detector
echo ========================================================
echo Starting Review Forensics AI Server...
echo Opening http://127.0.0.1:5000 in your browser...
echo ========================================================
cd /d "%~dp0"
start http://127.0.0.1:5000
python app.py
pause
