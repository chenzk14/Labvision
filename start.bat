@echo off
chcp 65001 >nul
REM Quick start backend service (Windows)
echo Starting Reagent Vision System Backend...
conda activate reagent_vision
cd /d %~dp0
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload