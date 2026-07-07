@echo off
echo ============================================
echo    VoxGuard — Deepfake Audio Detector
echo ============================================
echo.
echo Starting dashboard...
echo.
cd /d D:\Projects\DeepFake\deepfake_voice_conversion
call venv\Scripts\activate.bat
cd dashboard
streamlit run app.py
pause