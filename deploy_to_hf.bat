@echo off
REM Deploy to Hugging Face Spaces - Windows Version

echo ==========================================
echo Deploy to Hugging Face Spaces
echo ==========================================
echo.

set /p HF_USERNAME="Enter your Hugging Face username: "
set /p SPACE_NAME="Enter your Space name: "

echo.
echo Adding Hugging Face remote...

git remote add hf https://huggingface.co/spaces/%HF_USERNAME%/%SPACE_NAME%

echo Remote added!
echo.
echo Now pushing to Hugging Face...
echo You will be prompted for:
echo   Username: %HF_USERNAME%
echo   Password: Your HF token (starts with hf_...)
echo.

git push hf main --force

echo.
echo ==========================================
echo Deployment complete!
echo Check your Space at:
echo https://huggingface.co/spaces/%HF_USERNAME%/%SPACE_NAME%
echo ==========================================
echo.
pause
