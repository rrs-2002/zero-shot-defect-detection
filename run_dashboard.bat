@echo off
echo Starting Zero-Shot Defect Detection Dashboard...

:: 1. Activate the virtual environment
call .\.venv\Scripts\activate

:: 2. Set Flask variables
set FLASK_APP=run.py
set FLASK_ENV=development

:: 3. Run the app using the virtual environment's Python
python run.py

pause