@echo off
echo Running Evaluation...
set /p cat="Enter category (e.g. bottle): "
py src/evaluate.py %cat%
pause
