@echo off
echo Running Unit Tests...
echo.
echo [1/2] Testing Model Loading...
py tests/test_model.py
echo.
echo [2/2] Testing Inference...
py tests/test_inference.py
echo.
echo Done.
pause
