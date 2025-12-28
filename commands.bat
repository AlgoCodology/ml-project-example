@echo off
REM ML Project Commands for Windows
REM Usage: commands.bat [command]

if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="install" goto install
if "%1"=="install-dev" goto install-dev
if "%1"=="train" goto train
if "%1"=="train-prod" goto train-prod
if "%1"=="inference" goto inference
if "%1"=="test" goto test
if "%1"=="format" goto format
if "%1"=="lint" goto lint
if "%1"=="clean" goto clean
if "%1"=="mlflow" goto mlflow
goto help

:help
echo Available commands:
echo   install        - Install production dependencies
echo   install-dev    - Install development dependencies
echo   train          - Train model locally
echo   train-prod     - Train model with production config
echo   inference      - Run inference locally
echo   test           - Run tests
echo   format         - Format code
echo   lint           - Run linters
echo   clean          - Clean temporary files
echo   mlflow         - Start MLflow UI
goto end

:install
python -m pip install --upgrade pip
pip install -r requirements-prod.txt
echo Installation complete!
goto end

:install-dev
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
echo Development installation complete!
goto end

:train
python src/entrypoint/train.py --config config/local.yaml
goto end

:train-prod
python src/entrypoint/train.py --config config/prod.yaml
goto end

:inference
python src/entrypoint/inference.py
goto end

:test
pytest tests/ -v --cov=src --cov-report=html --cov-report=term
goto end

:format
black src/ tests/
isort src/ tests/
echo Code formatted!
goto end

:lint
flake8 src/ tests/
pylint src/ tests/
echo Linting complete!
goto end

:clean
echo Cleaning temporary files...
for /d /r %%i in (__pycache__) do @if exist "%%i" rd /s /q "%%i"
for /d /r %%i in (*.egg-info) do @if exist "%%i" rd /s /q "%%i"
for /r %%i in (*.pyc) do @if exist "%%i" del /q "%%i"
for /r %%i in (*.pyo) do @if exist "%%i" del /q "%%i"
if exist .pytest_cache rd /s /q .pytest_cache
if exist .coverage del /q .coverage
if exist htmlcov rd /s /q htmlcov
echo Cleanup complete!
goto end

:mlflow
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
goto end

:end