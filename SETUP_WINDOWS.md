# Setup Instructions for Windows

## Follow these steps to complete the windows setup if not already done:

## 1. Activate Virtual Environment

In PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```

In Command Prompt:
```cmd
.\venv\Scripts\activate.bat
```

In VS Code Terminal (PowerShell):
```powershell
.\venv\Scripts\Activate.ps1
```

**Note**: If you get an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 2. Install Dependencies

```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# Install development dependencies
pip install -r requirements-dev.txt
```

## 3. Create Configuration Files

Copy the content from the artifacts I provided into these files:
- config/local.yaml
- config/prod.yaml
- src/entrypoint/train.py
- src/entrypoint/inference.py
- src/pipelines/feature_eng_pipeline.py
- src/pipelines/training_pipeline.py
- src/utils/config.py
- src/utils/logger.py
- tests/test_training.py
- requirements-dev.txt
- requirements-prod.txt
- Dockerfile
- docker-compose.yml

## 4. VS Code Setup

- Open this folder in VS Code
- Select the Python interpreter: Ctrl+Shift+P → "Python: Select Interpreter" → Choose ./venv/Scripts/python.exe
- The workspace is already configured with proper settings

## 5. Verify Setup

```powershell
# Run tests
pytest tests/ -v

# Check Python can import your modules
python -c "import src; print('Success!')"
```

## 6. Start Developing

- Place your data in data/01-raw/
- Modify the pipelines for your use case
- Train your first model
- Run inference

## Common Windows-Specific Commands

Instead of the Makefile commands, use these PowerShell equivalents:

```powershell
# Train model
python src/entrypoint/train.py --config config/local.yaml

# Run inference
python src/entrypoint/inference.py

# Run tests
pytest tests/ -v --cov=src --cov-report=html

# Format code
black src/ tests/
isort src/ tests/

# Run linters
flake8 src/ tests/
pylint src/ tests/

# Start MLflow
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

## Docker on Windows

Make sure Docker Desktop is installed and running:

```powershell
# Build image
docker build -t ml-project:latest .

# Run container
docker run -p 8000:8000 -v C:\Users\Amit\VScodeAndPycharmProjects\TemplatebasedMLProject1\ml-project-example/models:/app/models ml-project:latest

# Docker Compose
docker-compose up -d
docker-compose down
```

## Troubleshooting

### Virtual Environment Not Activating
- Make sure Python is in your PATH
- Try using Command Prompt instead of PowerShell
- Check execution policy: Get-ExecutionPolicy

### Module Import Errors
- Make sure venv is activated
- Verify PYTHONPATH includes project root
- Check __init__.py files exist

### Docker Issues
- Ensure Docker Desktop is running
- Check WSL 2 backend is enabled
- Verify file sharing in Docker settings

## Next Steps

1. Customize the configuration files for your project
2. Add your data to data/01-raw/
3. Modify pipelines for your specific ML task
4. Start experimenting in notebooks/
5. Run your first training pipeline!

Happy coding! 🚀
