# ML Project Template

A production-ready machine learning project template with best practices for data science and ML engineering.

## 🎯 Project Structure

```
ml-project-example/
├── config/                    # Configuration files
│   ├── local.yaml            # Local development config
│   ├── prod.yaml             # Production config
│   └── databricks.yaml       # Databricks + Azure config
├── data/                      # Data directories
│   ├── 01-raw/               # Raw source data
│   ├── 02-preprocessed/      # Cleaned datasets
│   ├── 03-features/          # Engineered features
│   └── 04-predictions/       # Model predictions
├── src/
│   ├── entrypoint/           # Application entry points
│   │   ├── inference.py      # Inference service
│   │   └── train.py          # Training orchestration
│   ├── pipelines/            # ML pipelines
│   │   ├── feature_eng_pipeline.py
│   │   └── training_pipeline.py
│   └── utils/                # Utility functions
│       ├── config.py         # Config management
│       ├── logger.py         # Logging setup
│       └── databricks_loader.py  # Databricks connector
├── models/                    # Saved models
├── notebooks/                 # Jupyter notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experimentation.ipynb
├── tests/                     # Unit tests
├── docker-compose.yml        # Docker services
├── Dockerfile                # Production container
├── commands.bat              # Windows command helper
├── Makefile                  # Linux/Mac commands
├── requirements-dev.txt      # Dev dependencies
├── requirements-prod.txt     # Production dependencies
└── requirements-databricks.txt  # Databricks dependencies
```

## 🚀 Quick Start

### 1. Setup Environment

**Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# If you get execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

**Linux/Mac:**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements-dev.txt
```

### 2. Train Model

**Windows:**
```powershell
# Using commands.bat
commands.bat train

# Or directly
python src/entrypoint/train.py --config config/local.yaml
```

**Linux/Mac:**
```bash
# Using Makefile
make train

# Or directly
python src/entrypoint/train.py --config config/local.yaml
```

### 3. Run Inference

**Windows:**
```powershell
commands.bat inference
```

**Linux/Mac:**
```bash
make inference
```

### 4. Run Tests

**Windows:**
```powershell
commands.bat test
```

**Linux/Mac:**
```bash
make test
```

## 🐳 Docker Deployment

### Prerequisites
- Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Ensure Docker Desktop is running

### Build and Run

**Windows (PowerShell):**
```powershell
# Build Docker image
docker build -t ml-project:latest .

# Run container
docker run -p 8000:8000 `
  -v ${PWD}/models:/app/models `
  -v ${PWD}/data:/app/data `
  ml-project:latest

# Check if container is running
docker ps
```

**Linux/Mac:**
```bash
# Build Docker image
docker build -t ml-project:latest .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  ml-project:latest
```

### Docker Compose (Full Stack with MLflow)

**Windows & Linux/Mac:**
```powershell
# Start all services (ML app + MLflow + PostgreSQL)
docker-compose up -d

# Check running services
docker-compose ps

# View logs
docker-compose logs -f ml-app

# Access services
# - MLflow UI: http://localhost:5000
# - ML API: http://localhost:8000

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## 📊 MLflow Tracking

**Windows:**
```powershell
commands.bat mlflow
# or
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

**Linux/Mac:**
```bash
make mlflow
# or
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Access the UI at `http://localhost:5000`

## 🧪 Testing

**Windows:**
```powershell
# Run all tests with coverage
commands.bat test

# Run specific test file
pytest tests/test_training.py -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=html
```

**Linux/Mac:**
```bash
make test

# Or manually
pytest tests/ -v --cov=src --cov-report=html
```

## 🎨 Code Quality

**Windows:**
```powershell
# Format code
commands.bat format

# Run linters
commands.bat lint

# Clean temporary files
commands.bat clean
```

**Linux/Mac:**
```bash
make format
make lint
make clean
```

## 📝 Configuration

Configurations are stored in `config/` directory:

- **`local.yaml`**: Local development settings
- **`prod.yaml`**: Production settings with cloud storage
- **`databricks.yaml`**: Databricks + Azure configuration

### Key Configuration Sections:
- **data**: Data paths and storage locations
- **model**: Model type and parameters
- **training**: Training hyperparameters
- **logging**: Logging configuration
- **mlflow**: Experiment tracking settings
- **databricks**: Unity Catalog connection (if using Databricks)

### Example: Switching Configurations
```powershell
# Train with production config
python src/entrypoint/train.py --config config/prod.yaml

# Train with Databricks config
python src/entrypoint/train.py --config config/databricks.yaml
```

## 🔷 Using with Databricks + Azure

### Setup Databricks Connection

1. **Create `.env` file:**
```bash
DATABRICKS_TOKEN=your_personal_access_token
AZURE_SUBSCRIPTION_ID=your_subscription_id
AZURE_TENANT_ID=your_tenant_id
```

2. **Install Databricks dependencies:**
```powershell
pip install -r requirements-databricks.txt
```

3. **Update `config/databricks.yaml`** with your:
   - Workspace URL
   - SQL Warehouse path
   - Unity Catalog details
   - Azure storage account info

4. **Train with Databricks data:**
```powershell
python src/entrypoint/train.py --config config/databricks.yaml
```

## 📓 Jupyter Notebooks

The `notebooks/` folder contains three key notebooks:

1. **`01_exploratory_data_analysis.ipynb`**
   - Data loading and inspection
   - Missing value analysis
   - Feature distributions
   - Correlation analysis
   - Outlier detection

2. **`02_feature_engineering.ipynb`**
   - Baseline model
   - Feature creation
   - Feature importance
   - Feature selection
   - Performance comparison

3. **`03_model_experimentation.ipynb`**
   - Multiple algorithm comparison
   - Hyperparameter tuning
   - Model evaluation
   - ROC curves and metrics

**To use:**
```powershell
# Start Jupyter
jupyter notebook notebooks/
```

## 🏗️ Project Components

### Pipelines

1. **Feature Engineering Pipeline** (`feature_eng_pipeline.py`)
   - Data loading (local or Databricks)
   - Feature creation and transformations
   - Encoding and scaling
   - Train/test splitting

2. **Training Pipeline** (`training_pipeline.py`)
   - Model initialization
   - Training execution
   - Model evaluation
   - Model persistence

### Entry Points

1. **Training** (`train.py`)
   - Orchestrates the full training workflow
   - Logs experiments to MLflow
   - Saves models and metrics

2. **Inference** (`inference.py`)
   - Loads trained models
   - Handles prediction requests
   - Provides API interface

## 🔧 Development Workflow

1. **Data Preparation**: Place raw data in `data/01-raw/`
2. **Exploration**: Use notebooks for EDA and experimentation
3. **Feature Engineering**: Develop features, then move to pipeline
4. **Model Development**: Experiment in notebooks, finalize in training pipeline
5. **Testing**: Write tests for all components
6. **Deployment**: Build Docker image and deploy

## 📦 Available Commands

### Windows (commands.bat)

| Command | Description |
|---------|-------------|
| `commands.bat help` | Show all available commands |
| `commands.bat install` | Install production dependencies |
| `commands.bat install-dev` | Install development dependencies |
| `commands.bat train` | Train model locally |
| `commands.bat train-prod` | Train with production config |
| `commands.bat inference` | Run inference service |
| `commands.bat test` | Run tests with coverage |
| `commands.bat format` | Format code (Black + isort) |
| `commands.bat lint` | Run linters (flake8 + pylint) |
| `commands.bat clean` | Clean temporary files |
| `commands.bat mlflow` | Start MLflow UI |

### Linux/Mac (Makefile)

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make install` | Install production dependencies |
| `make install-dev` | Install development dependencies |
| `make train` | Train model locally |
| `make train-prod` | Train with production config |
| `make inference` | Run inference service |
| `make test` | Run tests with coverage |
| `make format` | Format code |
| `make lint` | Run linters |
| `make clean` | Clean temporary files |
| `make mlflow` | Start MLflow UI |
| `make docker-build` | Build Docker image |
| `make docker-run` | Run Docker container |
| `make docker-compose` | Start all services |

## 🐞 Troubleshooting

### Windows-Specific Issues

**Virtual Environment Not Activating:**
```powershell
# Check execution policy
Get-ExecutionPolicy

# Set if needed
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Try Command Prompt instead
.\venv\Scripts\activate.bat
```

**Module Import Errors:**
```powershell
# Verify venv is activated (you should see (venv) in prompt)
# Add project to PYTHONPATH
$env:PYTHONPATH = "$PWD;$env:PYTHONPATH"
```

**Docker Issues:**
- Ensure Docker Desktop is running
- Check WSL 2 backend is enabled (Settings → General)
- Verify file sharing is enabled (Settings → Resources → File Sharing)
- Make sure you've **built the image first**: `docker build -t ml-project:latest .`

### General Issues

**Tests Failing:**
```powershell
# Make sure all dependencies are installed
pip install -r requirements-dev.txt

# Check if all source files have content
python check_imports.py
```

**MLflow Connection Issues:**
```powershell
# Check if MLflow server is running
# Start it manually:
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

**Databricks Connection Issues:**
```powershell
# Verify .env file has DATABRICKS_TOKEN
# Test connection:
python -c "from src.utils.databricks_loader import DatabricksDataLoader; print('OK')"
```

## 📚 Dependencies

### Core Libraries
- **scikit-learn**: Traditional ML algorithms
- **XGBoost/LightGBM**: Gradient boosting
- **pandas/numpy**: Data manipulation
- **MLflow**: Experiment tracking
- **FastAPI**: API framework (for serving)

### Optional
- **databricks-sql-connector**: Databricks connectivity
- **azure-storage-blob**: Azure storage
- **boto3**: AWS storage (if needed)

## 🤝 Contributing

1. Create feature branch
2. Make changes with tests
3. Run code quality checks:
   ```powershell
   commands.bat format
   commands.bat lint
   commands.bat test
   ```
4. Submit pull request


ML Project Tryout team

---

## 🚀 Quick Reference Card

### First Time Setup (Windows)
```powershell
# 1. Clone/create project
cd your-project-folder

# 2. Create and activate venv
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements-dev.txt

# 4. Verify setup
python -c "import src; print('✅ Setup OK!')"
pytest tests/ -v
```

### Daily Development (Windows)
```powershell
# Activate venv
.\venv\Scripts\Activate.ps1

# Run your workflow
commands.bat train      # Train model
commands.bat test       # Run tests
commands.bat mlflow     # View experiments

# Before committing
commands.bat format     # Format code
commands.bat lint       # Check code quality
```

### Docker Workflow (Windows)
```powershell
# Build once
docker build -t ml-project:latest .

# Run when needed
docker run -p 8000:8000 -v ${PWD}/models:/app/models ml-project:latest

# Or use compose for full stack
docker-compose up -d
docker-compose logs -f
docker-compose down

# Other useful docker commands
# List all images
docker images

# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Stop a running container
docker stop <container-id>

# Remove a container
docker rm <container-id>

# Remove an image
docker rmi ml-project:latest

# View container logs
docker logs <container-id>

# Rebuild without cache (if needed)
docker build --no-cache -t ml-project:latest .
```

---

**Note**: This template is cross-platform. Windows users can use `commands.bat`, while Linux/Mac users can use `make`. All core functionality works the same across platforms.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📄 License

MIT License - see the [LICENSE](LICENSE) file for details (Major Attribution: assisted ofcourse by the overlord - Claude Sonnet 4.5 :-) )
