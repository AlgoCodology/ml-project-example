"""
Unit tests for training pipeline
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipelines.training_pipeline import TrainingPipeline
from src.utils.config import load_config


@pytest.fixture
def sample_config():
    """Create sample configuration for testing"""
    return {
        'project': {
            'name': 'test-project',
            'version': '0.1.0'
        },
        'model': {
            'type': 'random_forest',
            'random_state': 42
        },
        'hyperparameters': {
            'n_estimators': 10,
            'max_depth': 5
        },
        'paths': {
            'models': 'tests/tmp/models',
            'logs': 'tests/tmp/logs'
        },
        'logging': {
            'level': 'INFO'
        }
    }


@pytest.fixture
def sample_data():
    """Create sample training data"""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    X_test = pd.DataFrame(
        np.random.randn(n_samples // 4, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    y_train = pd.Series(np.random.randint(0, 2, n_samples))
    y_test = pd.Series(np.random.randint(0, 2, n_samples // 4))
    
    return X_train, X_test, y_train, y_test


def test_initialize_model(sample_config):
    """Test model initialization"""
    pipeline = TrainingPipeline(sample_config)
    model = pipeline.initialize_model()
    
    assert model is not None
    assert pipeline.model is not None


def test_training_pipeline(sample_config, sample_data):
    """Test full training pipeline"""
    X_train, X_test, y_train, y_test = sample_data
    
    pipeline = TrainingPipeline(sample_config)
    pipeline.initialize_model()
    pipeline.train(X_train, y_train)
    
    assert pipeline.model is not None


def test_model_evaluation(sample_config, sample_data):
    """Test model evaluation"""
    X_train, X_test, y_train, y_test = sample_data
    
    pipeline = TrainingPipeline(sample_config)
    pipeline.initialize_model()
    pipeline.train(X_train, y_train)
    
    metrics = pipeline.evaluate(X_test, y_test)
    
    assert 'accuracy' in metrics
    assert 0 <= metrics['accuracy'] <= 1


def test_model_prediction(sample_config, sample_data):
    """Test model predictions"""
    X_train, X_test, y_train, y_test = sample_data
    
    pipeline = TrainingPipeline(sample_config)
    pipeline.initialize_model()
    pipeline.train(X_train, y_train)
    
    predictions = pipeline.model.predict(X_test)
    
    assert len(predictions) == len(X_test)
    assert all(pred in [0, 1] for pred in predictions)