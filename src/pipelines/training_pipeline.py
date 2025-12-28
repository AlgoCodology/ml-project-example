"""
Model training pipeline
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from datetime import datetime
from typing import Dict, Tuple, Any

from src.utils.logger import setup_logger


class TrainingPipeline:
    """Pipeline for model training and evaluation"""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger(__name__, config)
        self.model = None
        
    def initialize_model(self):
        """Initialize model based on config"""
        model_type = self.config['model']['type']
        hyperparams = self.config['hyperparameters']
        random_state = self.config['model']['random_state']
        
        self.logger.info(f"Initializing {model_type} model")
        
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=hyperparams.get('n_estimators', 100),
                max_depth=hyperparams.get('max_depth', 10),
                random_state=random_state
            )
        elif model_type == "random_forest_regressor":
            self.model = RandomForestRegressor(
                n_estimators=hyperparams.get('n_estimators', 100),
                max_depth=hyperparams.get('max_depth', 10),
                random_state=random_state
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return self.model
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train the model"""
        self.logger.info("Starting model training...")
        
        if self.model is None:
            self.initialize_model()
        
        self.model.fit(X_train, y_train)
        self.logger.info("Model training completed")
        
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance"""
        self.logger.info("Evaluating model...")
        
        predictions = self.model.predict(X_test)
        
        # Determine if classification or regression
        if hasattr(self.model, 'predict_proba'):
            # Classification metrics
            metrics = {
                'accuracy': accuracy_score(y_test, predictions),
                'precision': precision_score(y_test, predictions, average='weighted'),
                'recall': recall_score(y_test, predictions, average='weighted'),
                'f1_score': f1_score(y_test, predictions, average='weighted')
            }
        else:
            # Regression metrics
            metrics = {
                'mse': mean_squared_error(y_test, predictions),
                'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                'mae': mean_absolute_error(y_test, predictions),
                'r2': r2_score(y_test, predictions)
            }
        
        self.logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def save_model(self, metrics: Dict[str, float]):
        """Save trained model and metadata"""
        models_path = Path(self.config['paths']['models'])
        models_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"model_{timestamp}.pkl"
        
        # Save model
        model_path = models_path / model_filename
        joblib.dump(self.model, model_path)
        
        # Save latest model link
        latest_path = models_path / "model_latest.pkl"
        joblib.dump(self.model, latest_path)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'model_type': self.config['model']['type'],
            'metrics': metrics,
            'hyperparameters': self.config['hyperparameters'],
            'version': self.config['project']['version']
        }
        
        import json
        metadata_path = models_path / f"metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        self.logger.info(f"Model saved to {model_path}")
        
    def run(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
            y_train: pd.Series, y_test: pd.Series) -> Tuple[Any, Dict[str, float]]:
        """Execute training pipeline"""
        
        # Initialize and train model
        self.initialize_model()
        self.train(X_train, y_train)
        
        # Evaluate model
        metrics = self.evaluate(X_test, y_test)
        
        # Save model
        self.save_model(metrics)
        
        self.logger.info("Training pipeline completed successfully")
        return self.model, metrics