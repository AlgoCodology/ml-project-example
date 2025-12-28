"""
Training entrypoint for model training pipeline
"""
import sys
from pathlib import Path
import mlflow
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipelines.feature_eng_pipeline import FeatureEngineeringPipeline
from src.pipelines.training_pipeline import TrainingPipeline
from src.utils.config import load_config
from src.utils.logger import setup_logger


def main(config_path: str = "config/local.yaml"):
    """Main training orchestration"""
    
    # Load configuration
    config = load_config(config_path)
    logger = setup_logger(__name__, config)
    
    logger.info("=" * 50)
    logger.info(f"Starting training pipeline - {config['project']['name']}")
    logger.info(f"Environment: {config['project']['environment']}")
    logger.info("=" * 50)
    
    # Setup MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log configuration
        mlflow.log_params({
            "environment": config['project']['environment'],
            "model_type": config['model']['type'],
            "test_size": config['training']['test_size']
        })
        
        try:
            # Feature Engineering
            logger.info("Starting feature engineering...")
            fe_pipeline = FeatureEngineeringPipeline(config)
            X_train, X_test, y_train, y_test = fe_pipeline.run()
            logger.info("Feature engineering completed")
            
            # Model Training
            logger.info("Starting model training...")
            training_pipeline = TrainingPipeline(config)
            model, metrics = training_pipeline.run(X_train, X_test, y_train, y_test)
            logger.info(f"Model training completed. Metrics: {metrics}")
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            logger.info("Training pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ML model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/local.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    main(args.config)