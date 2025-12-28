"""
Inference entrypoint for serving model predictions
"""
import sys
from pathlib import Path
import joblib
import pandas as pd
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.logger import setup_logger


class ModelInferenceService:
    """Service for loading model and making predictions"""
    
    def __init__(self, config_path: str = "config/prod.yaml"):
        self.config = load_config(config_path)
        self.logger = setup_logger(__name__, self.config)
        self.model = None
        self.preprocessor = None
        
    def load_model(self, model_path: str = None):
        """Load trained model and preprocessor"""
        if model_path is None:
            model_path = f"{self.config['paths']['models']}/model_latest.pkl"
        
        self.logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)
        
        preprocessor_path = model_path.replace("model_", "preprocessor_")
        self.preprocessor = joblib.load(preprocessor_path)
        self.logger.info("Model and preprocessor loaded successfully")
        
    def preprocess_input(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess input data for inference"""
        df = pd.DataFrame([data])
        
        if self.preprocessor:
            df_processed = self.preprocessor.transform(df)
        else:
            df_processed = df
            
        return df_processed
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction on input data"""
        if self.model is None:
            self.load_model()
        
        try:
            # Preprocess input
            processed_data = self.preprocess_input(data)
            
            # Make prediction
            prediction = self.model.predict(processed_data)[0]
            
            # Get probability if classifier
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(processed_data)[0]
                prob_dict = {f"class_{i}": float(p) for i, p in enumerate(probabilities)}
            else:
                prob_dict = {}
            
            result = {
                "prediction": float(prediction),
                "probabilities": prob_dict,
                "model_version": self.config['project']['version']
            }
            
            self.logger.info(f"Prediction made successfully: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise


def main():
    """Main inference function"""
    service = ModelInferenceService()
    service.load_model()
    
    # Example usage
    sample_input = {
        "feature_1": 1.5,
        "feature_2": 2.3,
        "feature_3": 0.8
    }
    
    result = service.predict(sample_input)
    print(f"Prediction result: {result}")


if __name__ == "__main__":
    main()