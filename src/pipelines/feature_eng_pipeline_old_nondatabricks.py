"""
Feature engineering pipeline
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from typing import Tuple

from src.utils.logger import setup_logger


class FeatureEngineeringPipeline:
    """Pipeline for feature engineering and data preprocessing"""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger(__name__, config)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load preprocessed data"""
        data_path = Path(self.config['data']['preprocessed_data_path'])
        self.logger.info(f"Loading data from {data_path}")
        
        # Assuming CSV for this example
        df = pd.read_csv(data_path / "data.csv")
        self.logger.info(f"Loaded {len(df)} rows")
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features"""
        self.logger.info("Creating engineered features")
        
        # Example feature engineering
        # Add your domain-specific feature engineering here
        
        # Interaction features
        if 'feature_1' in df.columns and 'feature_2' in df.columns:
            df['feature_1_x_feature_2'] = df['feature_1'] * df['feature_2']
        
        # Polynomial features
        if 'feature_1' in df.columns:
            df['feature_1_squared'] = df['feature_1'] ** 2
        
        # Log transforms for skewed features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if (df[col] > 0).all():
                df[f'{col}_log'] = np.log1p(df[col])
        
        self.logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
        return df
    
    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables"""
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col == 'target':  # Skip target column
                continue
                
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                if col in self.label_encoders:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features"""
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def save_artifacts(self):
        """Save preprocessing artifacts"""
        artifacts_path = Path(self.config['paths']['artifacts'])
        artifacts_path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.scaler, artifacts_path / "scaler.pkl")
        joblib.dump(self.label_encoders, artifacts_path / "label_encoders.pkl")
        
        self.logger.info(f"Saved preprocessing artifacts to {artifacts_path}")
    
    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Execute feature engineering pipeline"""
        
        # Load data
        df = self.load_data()
        
        # Separate features and target
        target_col = 'target'  # Adjust based on your data
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        # Create features
        X = self.create_features(X)
        
        # Encode categorical variables
        X = self.encode_categorical(X, fit=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['training']['test_size'],
            random_state=self.config['model']['random_state']
        )
        
        # Scale features
        X_train = self.scale_features(X_train, fit=True)
        X_test = self.scale_features(X_test, fit=False)
        
        # Save feature engineered data
        features_path = Path(self.config['data']['features_path'])
        features_path.mkdir(parents=True, exist_ok=True)
        
        X_train.to_csv(features_path / "X_train.csv", index=False)
        X_test.to_csv(features_path / "X_test.csv", index=False)
        y_train.to_csv(features_path / "y_train.csv", index=False)
        y_test.to_csv(features_path / "y_test.csv", index=False)
        
        # Save preprocessing artifacts
        self.save_artifacts()
        
        self.logger.info("Feature engineering pipeline completed")
        return X_train, X_test, y_train, y_test