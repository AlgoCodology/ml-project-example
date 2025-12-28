"""
Feature engineering pipeline - Databricks version
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from typing import Tuple

from src.utils.logger import setup_logger
from src.utils.databricks_loader import DatabricksDataLoader


class FeatureEngineeringPipeline:
    """Pipeline for feature engineering with Databricks data"""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger(__name__, config)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.databricks_loader = DatabricksDataLoader(config)
        
    def load_data(self) -> pd.DataFrame:
        """Load data from Databricks Unity Catalog"""
        data_config = self.config['data']
        catalog = data_config['catalog']
        schema = data_config['schema']
        table = data_config['raw_table']
        
        self.logger.info(f"Loading data from Databricks: {catalog}.{schema}.{table}")
        
        # Load from Delta table
        df = self.databricks_loader.load_table(
            catalog=catalog,
            schema=schema,
            table=table
        )
        
        self.logger.info(f"Loaded {len(df)} rows from Databricks")
        return df
    
    def load_data_with_filter(self, date_filter: str = None) -> pd.DataFrame:
        """
        Load data with optional date filtering
        
        Args:
            date_filter: SQL date filter, e.g., "date >= '2024-01-01'"
        """
        data_config = self.config['data']
        catalog = data_config['catalog']
        schema = data_config['schema']
        table = data_config['raw_table']
        
        self.logger.info(f"Loading filtered data from Databricks")
        
        df = self.databricks_loader.load_table(
            catalog=catalog,
            schema=schema,
            table=table,
            filters=date_filter
        )
        
        self.logger.info(f"Loaded {len(df)} rows")
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features"""
        self.logger.info("Creating engineered features")
        
        # Your feature engineering logic here
        # Example: interaction features
        if 'feature_1' in df.columns and 'feature_2' in df.columns:
            df['feature_1_x_feature_2'] = df['feature_1'] * df['feature_2']
        
        # Polynomial features
        if 'feature_1' in df.columns:
            df['feature_1_squared'] = df['feature_1'] ** 2
        
        # Log transforms
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
            if col == 'target':
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
    
    def save_features_to_databricks(self, df: pd.DataFrame, table_name: str):
        """
        Save features back to Databricks (requires Databricks notebook environment)
        
        In Databricks notebook, use:
        spark_df = spark.createDataFrame(df)
        spark_df.write.format("delta").mode("overwrite").saveAsTable(f"{catalog}.{schema}.{table_name}")
        """
        self.logger.info(f"To save to Databricks, run in Databricks notebook:")
        catalog = self.config['data']['catalog']
        schema = self.config['data']['schema']
        
        code = f"""
        # In Databricks notebook:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        
        spark_df = spark.createDataFrame(df)
        spark_df.write.format("delta").mode("overwrite") \\
            .saveAsTable("{catalog}.{schema}.{table_name}")
        """
        self.logger.info(code)
    
    def save_artifacts_to_azure(self):
        """Save preprocessing artifacts to Azure Blob Storage"""
        # This would use Azure SDK
        artifacts_path = Path(self.config['paths']['artifacts'])
        
        self.logger.info("Saving artifacts locally for now")
        artifacts_path = Path("artifacts")  # Save locally
        artifacts_path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.scaler, artifacts_path / "scaler.pkl")
        joblib.dump(self.label_encoders, artifacts_path / "label_encoders.pkl")
        
        self.logger.info(f"Saved preprocessing artifacts")
        
        # TODO: Upload to Azure Blob Storage using azure-storage-blob
        # from azure.storage.blob import BlobServiceClient
        # Upload logic here
    
    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Execute feature engineering pipeline"""
        
        # Load data from Databricks
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
        
        # Save artifacts
        self.save_artifacts_to_azure()
        
        # Close Databricks connection
        self.databricks_loader.close()
        
        self.logger.info("Feature engineering pipeline completed")
        return X_train, X_test, y_train, y_test