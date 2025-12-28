"""
Databricks Delta Table loader for Azure
"""
import pandas as pd
from typing import Optional, Dict, Any
from databricks import sql
from databricks.sdk import WorkspaceClient
import os


class DatabricksDataLoader:
    """Load data from Databricks Unity Catalog Delta Tables"""
    
    def __init__(self, config: dict):
        """
        Initialize Databricks connection
        
        Args:
            config: Configuration dictionary with Databricks settings
        """
        self.config = config
        self.databricks_config = config.get('databricks', {})
        
        # Connection parameters
        self.server_hostname = self.databricks_config.get('server_hostname')
        self.http_path = self.databricks_config.get('http_path')
        self.access_token = os.getenv('DATABRICKS_TOKEN') or self.databricks_config.get('access_token')
        
        self.connection = None
        
    def connect(self):
        """Establish connection to Databricks"""
        if self.connection is None:
            self.connection = sql.connect(
                server_hostname=self.server_hostname,
                http_path=self.http_path,
                access_token=self.access_token
            )
        return self.connection
    
    def load_table(
        self, 
        catalog: str, 
        schema: str, 
        table: str,
        filters: Optional[str] = None,
        columns: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Load data from Unity Catalog Delta table
        
        Args:
            catalog: Unity Catalog name
            schema: Schema name
            table: Table name
            filters: Optional SQL WHERE clause (without WHERE keyword)
            columns: Optional list of columns to select
            
        Returns:
            DataFrame with the data
        """
        # Build query
        cols = "*" if columns is None else ", ".join(columns)
        full_table_name = f"`{catalog}`.`{schema}`.`{table}`"
        
        query = f"SELECT {cols} FROM {full_table_name}"
        
        if filters:
            query += f" WHERE {filters}"
        
        # Execute query
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute(query)
            result = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            # Convert to DataFrame
            df = pd.DataFrame(result, columns=columns)
            return df
            
        finally:
            cursor.close()
    
    def load_query(self, query: str) -> pd.DataFrame:
        """
        Execute custom SQL query
        
        Args:
            query: SQL query string
            
        Returns:
            DataFrame with results
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute(query)
            result = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            df = pd.DataFrame(result, columns=columns)
            return df
            
        finally:
            cursor.close()
    
    def save_table(
        self,
        df: pd.DataFrame,
        catalog: str,
        schema: str,
        table: str,
        mode: str = "overwrite"
    ):
        """
        Save DataFrame to Unity Catalog Delta table
        
        Args:
            df: DataFrame to save
            catalog: Unity Catalog name
            schema: Schema name
            table: Table name
            mode: Write mode ('overwrite', 'append')
        """
        # For saving, we'll use Databricks SDK or spark
        # This requires spark session which is typically available in Databricks
        raise NotImplementedError(
            "Saving requires Databricks notebook environment with Spark. "
            "Use this in a Databricks notebook with: "
            "df.write.format('delta').mode(mode).saveAsTable(f'{catalog}.{schema}.{table}')"
        )
    
    def close(self):
        """Close the connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Alternative: Using Databricks SDK (for more modern approach)
class DatabricksSDKLoader:
    """Load data using Databricks SDK"""
    
    def __init__(self, config: dict):
        self.config = config
        self.workspace_client = WorkspaceClient(
            host=config['databricks']['workspace_url'],
            token=os.getenv('DATABRICKS_TOKEN')
        )
    
    def read_table(self, catalog: str, schema: str, table: str) -> pd.DataFrame:
        """
        Read table using SQL warehouse
        """
        query = f"SELECT * FROM `{catalog}`.`{schema}`.`{table}`"
        
        # Execute via SQL warehouse
        warehouse_id = self.config['databricks']['warehouse_id']
        
        # This is a simplified version
        # In practice, you'd use the workspace_client.statement_execution API
        raise NotImplementedError("Use DatabricksDataLoader for now")


# Helper function for easy usage
def load_from_databricks(
    config: dict,
    catalog: str,
    schema: str,
    table: str,
    **kwargs
) -> pd.DataFrame:
    """
    Convenience function to load data from Databricks
    
    Example:
        df = load_from_databricks(
            config,
            catalog='prod_data',
            schema='ml_features',
            table='customer_features',
            filters="created_date >= '2024-01-01'"
        )
    """
    with DatabricksDataLoader(config) as loader:
        return loader.load_table(catalog, schema, table, **kwargs)