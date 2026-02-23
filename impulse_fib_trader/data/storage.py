import pandas as pd
import pathlib

class DataStorage:
    @staticmethod
    def save_to_parquet(df: pd.DataFrame, file_path: str):
        """Saves DataFrame to Parquet."""
        df.to_parquet(file_path, index=False)

    @staticmethod
    def load_from_parquet(file_path: str) -> pd.DataFrame:
        """Loads DataFrame from Parquet."""
        return pd.read_parquet(file_path)
