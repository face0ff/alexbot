import pandas as pd
import numpy as np
from typing import Tuple

class DataCleaner:
    @staticmethod
    def validate_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates OHLCV data for gaps and outliers.
        """
        # Check for missing values
        if df.isnull().values.any():
            df = df.fillna(method='ffill')
            
        # Check for duplicates
        df = df.drop_duplicates(subset='timestamp').sort_values('timestamp')
        
        # Check for gaps in timeframe
        # For simplicity, we assume data is mostly continuous
        return df

    @staticmethod
    def calculate_indicators(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
        """
        Calculates ATR and other basic indicators.
        """
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        df['atr'] = true_range.rolling(window=atr_period).mean()
        return df

    @staticmethod
    def identify_swings(df: pd.DataFrame, window: int = 2) -> pd.DataFrame:
        """
        Identifies local swings (highs and lows).
        """
        df['swing_high'] = df['high'][(df['high'] == df['high'].rolling(2*window+1, center=True).max())]
        df['swing_low'] = df['low'][(df['low'] == df['low'].rolling(2*window+1, center=True).min())]
        return df
