import pandas as pd
import numpy as np
from typing import Tuple

class DataCleaner:
    @staticmethod
    def validate_data(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        df = df.drop_duplicates(subset='timestamp').sort_values('timestamp')
        if df.isnull().values.any():
            df = df.fillna(method='ffill')
        return df

    @staticmethod
    def calculate_indicators(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
        """Calculates ATR, EMA 20, 50, 200, and RSI."""
        if df.empty: return df
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(window=atr_period).mean()
        
        # EMAs
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df

    @staticmethod
    def identify_swings(df: pd.DataFrame, window: int = 2) -> pd.DataFrame:
        df['swing_high'] = df['high'][(df['high'] == df['high'].rolling(2*window+1, center=True).max())]
        df['swing_low'] = df['low'][(df['low'] == df['low'].rolling(2*window+1, center=True).min())]
        return df
