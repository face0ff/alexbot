import pandas as pd
import numpy as np
from typing import List, Dict

class FeatureEngineer:
    def extract_features(self, patterns: List[Dict], df: pd.DataFrame) -> pd.DataFrame:
        features = []
        if not patterns: return pd.DataFrame()

        for p in patterns:
            idx = p['entry_idx']
            candle = df.iloc[idx]
            
            f = {
                'rsi': candle['rsi'],
                'atr_ratio': candle['atr'] / candle['close'],
                'dist_ema_20': (candle['close'] / candle['ema_20']) - 1,
                'dist_ema_200': (candle['close'] / candle['ema_200']) - 1,
                'volume_ratio': candle['volume'] / df.iloc[idx-20:idx]['volume'].mean() if idx > 20 else 1.0
            }
            
            # Параметры хвоста
            c_range = candle['high'] - candle['low']
            wick = min(candle['open'], candle['close']) - candle['low']
            f['wick_ratio'] = wick / c_range if c_range > 0 else 0
            
            features.append(f)
            
        return pd.DataFrame(features)
