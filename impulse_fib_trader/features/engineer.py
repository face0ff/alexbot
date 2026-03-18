import pandas as pd
import numpy as np
from typing import List, Dict

class FeatureEngineer:
    def extract_features(self, patterns: List[Dict], df: pd.DataFrame) -> pd.DataFrame:
        features = []
        
        if not patterns:
            return pd.DataFrame()

        is_tas = patterns[0].get('type') == 'TAS_v1'

        for p in patterns:
            idx = p['entry_idx']
            candle = df.iloc[idx]
            
            # Common features
            f = {
                'rsi': candle['rsi'],
                'atr_ratio': candle['atr'] / candle['close'],
                'dist_ema_20': (candle['close'] / candle['ema_20']) - 1,
                'dist_ema_50': (candle['close'] / candle['ema_50']) - 1,
                'volume_ratio': candle['volume'] / df.iloc[idx-20:idx]['volume'].mean() if idx > 20 else 1.0
            }
            
            if is_tas:
                # TAS Specific Features
                tail_idx = p['tail_idx']
                tail_candle = df.iloc[tail_idx]
                
                # Wick ratio of the tail
                tail_range = tail_candle['high'] - tail_candle['low']
                lower_wick = min(tail_candle['open'], tail_candle['close']) - tail_candle['low']
                f['tail_wick_ratio'] = lower_wick / tail_range if tail_range > 0 else 0
                
                # Shelf characteristics
                f['shelf_duration'] = idx - tail_idx
                f['shelf_low_ratio'] = (p['shelf_low'] / p['tail_low']) - 1
                f['breakout_power'] = (p['entry_price'] / p['breakout_level']) - 1
                f['tail_volume_ratio'] = tail_candle['volume'] / df.iloc[tail_idx-10:tail_idx]['volume'].mean() if tail_idx > 10 else 1.0
            else:
                # Impulse Fib Specific Features
                imp = p['impulse']
                pb = p['pullback']
                f['impulse_pct'] = (imp['end_price'] / imp['start_price']) - 1
                f['pullback_depth'] = p['pullback']['depth']
                f['impulse_duration'] = imp['end_idx'] - imp['start_idx']
                f['pullback_duration'] = pb['end_idx'] - pb['start_idx']

            features.append(f)
            
        return pd.DataFrame(features)
