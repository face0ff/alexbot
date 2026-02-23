import pandas as pd
import numpy as np
from typing import List, Dict

class FeatureEngineer:
    def __init__(self):
        pass

    def extract_features(self, patterns: List[Dict], df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts features for each detected pattern.
        """
        features_list = []
        
        for p in patterns:
            imp = p['impulse']
            pb = p['pullback']
            struct = p['structure']
            
            # 1. Impulse features
            imp_df = df.iloc[imp['start_idx'] : imp['end_idx'] + 1]
            imp_duration = len(imp_df)
            imp_range_abs = imp['range']
            avg_atr = imp_df['atr'].mean()
            imp_range_atr = imp_range_abs / avg_atr if avg_atr > 0 else 0
            
            # 2. Pullback features
            pb_df = df.iloc[pb['start_idx'] : pb['end_idx'] + 1]
            pb_duration = len(pb_df)
            pb_depth = pb['depth']
            
            # 3. Volatility contraction (std of bodies in pb vs imp)
            imp_bodies = np.abs(imp_df['close'] - imp_df['open'])
            pb_bodies = np.abs(pb_df['close'] - pb_df['open'])
            vol_contraction = pb_bodies.std() / imp_bodies.std() if imp_bodies.std() > 0 else 1.0
            
            # 4. Extremum wick ratio (at the end of pullback)
            pb_end_candle = pb_df.iloc[-1]
            wick_total = (pb_end_candle['high'] - pb_end_candle['low']) - np.abs(pb_end_candle['close'] - pb_end_candle['open'])
            wick_ratio = wick_total / (pb_end_candle['high'] - pb_end_candle['low']) if (pb_end_candle['high'] - pb_end_candle['low']) > 0 else 0
            
            # 5. Structure break strength (momentum of the break candle)
            break_candle = df.iloc[struct['entry_idx']]
            break_strength = np.abs(break_candle['close'] - break_candle['open']) / (break_candle['high'] - break_candle['low']) if (break_candle['high'] - break_candle['low']) > 0 else 0
            
            # 6. Volume profile (volume increase on impulse vs decrease on pullback)
            imp_vol_avg = imp_df['volume'].mean()
            pb_vol_avg = pb_df['volume'].mean()
            vol_ratio = imp_vol_avg / pb_vol_avg if pb_vol_avg > 0 else 1.0
            
            features = {
                'impulse_range_atr': imp_range_atr,
                'impulse_duration': imp_duration,
                'pullback_depth': pb_depth,
                'pullback_duration': pb_duration,
                'volatility_contraction': vol_contraction,
                'extremum_wick_ratio': wick_ratio,
                'structure_break_strength': break_strength,
                'volume_ratio': vol_ratio,
                'is_bullish': 1 if imp['type'] == 'bullish' else 0
            }
            features_list.append(features)
            
        return pd.DataFrame(features_list)
