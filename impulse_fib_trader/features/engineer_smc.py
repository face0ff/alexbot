import pandas as pd
import numpy as np
from typing import List, Dict

class SMCFeatureEngineer:
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Извлекает признаки для каждого момента, где bos_signal != 0.
        """
        # Находим индексы всех BOS
        bos_indices = df[df['bos_signal'] != 0].index
        features_list = []

        for idx in bos_indices:
            pos = df.index.get_loc(idx)
            candle = df.iloc[pos]
            prev_candle = df.iloc[pos-1]
            
            # 1. Характеристики свечи пробоя (BOS Candle)
            body_size = abs(candle['close'] - candle['open'])
            full_range = candle['high'] - candle['low']
            body_ratio = body_size / full_range if full_range > 0 else 0
            
            # 2. Объем (Volume Delta)
            avg_vol = df['volume'].iloc[max(0, pos-20):pos].mean()
            vol_ratio = candle['volume'] / avg_vol if avg_vol > 0 else 1.0
            
            # 3. Волатильность (ATR)
            atr = df['atr'].iloc[pos] if 'atr' in df.columns else (full_range)
            atr_ratio = atr / candle['close']
            
            # 4. Расстояние до EMA 200 (Тренд)
            ema_200 = df['ema_200'].iloc[pos] if 'ema_200' in df.columns else candle['close']
            dist_ema = (candle['close'] / ema_200) - 1
            
            # 5. RSI (Перекупленность/Перепроданность)
            rsi = df['rsi'].iloc[pos] if 'rsi' in df.columns else 50
            
            features_list.append({
                'timestamp': idx,
                'bos_type': df['bos_signal'].iloc[pos],
                'body_ratio': body_ratio,
                'vol_ratio': vol_ratio,
                'atr_ratio': atr_ratio,
                'dist_ema': dist_ema,
                'rsi': rsi,
                'prev_candle_size': abs(prev_candle['close'] - prev_candle['open']) / atr if atr > 0 else 0
            })
            
        return pd.DataFrame(features_list).set_index('timestamp')
