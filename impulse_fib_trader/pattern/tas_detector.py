import pandas as pd
import numpy as np
from typing import List, Dict, Optional

class ImpulseRejectionDetector:
    def __init__(self, config: Dict):
        self.config = config

    def detect_patterns(self, df: pd.DataFrame, strict: bool = True) -> List[Dict]:
        patterns = []
        if len(df) < 200: return patterns

        ema_200 = df['close'].ewm(span=200, adjust=False).mean()
        
        # Для обзора (strict=False) делаем хвост чуть меньше
        min_wick = 0.40 if strict else 0.30
        
        for i in range(50, len(df) - 1):
            candle = df.iloc[i]
            if candle['close'] < ema_200.iloc[i]: continue
            
            prev_5 = df.iloc[i-5:i]
            bearish_count = len(prev_5[prev_5['close'] < prev_5['open']])
            if bearish_count < 3: continue
            
            candle_range = candle['high'] - candle['low']
            if candle_range == 0: continue
            
            lower_wick = min(candle['open'], candle['close']) - candle['low']
            wick_ratio = lower_wick / candle_range
            body_pos = (max(candle['open'], candle['close']) - candle['low']) / candle_range
            
            if wick_ratio >= min_wick and body_pos >= 0.50:
                if candle['close'] > df.iloc[i-3:i]['close'].mean() or not strict:
                    patterns.append({
                        'symbol': 'UNKNOWN',
                        'type': 'Impulse_Rejection',
                        'side': 'bullish',
                        'entry_idx': i,
                        'entry_price': candle['close'],
                        'sl': candle['low'] * 0.998,
                        'timestamp': candle.name,
                        'wick_ratio': wick_ratio
                    })
        return patterns

    def detect_potential(self, df: pd.DataFrame) -> Optional[Dict]:
        """Ищет монеты, которые сейчас в откате к EMA 200."""
        if len(df) < 200: return None
        
        ema_200 = df['close'].iloc[-1]
        last_price = df['close'].iloc[-1]
        
        # 1. Тренд: цена выше EMA 200, но близко к ней (в пределах 3%)
        if last_price > ema_200 and last_price < ema_200 * 1.03:
            # 2. Откат: последние 2-3 свечи красные
            last_3 = df.iloc[-3:]
            bearish_count = len(last_3[last_3['close'] < last_3['open']])
            
            if bearish_count >= 2:
                return {
                    'current_price': last_price,
                    'ema_200': ema_200,
                    'dist_pct': (last_price / ema_200 - 1) * 100
                }
        return None
