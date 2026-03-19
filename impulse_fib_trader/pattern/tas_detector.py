import pandas as pd
import numpy as np
from typing import List, Dict

class ImpulseRejectionDetector:
    def __init__(self, config: Dict):
        self.config = config

    def detect_patterns(self, df: pd.DataFrame) -> List[Dict]:
        patterns = []
        if len(df) < 200: return patterns

        ema_200 = df['close'].ewm(span=200, adjust=False).mean()
        
        for i in range(50, len(df) - 1):
            candle = df.iloc[i]
            
            # 1. ТРЕНД: Цена выше EMA 200
            if candle['close'] < ema_200.iloc[i]: continue
            
            # 2. ОТКАТ: Минимум 3 из 5 последних свечей были медвежьими (локальная коррекция)
            prev_5 = df.iloc[i-5:i]
            bearish_count = len(prev_5[prev_5['close'] < prev_5['open']])
            if bearish_count < 3: continue
            
            # 3. ХВОСТ (Rejection): Нижняя тень >= 40%, закрытие в верхней половине
            candle_range = candle['high'] - candle['low']
            if candle_range == 0: continue
            
            lower_wick = min(candle['open'], candle['close']) - candle['low']
            wick_ratio = lower_wick / candle_range
            
            body_top = max(candle['open'], candle['close'])
            body_pos = (body_top - candle['low']) / candle_range
            
            if wick_ratio >= 0.40 and body_pos >= 0.50:
                # 4. ПОДТВЕРЖДЕНИЕ: Свеча должна закрыться выше средней цены за последние 3 часа
                if candle['close'] > df.iloc[i-3:i]['close'].mean():
                    patterns.append({
                        'symbol': 'UNKNOWN',
                        'type': 'Impulse_Rejection',
                        'side': 'bullish',
                        'entry_idx': i,
                        'entry_price': candle['close'],
                        'sl': candle['low'] * 0.998, # Небольшой запас под хвост
                        'timestamp': candle.name
                    })
                    
        return patterns
