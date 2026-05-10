import pandas as pd
import numpy as np
from typing import List, Dict, Optional

class ImpulseFib75Detector:
    def __init__(self, config: Dict = None):
        self.config = config or {}

    def detect_patterns(self, df: pd.DataFrame) -> List[Dict]:
        patterns = []
        if len(df) < 50: return patterns

        # 1. Расчет ATR для волатильности
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(14).mean()

        for i in range(20, len(df) - 1):
            # Ищем импульс вверх (минимум 2 свечи)
            # Упрощенно: берем диапазон последних 5 свечей как потенциальный импульс
            impulse_start_idx = i - 5
            impulse_end_idx = i - 2 # Оставляем место для отката
            
            low_p0 = df.iloc[impulse_start_idx:impulse_end_idx]['low'].min()
            high_p1 = df.iloc[impulse_start_idx:impulse_end_idx]['high'].max()
            range_height = high_p1 - low_p0
            
            if range_height < atr.iloc[i] * 1.0: continue # min_atr_multiplier: 1.0
            
            # Проверяем откат (текущая или предыдущая свеча)
            current_low = df.iloc[i]['low']
            current_close = df.iloc[i]['close']
            
            # Расчет уровня Фибо (0 - старт, 1 - хай)
            # Retracement = (High - Current) / (High - Low)
            retracement = (high_p1 - current_low) / range_height if range_height > 0 else 0
            
            # Условие: глубокий откат (0.75 - 0.90)
            if 0.75 <= retracement <= 0.95:
                # Ложный пробой: тень ниже 0.75, но закрытие выше
                close_retracement = (high_p1 - current_close) / range_height
                
                if close_retracement < 0.90: # Закрылись не слишком низко
                    # Защита от отрицательного стопа (особенно на дешевых монетах)
                    calculated_sl = low_p0 - (atr.iloc[i] * 0.1)
                    final_sl = max(calculated_sl, current_close * 0.90) # Не более 10% стопа от цены входа в любом случае

                    patterns.append({
                        'type': 'Fib75_Reversion',
                        'entry_idx': i,
                        'entry_price': current_close,
                        'p0': low_p0,
                        'p1': high_p1,
                        'retracement': retracement,
                        'sl': final_sl,
                        'tp': high_p1,
                        'timestamp': df.index[i]
                    })
        
        return patterns
