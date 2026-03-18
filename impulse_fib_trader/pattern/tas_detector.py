import pandas as pd
import numpy as np
from typing import List, Dict, Optional

class TASDetector:
    def __init__(self, config: Dict):
        self.config = config

    def detect_patterns(self, df: pd.DataFrame) -> List[Dict]:
        patterns = []
        if len(df) < 30:
            return patterns

        for i in range(10, len(df) - 5):
            # 1. Phase 2: Liquidity Tail (Хвост)
            candle = df.iloc[i]
            candle_range = candle['high'] - candle['low']
            if candle_range == 0: continue
            
            lower_wick = min(candle['open'], candle['close']) - candle['low']
            wick_ratio = lower_wick / candle_range
            
            body_top = max(candle['open'], candle['close'])
            body_pos = (body_top - candle['low']) / candle_range

            # Хвост должен быть заметным (от 35%) и закрытие в верхней половине
            if wick_ratio >= 0.35 and body_pos >= 0.50:
                
                # Phase 3: Поиск локального максимума после хвоста (Breakout Level)
                # Ищем в следующих 5 свечах
                search_end_v = min(i + 6, len(df))
                v_zone = df.iloc[i+1 : search_end_v]
                if v_zone.empty: continue
                
                breakout_level = v_zone['high'].max()
                b_idx = v_zone['high'].idxmax()
                
                # Phase 4: Shelf Formation (Полка)
                # Полка идет ПОСЛЕ максимума отскока
                shelf_start = df.index.get_loc(b_idx) + 1
                if shelf_start >= len(df) - 1: continue
                
                # Ищем пробой уровня в последующих 15 свечах
                shelf_limit = min(shelf_start + 15, len(df))
                
                for j in range(shelf_start + 2, shelf_limit):
                    current_candle = df.iloc[j]
                    shelf_zone = df.iloc[shelf_start : j]
                    
                    if shelf_zone.empty: continue
                    
                    # Условия полки:
                    # - Не падать ниже хвоста
                    # - Находиться ниже уровня пробоя
                    if shelf_zone['low'].min() < candle['low'] * 0.999: break
                    if shelf_zone['high'].max() > breakout_level * 1.001: break
                    
                    # Entry: Пробой уровня закрытием
                    if current_candle['close'] > breakout_level:
                        patterns.append({
                            'symbol': 'ETH/USDT',
                            'type': 'TAS_v1',
                            'tail_idx': i,
                            'tail_low': candle['low'],
                            'breakout_level': breakout_level,
                            'shelf_low': shelf_zone['low'].min(),
                            'entry_idx': j,
                            'entry_price': current_candle['close'],
                            'timestamp': current_candle.name if hasattr(current_candle.name, 'isoformat') else str(current_candle.name)
                        })
                        break # Нашли вход для этого хвоста
                        
        return patterns
