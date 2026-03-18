import pandas as pd
import numpy as np
from typing import List, Dict

class Labeler:
    def __init__(self, config: Dict):
        self.config = config

    def create_labels(self, patterns: List[Dict], df: pd.DataFrame) -> pd.Series:
        labels = []
        
        # Разные параметры для разных типов паттернов
        if not patterns:
            return pd.Series([])

        is_tas = patterns[0].get('type') == 'TAS_v1'
        
        for p in patterns:
            entry_idx = p['entry_idx']
            entry_price = p['entry_price']
            
            if is_tas:
                # Для TAS стоп за хвост
                sl = p['tail_low']
                risk = entry_price - sl
                tp = entry_price + (risk * 2.0) # RR 2.0 для TAS по спецификации
            else:
                # Старая логика Impulse_Fib
                pb = p['pullback']
                sl = pb['low']
                risk = entry_price - sl
                tp = entry_price + (risk * 1.5)

            if risk <= 0:
                labels.append(0)
                continue

            # Симуляция (макс 48 часов для H1)
            label = 0
            end_search = min(entry_idx + 48, len(df) - 1)
            
            for i in range(entry_idx + 1, end_search + 1):
                high = df.iloc[i]['high']
                low = df.iloc[i]['low']
                
                if low <= sl:
                    label = 0
                    break
                if high >= tp:
                    label = 1
                    break
            
            labels.append(label)
            
        return pd.Series(labels)
