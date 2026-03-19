import pandas as pd
import numpy as np
from typing import List, Dict

class Labeler:
    def __init__(self, config: Dict):
        self.config = config

    def create_labels(self, patterns: List[Dict], df: pd.DataFrame) -> pd.Series:
        labels = []
        if not patterns: return pd.Series([])

        for p in patterns:
            entry_idx = p['entry_idx']
            entry_price = p['entry_price']
            sl = p['sl']
            
            risk = entry_price - sl
            if risk <= 0:
                labels.append(0)
                continue
            
            tp = entry_price + (risk * 2.0)
            label = 0
            end_search = min(entry_idx + 48, len(df) - 1)
            
            for i in range(entry_idx + 1, end_search + 1):
                if df.iloc[i]['low'] <= sl: break
                if df.iloc[i]['high'] >= tp:
                    label = 1
                    break
            labels.append(label)
        return pd.Series(labels)
