import pandas as pd
import numpy as np
from typing import List, Dict

class Labeler:
    def __init__(self, config: Dict):
        self.config = config['risk_management']

    def create_labels(self, patterns: List[Dict], df: pd.DataFrame) -> pd.Series:
        """
        Creates binary labels for patterns based on future price action.
        1 = Success (Profit target hit), 0 = Failure (Stop loss hit).
        """
        labels = []
        max_bars = self.config['max_bars_in_trade']
        atr_buffer = self.config['stop_loss']['buffer_atr']
        
        for p in patterns:
            imp = p['impulse']
            pb = p['pullback']
            struct = p['structure']
            
            entry_idx = struct['entry_idx']
            entry_price = struct['entry_price']
            
            # Calculate Stop Loss
            if imp['type'] == 'bullish':
                sl = pb['low'] - atr_buffer * df.iloc[pb['end_idx']]['atr']
                # TP target (using fixed RR for labeling in this phase)
                target_rr = 1.5
                tp = entry_price + target_rr * (entry_price - sl)
            else:
                sl = pb['high'] + atr_buffer * df.iloc[pb['end_idx']]['atr']
                target_rr = 1.5
                tp = entry_price - target_rr * (sl - entry_price)
            
            # Simulation
            label = 0 # Default to failure
            end_search = min(entry_idx + max_bars, len(df) - 1)
            
            for i in range(entry_idx + 1, end_search + 1):
                high = df.iloc[i]['high']
                low = df.iloc[i]['low']
                
                if imp['type'] == 'bullish':
                    if low <= sl:
                        label = 0
                        break
                    if high >= tp:
                        label = 1
                        break
                else:
                    if high >= sl:
                        label = 0
                        break
                    if low <= tp:
                        label = 1
                        break
                        
            labels.append(label)
            
        return pd.Series(labels)
