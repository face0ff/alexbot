import pandas as pd
import numpy as np
from typing import List, Dict, Optional

class PullbackMeasurer:
    def __init__(self, config: Dict):
        self.config = config['pullback_detection']

    def measure(self, impulse: Dict, df: pd.DataFrame) -> Optional[Dict]:
        """
        Measures if a pullback following an impulse is valid.
        """
        start_idx = impulse['end_idx'] + 1
        max_duration = self.config['max_duration_candles']
        fib_min = self.config['fib_range']['min']
        fib_max = self.config['fib_range']['max']
        
        if start_idx >= len(df):
            return None
            
        impulse_range = impulse['range']
        impulse_high = impulse['high']
        impulse_low = impulse['low']
        
        for length in range(1, max_duration + 1):
            current_idx = start_idx + length - 1
            if current_idx >= len(df):
                break
                
            window = df.iloc[start_idx : current_idx + 1]
            
            if impulse['type'] == 'bullish':
                pullback_low = window['low'].min()
                # Check if price broke below impulse start (invalidated)
                if pullback_low < impulse_low:
                    return None
                    
                retracement = (impulse_high - pullback_low) / impulse_range
                
                # Check if current close is within Fib range or we found a swing low in Fib range
                if fib_min <= retracement <= fib_max:
                    # Slowdown check: average candle size in pullback < impulse avg
                    impulse_avg_body = np.abs(df.iloc[impulse['start_idx']:impulse['end_idx']+1]['close'] - 
                                            df.iloc[impulse['start_idx']:impulse['end_idx']+1]['open']).mean()
                    pullback_avg_body = np.abs(window['close'] - window['open']).mean()
                    
                    if self.config['require_slowdown'] and pullback_avg_body >= impulse_avg_body:
                        continue
                        
                    return {
                        'start_idx': start_idx,
                        'end_idx': current_idx,
                        'depth': retracement,
                        'low': pullback_low,
                        'high': window['high'].max()
                    }
            else: # bearish
                pullback_high = window['high'].max()
                if pullback_high > impulse_high:
                    return None
                    
                retracement = (pullback_high - impulse_low) / impulse_range
                
                if fib_min <= retracement <= fib_max:
                    impulse_avg_body = np.abs(df.iloc[impulse['start_idx']:impulse['end_idx']+1]['close'] - 
                                            df.iloc[impulse['start_idx']:impulse['end_idx']+1]['open']).mean()
                    pullback_avg_body = np.abs(window['close'] - window['open']).mean()
                    
                    if self.config['require_slowdown'] and pullback_avg_body >= impulse_avg_body:
                        continue
                        
                    return {
                        'start_idx': start_idx,
                        'end_idx': current_idx,
                        'depth': retracement,
                        'high': pullback_high,
                        'low': window['low'].min()
                    }
                    
        return None
