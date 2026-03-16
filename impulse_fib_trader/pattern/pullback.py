import pandas as pd
import numpy as np
from typing import List, Dict, Optional

class PullbackMeasurer:
    def __init__(self, config: Dict):
        # Support both old and new config structures
        self.config = config.get('pullback_requirements', 
                                config.get('pullback_detection', {}))
        self.old_config = config.get('pullback_detection', {})

    def measure(self, impulse: Dict, df: pd.DataFrame) -> Optional[Dict]:
        """
        Measures if a pullback following an impulse is valid.
        """
        start_idx = impulse['end_idx'] + 1
        # Увеличиваем до 48 часов (2 дня на H1), чтобы ловить долгие боковики
        max_duration = 48 
        
        # New config keys
        fib_min = self.config.get('min_retracement', 
                                 self.old_config.get('fib_range', {}).get('min', 0.50))
        fib_max = self.config.get('max_retracement', 
                                 self.old_config.get('fib_range', {}).get('max', 0.705))
        
        touch_50_required = self.config.get('touch_50_level', False)
        
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
                if pullback_low < impulse_low:
                    return None
                    
                retracement = (impulse_high - pullback_low) / impulse_range
                
                # Check touch 50 level if required
                touched_50 = (impulse_high - window['low'].min()) / impulse_range >= 0.5
                if touch_50_required and not touched_50:
                    continue
                
                if fib_min <= retracement <= fib_max:
                    return {
                        'start_idx': start_idx,
                        'end_idx': current_idx,
                        'depth': retracement,
                        'low': pullback_low,
                        'high': window['high'].max(),
                        'touched_50': touched_50
                    }
            else: # bearish
                pullback_high = window['high'].max()
                if pullback_high > impulse_high:
                    return None
                    
                retracement = (pullback_high - impulse_low) / impulse_range
                
                touched_50 = (window['high'].max() - impulse_low) / impulse_range >= 0.5
                if touch_50_required and not touched_50:
                    continue
                
                if fib_min <= retracement <= fib_max:
                    return {
                        'start_idx': start_idx,
                        'end_idx': current_idx,
                        'depth': retracement,
                        'high': pullback_high,
                        'low': window['low'].min(),
                        'touched_50': touched_50
                    }
                    
        return None
