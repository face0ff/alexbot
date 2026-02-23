import pandas as pd
from typing import Dict, Optional

class StructureValidator:
    def __init__(self, config: Dict):
        self.config = config['structure_requirements']

    def validate(self, impulse: Dict, pullback: Dict, df: pd.DataFrame) -> Optional[Dict]:
        """
        Validates if the price breaks the impulse extreme after a pullback.
        """
        start_idx = pullback['end_idx'] + 1
        impulse_high = impulse['high']
        impulse_low = impulse['low']
        max_bars = 40 # from risk_management
        
        if start_idx >= len(df):
            return None
            
        for i in range(start_idx, min(start_idx + max_bars, len(df))):
            current_close = df.iloc[i]['close']
            
            if impulse['type'] == 'bullish':
                # Check for break of impulse high
                if current_close > impulse_high:
                    return {
                        'entry_idx': i,
                        'entry_price': current_close,
                        'confirmation': 'close_beyond_high'
                    }
                # Check for break of pullback low (invalidation)
                if current_close < pullback['low']:
                    return None
            else: # bearish
                # Check for break of impulse low
                if current_close < impulse_low:
                    return {
                        'entry_idx': i,
                        'entry_price': current_close,
                        'confirmation': 'close_beyond_low'
                    }
                # Check for break of pullback high (invalidation)
                if current_close > pullback['high']:
                    return None
                    
        return None
