import pandas as pd
from typing import Dict, Optional

class StructureValidator:
    def __init__(self, config: Dict):
        self.config = config.get('entry_trigger', 
                                config.get('structure_requirements', {}))
        self.risk_config = config.get('risk_management', {})

    def validate(self, impulse: Dict, pullback: Dict, df: pd.DataFrame) -> Optional[Dict]:
        """
        Validates the entry based on the trigger type.
        """
        start_idx = pullback['end_idx'] + 1
        max_bars = self.risk_config.get('max_bars_in_trade', 40)
        
        if start_idx >= len(df):
            return None

        trigger_type = self.config.get('type', 'close_beyond_structure')
        
        for i in range(start_idx, min(start_idx + max_bars, len(df))):
            candle = df.iloc[i]
            
            if trigger_type == 'false_break_wick_only':
                # For bullish: false break of pullback low (discount zone sweep)
                if impulse['type'] == 'bullish':
                    sweep_level = pullback['low']
                    if candle['low'] < sweep_level and candle['close'] > sweep_level:
                        return {
                            'entry_idx': i,
                            'entry_price': candle['close'],
                            'confirmation': 'false_break_wick_only',
                            'stop_loss': candle['low'] - 0.0001 # Simple buffer
                        }
                else: # bearish: false break of pullback high (premium zone sweep)
                    sweep_level = pullback['high']
                    if candle['high'] > sweep_level and candle['close'] < sweep_level:
                        return {
                            'entry_idx': i,
                            'entry_price': candle['close'],
                            'confirmation': 'false_break_wick_only',
                            'stop_loss': candle['high'] + 0.0001
                        }
            else:
                # Default: breakout of impulse extreme
                if impulse['type'] == 'bullish':
                    if candle['close'] > impulse['high']:
                        return {
                            'entry_idx': i,
                            'entry_price': candle['close'],
                            'confirmation': 'close_beyond_high'
                        }
                else:
                    if candle['close'] < impulse['low']:
                        return {
                            'entry_idx': i,
                            'entry_price': candle['close'],
                            'confirmation': 'close_beyond_low'
                        }
                    
        return None
