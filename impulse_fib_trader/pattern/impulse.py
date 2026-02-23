import pandas as pd
import numpy as np
from typing import List, Dict

class ImpulseDetector:
    def __init__(self, config: Dict):
        self.config = config['impulse_detection']

    def detect(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detects impulses in the data.
        Returns a list of impulse metadata.
        """
        impulses = []
        min_candles = self.config['min_candles']
        min_atr_mult = self.config['min_atr_multiplier']
        min_body_ratio = self.config['min_body_ratio']
        max_internal_retr = self.config['max_internal_retracement']

        # We need a rolling window to detect sequences
        # For simplicity in this version, we'll use a sliding window approach
        # optimized with numpy where possible.
        
        closes = df['close'].values
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        atr = df['atr'].values
        
        for i in range(len(df) - min_candles):
            # Check for bullish impulse
            for length in range(min_candles, min_candles + 10): # Look for impulses of 4-14 candles
                if i + length >= len(df):
                    break
                    
                start_price = opens[i]
                end_price = closes[i+length-1]
                total_range = highs[i:i+length].max() - lows[i:i+length].min()
                net_move = end_price - start_price
                
                if net_move <= 0:
                    continue # Not bullish
                
                # ATR check (move must be > N * ATR)
                if net_move < min_atr_mult * atr[i]:
                    continue
                
                # Body ratio check (sum of bodies / total range)
                bodies = np.abs(closes[i:i+length] - opens[i:i+length])
                if bodies.sum() / total_range < min_body_ratio:
                    continue
                
                # Internal retracement check
                highest_point = highs[i:i+length].max()
                if (highest_point - end_price) / net_move > max_internal_retr:
                    continue
                
                # If all checks pass
                impulses.append({
                    'type': 'bullish',
                    'start_idx': i,
                    'end_idx': i + length - 1,
                    'start_price': start_price,
                    'end_price': end_price,
                    'high': highs[i:i+length].max(),
                    'low': lows[i:i+length].min(),
                    'range': net_move
                })
                break # Found one, move to next possible start

            # Check for bearish impulse
            for length in range(min_candles, min_candles + 10):
                if i + length >= len(df):
                    break
                    
                start_price = opens[i]
                end_price = closes[i+length-1]
                total_range = highs[i:i+length].max() - lows[i:i+length].min()
                net_move = start_price - end_price
                
                if net_move <= 0:
                    continue # Not bearish
                
                if net_move < min_atr_mult * atr[i]:
                    continue
                
                bodies = np.abs(closes[i:i+length] - opens[i:i+length])
                if bodies.sum() / total_range < min_body_ratio:
                    continue
                
                lowest_point = lows[i:i+length].min()
                if (end_price - lowest_point) / net_move > max_internal_retr:
                    continue
                
                impulses.append({
                    'type': 'bearish',
                    'start_idx': i,
                    'end_idx': i + length - 1,
                    'start_price': start_price,
                    'end_price': end_price,
                    'high': highs[i:i+length].max(),
                    'low': lows[i:i+length].min(),
                    'range': net_move
                })
                break
                
        return impulses
