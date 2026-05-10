import pandas as pd
import numpy as np
from typing import List, Dict
from pattern.impulse import ImpulseDetector
from pattern.pullback import PullbackMeasurer

class CombinedTASFibDetector:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'impulse_detection': {
                'min_atr_multiplier': 1.8,
                'min_candles': 4,
                'min_body_ratio': 0.6,
                'max_internal_retracement': 0.3
            },
            'pullback_requirements': {
                'min_retracement': 0.60, # Возвращаем 0.60 для RR
                'max_retracement': 0.85,
                'touch_50_level': True
            }
        }
        self.impulse_detector = ImpulseDetector(self.config)
        self.pullback_measurer = PullbackMeasurer(self.config)

    def detect_patterns(self, df: pd.DataFrame) -> List[Dict]:
        if 'atr' not in df.columns:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            df['atr'] = ranges.max(axis=1).rolling(14).mean()
        
        if 'ema_200' not in df.columns:
            df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

        patterns = []
        impulses = self.impulse_detector.detect(df)
        
        for impulse in impulses:
            pullback = self.pullback_measurer.measure(impulse, df)
            
            if pullback:
                last_idx = pullback['end_idx']
                rsi_val = df['rsi'].iloc[last_idx] if 'rsi' in df.columns else 50
                ema_val = df['ema_200'].iloc[last_idx]
                curr_price = df['close'].iloc[last_idx]
                
                # Фильтры индикаторов
                ema_val = df['ema_200'].iloc[last_idx]
                macd_hist = df['macd_hist'].iloc[last_idx] if 'macd_hist' in df.columns else 0
                curr_price = df['close'].iloc[last_idx]
                
                # LONG: Price > EMA 200 И MACD Histogram > 0 (растущий моментум)
                if impulse['type'] == 'bullish' and curr_price > ema_val:
                    if macd_hist <= 0:
                        continue # Пропускаем, если моментум еще падает
                        
                    entry_price = curr_price
                    sl = impulse['low'] * 0.998
                    tp = impulse['high']
                    
                    risk = entry_price - sl
                    reward = tp - entry_price
                    # Снижаем RR до 1.2 для консервативности, так как TP теперь ближе
                    if risk > 0 and reward / risk >= 1.2:
                        patterns.append({
                            'type': 'IFPC_STRICT',
                            'entry_idx': last_idx,
                            'entry_price': entry_price,
                            'p0': impulse['low'],
                            'p1': impulse['high'],
                            'retracement': pullback['depth'],
                            'rsi': rsi_val,
                            'sl': sl,
                            'tp': tp,
                            'timestamp': df.index[last_idx]
                        })
                
                # SHORT: Price < EMA 200 И MACD Histogram < 0
                elif impulse['type'] == 'bearish' and curr_price < ema_val:
                    if macd_hist >= 0:
                        continue # Пропускаем, если моментум уже растет (отскок)
                        
                    entry_price = curr_price
                    sl = impulse['high'] * 1.002
                    tp = impulse['low']
                    
                    risk = sl - entry_price
                    reward = entry_price - tp
                    if risk > 0 and reward / risk >= 1.2:
                        patterns.append({
                            'type': 'IFPC_STRICT_SHORT',
                            'entry_idx': last_idx,
                            'entry_price': entry_price,
                            'p0': impulse['high'],
                            'p1': impulse['low'],
                            'retracement': pullback['depth'],
                            'rsi': rsi_val,
                            'sl': sl,
                            'tp': tp,
                            'timestamp': df.index[last_idx]
                        })
        
        unique_patterns = []
        last_entry = -100
        for p in patterns:
            if p['entry_idx'] > last_entry + 5:
                unique_patterns.append(p)
                last_entry = p['entry_idx']
                
        return unique_patterns
