import pandas as pd
import numpy as np
from typing import List, Dict

class BacktestEngine:
    def __init__(self, config: Dict):
        self.config = config['risk_management']

    def run_backtest(self, patterns: List[Dict], df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs a rule-based backtest for the detected patterns.
        """
        results = []
        atr_buffer = self.config['stop_loss']['buffer_atr']
        max_bars = self.config['max_bars_in_trade']
        
        for p in patterns:
            imp = p['impulse']
            pb = p['pullback']
            struct = p['structure']
            
            entry_idx = struct['entry_idx']
            entry_price = struct['entry_price']
            
            # 1. SL & TP
            if imp['type'] == 'bullish':
                sl = pb['low'] - atr_buffer * df.iloc[pb['end_idx']]['atr']
                tp_ext = imp['high'] + (imp['high'] - imp['low']) * 0.272 # Fib extension 1.272
                tp_rr = entry_price + 2.5 * (entry_price - sl)
                tp = min(tp_ext, tp_rr) # Conservative TP
            else:
                sl = pb['high'] + atr_buffer * df.iloc[pb['end_idx']]['atr']
                tp_ext = imp['low'] - (imp['high'] - imp['low']) * 0.272
                tp_rr = entry_price - 2.5 * (sl - entry_price)
                tp = max(tp_ext, tp_rr)
            
            # Risk/Reward calculations
            risk = abs(entry_price - sl)
            reward = abs(tp - entry_price)
            rr_ratio = reward / risk if risk > 0 else 0
            
            # 2. Simulation
            trade_result = None
            exit_idx = None
            exit_price = None
            
            for i in range(entry_idx + 1, min(entry_idx + max_bars + 1, len(df))):
                high = df.iloc[i]['high']
                low = df.iloc[i]['low']
                close = df.iloc[i]['close']
                
                if imp['type'] == 'bullish':
                    if low <= sl:
                        trade_result = -1.0 # -1 R
                        exit_idx = i
                        exit_price = sl
                        break
                    if high >= tp:
                        trade_result = rr_ratio
                        exit_idx = i
                        exit_price = tp
                        break
                else:
                    if high >= sl:
                        trade_result = -1.0
                        exit_idx = i
                        exit_price = sl
                        break
                    if low <= tp:
                        trade_result = rr_ratio
                        exit_idx = i
                        exit_price = tp
                        break
                        
            if trade_result is None:
                # Timed out exit
                exit_idx = min(entry_idx + max_bars, len(df) - 1)
                exit_price = df.iloc[exit_idx]['close']
                if imp['type'] == 'bullish':
                    trade_result = (exit_price - entry_price) / risk if risk > 0 else 0
                else:
                    trade_result = (entry_price - exit_price) / risk if risk > 0 else 0
            
            results.append({
                'entry_idx': entry_idx,
                'exit_idx': exit_idx,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'type': imp['type'],
                'risk': risk,
                'reward': reward,
                'r_multiple': trade_result,
                'timestamp': df.iloc[entry_idx]['timestamp']
            })
            
        return pd.DataFrame(results)
