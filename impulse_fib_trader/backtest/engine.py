import pandas as pd
import numpy as np
from typing import List, Dict

class BacktestEngine:
    def __init__(self, config: Dict):
        self.config = config['risk_management']

    def run_backtest(self, patterns: List[Dict], df: pd.DataFrame, entry_mode: str = 'BREAKOUT') -> pd.DataFrame:
        """
        Runs a rule-based backtest.
        entry_mode: 'BREAKOUT' (current) or 'LIMIT' (0.618 Fib level).
        """
        results = []
        atr_buffer = self.config['stop_loss']['buffer_atr']
        max_bars = self.config['max_bars_in_trade']
        
        for p in patterns:
            imp = p['impulse']
            pb = p['pullback']
            
            # 1. Determine Entry Price and Index
            if entry_mode == 'BREAKOUT':
                # Текущая логика: вход после пробоя структуры
                if 'structure' not in p or not p['structure']: continue
                entry_idx = p['structure']['entry_idx']
                entry_price = p['structure']['entry_price']
            else:
                # Новая логика: вход лимиткой на уровне 0.618 Фибоначчи
                entry_idx = pb['end_idx'] # Момент, когда цена коснулась уровня
                # Рассчитываем точный уровень 0.618
                fib_level = 0.618
                if imp['type'] == 'bullish':
                    entry_price = imp['high'] - (imp['range'] * fib_level)
                else:
                    entry_price = imp['low'] + (imp['range'] * fib_level)

            # 2. SL & TP
            if imp['type'] == 'bullish':
                sl = pb['low'] - atr_buffer * df.iloc[pb['end_idx']]['atr']
                # TP ставим выше: для лимитки RR обычно лучше
                tp = entry_price + 2.5 * (entry_price - sl)
            else:
                sl = pb['high'] + atr_buffer * df.iloc[pb['end_idx']]['atr']
                tp = entry_price - 2.5 * (sl - entry_price)
            
            risk = abs(entry_price - sl)
            if risk == 0: continue
            
            # 3. Simulation
            trade_result = None
            exit_idx = None
            exit_price = None
            max_p = entry_price if imp['type'] == 'bullish' else entry_price
            current_sl = sl
            
            # Настройки (можно вынести в конфиг)
            BE_TRIGGER = 0.012 
            BE_LEVEL = 0.002
            TRAILING_TRIGGER = 0.02
            TRAILING_DIST = 0.012
            
            for i in range(entry_idx + 1, min(entry_idx + max_bars + 1, len(df))):
                high = df.iloc[i]['high']
                low = df.iloc[i]['low']
                close = df.iloc[i]['close']
                
                if imp['type'] == 'bullish':
                    if high > max_p: max_p = high
                    
                    # Логика безубытка
                    if (max_p / entry_price - 1) >= BE_TRIGGER:
                        be_sl = entry_price * (1 + BE_LEVEL)
                        if be_sl > current_sl: current_sl = be_sl
                    
                    # Логика трейлинга
                    if (max_p / entry_price - 1) >= TRAILING_TRIGGER:
                        t_sl = max_p * (1 - TRAILING_DIST)
                        if t_sl > current_sl: current_sl = t_sl

                    if low <= current_sl:
                        trade_result = (current_sl - entry_price) / risk
                        exit_idx = i
                        exit_price = current_sl
                        break
                    if high >= tp:
                        trade_result = (tp - entry_price) / risk
                        exit_idx = i
                        exit_price = tp
                        break
                else:
                    # Bearish logic (optional, mainly spot bot but for completeness)
                    if low < max_p: max_p = low # max_p here is min_p
                    
                    if (entry_price / max_p - 1) >= BE_TRIGGER:
                        be_sl = entry_price * (1 - BE_LEVEL)
                        if be_sl < current_sl: current_sl = be_sl
                        
                    if (entry_price / max_p - 1) >= TRAILING_TRIGGER:
                        t_sl = max_p * (1 + TRAILING_DIST)
                        if t_sl < current_sl: current_sl = t_sl

                    if high >= current_sl:
                        trade_result = (entry_price - current_sl) / risk
                        exit_idx = i
                        exit_price = current_sl
                        break
                    if low <= tp:
                        trade_result = (entry_price - tp) / risk
                        exit_idx = i
                        exit_price = tp
                        break
                        
            if trade_result is None:
                exit_idx = min(entry_idx + max_bars, len(df) - 1)
                exit_price = df.iloc[exit_idx]['close']
                trade_result = (exit_price - entry_price) / risk if imp['type'] == 'bullish' else (entry_price - exit_price) / risk
            
            results.append({
                'symbol': p.get('symbol', 'UNKNOWN'),
                'entry_idx': entry_idx,
                'r_multiple': trade_result,
                'type': imp['type']
            })
            
        return pd.DataFrame(results)
