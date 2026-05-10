
import asyncio
import ccxt
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from collections import defaultdict

try:
    from pattern.combined_detector import CombinedTASFibDetector
    from data.cleaner import DataCleaner
except ImportError:
    from impulse_fib_trader.pattern.combined_detector import CombinedTASFibDetector
    from impulse_fib_trader.data.cleaner import DataCleaner

class HonestComparison2025:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.cleaner = DataCleaner()
        self.detector = CombinedTASFibDetector()
        
        WHITELIST_PATH = 'impulse_fib_trader/config/whitelist.json'
        with open(WHITELIST_PATH, 'r') as f:
            self.whitelist = json.load(f)[:15] # Top 15 for accuracy and speed

    async def fetch_data(self, symbol):
        start_dt = datetime(2025, 1, 1)
        end_dt = datetime(2025, 12, 31, 23, 59)
        since = int(start_dt.timestamp() * 1000)
        all_ohlcv = []
        while since < int(end_dt.timestamp() * 1000):
            try:
                ohlcv = await asyncio.to_thread(self.exchange.fetch_ohlcv, symbol, '1h', since, 1000)
                if not ohlcv: break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 3600000
                if len(ohlcv) < 1000: break
                await asyncio.sleep(0.05)
            except: break
        if not all_ohlcv: return pd.DataFrame()
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def sim_martingale(self, df, p):
        """Реальная логика Мартингейла из tiered_2025."""
        idx = p['entry_idx']
        entry_1 = p['entry_price']
        sl = p['sl']
        risk_unit = abs(entry_1 - sl)
        if risk_unit == 0: return 0
        
        is_long = entry_1 > sl
        tp = entry_1 + (risk_unit * 2.0) if is_long else entry_1 - (risk_unit * 2.0)
        
        # Уровни добора (Сетка)
        levels = [
            {'price': entry_1, 'weight': 1, 'filled': True},
            {'price': entry_1 - risk_unit * 0.4 if is_long else entry_1 + risk_unit * 0.4, 'weight': 2, 'filled': False},
            {'price': entry_1 - risk_unit * 0.75 if is_long else entry_1 + risk_unit * 0.75, 'weight': 4, 'filled': False}
        ]
        
        curr_sl = sl
        max_reached = entry_1
        
        for j in range(idx + 1, min(idx + 50, len(df))):
            low, high = df.iloc[j]['low'], df.iloc[j]['high']
            
            # Проверка доборов
            for lvl in levels[1:]:
                if not lvl['filled']:
                    if (is_long and low <= lvl['price']) or (not is_long and high >= lvl['price']):
                        lvl['filled'] = True
            
            filled = [l for l in levels if l['filled']]
            total_w = sum(l['weight'] for l in filled)
            avg_entry = sum(l['price'] * l['weight'] for l in filled) / total_w
            
            # Динамический тейк для мартина (выход в небольшой плюс при доборах)
            curr_tp = tp if len(filled) == 1 else (avg_entry + risk_unit * 0.3 if is_long else avg_entry - risk_unit * 0.3)
            
            # Защита: Безубыток (BE)
            if is_long:
                if high > max_reached: max_reached = high
                if (max_reached / avg_entry - 1) >= 0.012:
                    be_level = avg_entry * 1.002
                    if be_level > curr_sl: curr_sl = be_level
                
                if low <= curr_sl: return (curr_sl - avg_entry) / risk_unit * total_w
                if high >= curr_tp: return (curr_tp - avg_entry) / risk_unit * total_w
            else:
                # Short sim... (skipped for brevity, but logic same)
                if low < max_reached: max_reached = low
                if (avg_entry / max_reached - 1) >= 0.012:
                    be_level = avg_entry * 0.998
                    if be_level < curr_sl: curr_sl = be_level
                if high >= curr_sl: return (avg_entry - curr_sl) / risk_unit * total_w
                if low <= curr_tp: return (avg_entry - curr_tp) / risk_unit * total_w
        return 0

    async def run(self):
        print(f"🚀 Running Accurate Comparison for 2025...")
        stats = {'Standard_Marti': 0, 'Fib75_Fixed': 0}
        
        for symbol in self.whitelist:
            df = await self.fetch_data(symbol)
            if df.empty: continue
            df = self.cleaner.calculate_indicators(df)
            
            patterns = self.detector.detect_patterns(df)
            
            pnl_m = 0
            pnl_f = 0
            for p in patterns:
                # 1. Martingale Simulation
                pnl_m += self.sim_martingale(df, p)
                
                # 2. Fib75 (Fixed logic - strictly reward based on entry/tp)
                risk = abs(p['entry_price'] - p['sl'])
                reward = abs(p['tp'] - p['entry_price'])
                rr = reward / risk if risk > 0 else 0
                
                res_f = 0
                for j in range(p['entry_idx'] + 1, min(p['entry_idx'] + 48, len(df))):
                    l, h = df.iloc[j]['low'], df.iloc[j]['high']
                    if p['entry_price'] > p['sl']: # Long
                        if l <= p['sl']: res_f = -1.0; break
                        if h >= p['tp']: res_f = rr; break
                pnl_f += res_f

            print(f"{symbol:<12} | Martingale: {pnl_m:>8.1f} R | Fib75: {pnl_f:>8.1f} R")
            stats['Standard_Marti'] += pnl_m
            stats['Fib75_Fixed'] += pnl_f

        print("-" * 50)
        print(f"FINAL 2025 | Martingale: {stats['Standard_Marti']:.1f} R | Fib75: {stats['Fib75_Fixed']:.1f} R")

if __name__ == "__main__":
    asyncio.run(HonestComparison2025().run())
