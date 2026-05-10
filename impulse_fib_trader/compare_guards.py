
import asyncio
import ccxt
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from collections import defaultdict

try:
    from pattern.combined_detector import CombinedTASFibDetector
    from data.cleaner import DataCleaner
except ImportError:
    from impulse_fib_trader.pattern.combined_detector import CombinedTASFibDetector
    from impulse_fib_trader.data.cleaner import DataCleaner

class ProfitGuardComparison:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.cleaner = DataCleaner()
        self.detector = CombinedTASFibDetector()
        WHITELIST_PATH = 'impulse_fib_trader/config/whitelist.json'
        with open(WHITELIST_PATH, 'r') as f:
            self.whitelist = json.load(f)[:15] # Top 15 for speed

    async def fetch_data(self, symbol, year):
        start_dt = datetime(year, 1, 1)
        end_dt = datetime(year, 12, 31, 23, 59)
        since = int(start_dt.timestamp() * 1000)
        all_ohlcv = []
        while since < int(end_dt.timestamp() * 1000):
            try:
                ohlcv = await asyncio.to_thread(self.exchange.fetch_ohlcv, symbol, '1h', since, 1000)
                if not ohlcv: break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 3600000
                if len(ohlcv) < 1000: break
                await asyncio.sleep(0.01)
            except: break
        if not all_ohlcv: return pd.DataFrame()
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def sim_trade(self, df, p, guard_type='fixed'):
        idx = p['entry_idx']
        entry_1 = p['entry_price']
        sl = p['sl']
        risk_unit = abs(entry_1 - sl)
        if risk_unit == 0: return 0
        is_long = entry_1 > sl
        tp = entry_1 + (risk_unit * 2.0) if is_long else entry_1 - (risk_unit * 2.0)
        
        # Grid Martingale
        levels = [
            {'price': entry_1, 'weight': 1, 'filled': True},
            {'price': entry_1 - risk_unit * 0.45 if is_long else entry_1 + risk_unit * 0.45, 'weight': 1.5, 'filled': False},
            {'price': entry_1 - risk_unit * 0.8 if is_long else entry_1 + risk_unit * 0.8, 'weight': 2.5, 'filled': False}
        ]
        
        curr_sl = sl
        max_pnl_seen = 0.0
        rsi_threshold_hit = False
        
        for j in range(idx + 1, min(idx + 72, len(df))):
            low, high, close = df.iloc[j]['low'], df.iloc[j]['high'], df.iloc[j]['close']
            rsi = df.iloc[j]['rsi']
            
            for lvl in levels[1:]:
                if not lvl['filled']:
                    if (is_long and low <= lvl['price']) or (not is_long and high >= lvl['price']):
                        lvl['filled'] = True
            
            filled = [l for l in levels if l['filled']]
            total_w = sum(l['weight'] for l in filled)
            avg_entry = sum(l['price'] * l['weight'] for l in filled) / total_w
            curr_pnl_pct = (close / avg_entry - 1) * 100 if is_long else (avg_entry / close - 1) * 100
            
            if curr_pnl_pct > max_pnl_seen: max_pnl_seen = curr_pnl_pct

            # EXIT LOGIC
            # 1. FIXED GUARD
            if guard_type == 'fixed':
                num = len(filled)
                if num == 1 and max_pnl_seen >= 3.0 and curr_pnl_pct <= 2.5: return 2.5 / (risk_unit/avg_entry*100) * total_w
                elif num == 2 and max_pnl_seen >= 2.0 and curr_pnl_pct <= 1.5: return 1.5 / (risk_unit/avg_entry*100) * total_w
                elif num == 3 and max_pnl_seen >= 1.0 and curr_pnl_pct <= 0.5: return 0.5 / (risk_unit/avg_entry*100) * total_w
            
            # 2. RSI GUARD (New)
            elif guard_type == 'rsi':
                if is_long:
                    if rsi >= 70: rsi_threshold_hit = True
                    if rsi_threshold_hit and rsi < 65 and curr_pnl_pct > 0.5:
                        return curr_pnl_pct / (risk_unit/avg_entry*100) * total_w
                else:
                    if rsi <= 30: rsi_threshold_hit = True
                    if rsi_threshold_hit and rsi > 35 and curr_pnl_pct > 0.5:
                        return curr_pnl_pct / (risk_unit/avg_entry*100) * total_w

            # TP/SL/BE
            curr_tp = (entry_1 + risk_unit * 2.0 if is_long else entry_1 - risk_unit * 2.0) if len(filled) == 1 else (avg_entry + risk_unit * 0.3 if is_long else avg_entry - risk_unit * 0.3)
            
            if is_long:
                if (high / avg_entry - 1) >= 0.012:
                    be = avg_entry * 1.002
                    if be > curr_sl: curr_sl = be
                if low <= curr_sl: return (curr_sl - avg_entry) / risk_unit * total_w
                if high >= curr_tp: return (curr_tp - avg_entry) / risk_unit * total_w
            else:
                if (avg_entry / low - 1) >= 0.012:
                    be = avg_entry * 0.998
                    if be < curr_sl: curr_sl = be
                if high >= curr_sl: return (avg_entry - curr_sl) / risk_unit * total_w
                if low <= curr_tp: return (avg_entry - curr_tp) / risk_unit * total_w
        return 0

    async def run(self):
        print("🚀 Comparing FIXED % Guard vs RSI Guard (2024-2025)...")
        results = {'fixed': 0, 'rsi': 0}
        
        for symbol in self.whitelist:
            print(f"Testing {symbol}...")
            data_all = []
            for y in [2024, 2025]:
                df = await self.fetch_data(symbol, y)
                if not df.empty: data_all.append(df)
            
            if not data_all: continue
            df = pd.concat(data_all)
            df = self.cleaner.calculate_indicators(df)
            patterns = self.detector.detect_patterns(df)
            
            for p in patterns:
                results['fixed'] += self.sim_trade(df, p, guard_type='fixed')
                results['rsi'] += self.sim_trade(df, p, guard_type='rsi')

        print("\n" + "="*50)
        print("FINAL RESULTS (Total R-Profit)")
        print(f"FIXED % GUARD: {results['fixed']:.1f} R")
        print(f"RSI 70/30 GUARD: {results['rsi']:.1f} R")
        print("="*50)

if __name__ == "__main__":
    asyncio.run(ProfitGuardComparison().run())
