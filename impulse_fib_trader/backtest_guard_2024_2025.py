
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

class FinalBacktest2024_2025:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.cleaner = DataCleaner()
        self.detector = CombinedTASFibDetector()
        
        WHITELIST_PATH = 'impulse_fib_trader/config/whitelist.json'
        with open(WHITELIST_PATH, 'r') as f:
            self.whitelist = json.load(f)

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

    def sim_martingale_with_guard(self, df, p):
        idx = p['entry_idx']
        entry_1 = p['entry_price']
        sl = p['sl']
        risk_unit = abs(entry_1 - sl)
        if risk_unit == 0: return 0
        
        is_long = entry_1 > sl
        tp = entry_1 + (risk_unit * 2.0) if is_long else entry_1 - (risk_unit * 2.0)
        
        levels = [
            {'price': entry_1, 'weight': 1, 'filled': True},
            {'price': entry_1 - risk_unit * 0.45 if is_long else entry_1 + risk_unit * 0.45, 'weight': 1.5, 'filled': False},
            {'price': entry_1 - risk_unit * 0.8 if is_long else entry_1 + risk_unit * 0.8, 'weight': 2.5, 'filled': False}
        ]
        
        curr_sl = sl
        max_reached = entry_1
        max_pnl_seen = 0.0
        
        for j in range(idx + 1, min(idx + 72, len(df))): # Up to 72h trade duration
            low, high = df.iloc[j]['low'], df.iloc[j]['high']
            
            # Check Grid Fills
            for lvl in levels[1:]:
                if not lvl['filled']:
                    if (is_long and low <= lvl['price']) or (not is_long and high >= lvl['price']):
                        lvl['filled'] = True
            
            filled = [l for l in levels if l['filled']]
            total_w = sum(l['weight'] for l in filled)
            avg_entry = sum(l['price'] * l['weight'] for l in filled) / total_w
            
            # Dynamic TP
            curr_tp = tp if len(filled) == 1 else (avg_entry + risk_unit * 0.3 if is_long else avg_entry - risk_unit * 0.3)
            
            # Update Max PnL
            curr_pnl_pct = (high / avg_entry - 1) * 100 if is_long else (avg_entry / low - 1) * 100
            if curr_pnl_pct > max_pnl_seen: max_pnl_seen = curr_pnl_pct

            # PROFIT GUARD EXIT
            num = len(filled)
            if num == 1 and max_pnl_seen >= 3.0:
                p_current = (df.iloc[j]['close'] / avg_entry - 1) * 100 if is_long else (avg_entry / df.iloc[j]['close'] - 1) * 100
                if p_current <= 2.5: return 2.5 / (risk_unit/avg_entry*100) * total_w
            elif num == 2 and max_pnl_seen >= 2.0:
                p_current = (df.iloc[j]['close'] / avg_entry - 1) * 100 if is_long else (avg_entry / df.iloc[j]['close'] - 1) * 100
                if p_current <= 1.5: return 1.5 / (risk_unit/avg_entry*100) * total_w
            elif num == 3 and max_pnl_seen >= 1.0:
                p_current = (df.iloc[j]['close'] / avg_entry - 1) * 100 if is_long else (avg_entry / df.iloc[j]['close'] - 1) * 100
                if p_current <= 0.5: return 0.5 / (risk_unit/avg_entry*100) * total_w

            # Breakeven
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

    async def run_for_year(self, year):
        print(f"\n📅 TESTING YEAR {year}...")
        results = {}
        for symbol in self.whitelist[:20]: # Top 20 for speed/accuracy
            df = await self.fetch_data(symbol, year)
            if df.empty: continue
            df = self.cleaner.calculate_indicators(df)
            patterns = self.detector.detect_patterns(df)
            
            pnl = 0
            trades = 0
            for p in patterns:
                res = self.sim_martingale_with_guard(df, p)
                if res != 0:
                    pnl += res
                    trades += 1
            results[symbol] = {'pnl': pnl, 'trades': trades}
            print(f" {symbol}: {pnl:+.1f} R ({trades} trades)")
        
        total_pnl = sum(r['pnl'] for r in results.values())
        total_trades = sum(r['trades'] for r in results.values())
        print(f"--- YEAR {year} TOTAL: {total_pnl:.1f} R ({total_trades} trades) ---")
        return total_pnl

    async def run(self):
        p2024 = await self.run_for_year(2024)
        p2025 = await self.run_for_year(2025)
        print("\n" + "="*50)
        print(f"FINAL REPORT (Martingale + Profit Guard)")
        print(f"2024: {p2024:+.1f} R")
        print(f"2025: {p2025:+.1f} R")
        print(f"TOTAL: {p2024+p2025:+.1f} R")
        print("="*50)

if __name__ == "__main__":
    asyncio.run(FinalBacktest2024_2025().run())
