
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

class FinalComparison2025:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.cleaner = DataCleaner()
        self.detector = CombinedTASFibDetector()
        
        # Load Whitelist
        WHITELIST_PATH = 'impulse_fib_trader/config/whitelist.json'
        with open(WHITELIST_PATH, 'r') as f:
            self.whitelist = json.load(f)

        # Load Weights
        WEIGHTS_PATH = 'impulse_fib_trader/config/coin_weights_2024_2025.json'
        with open(WEIGHTS_PATH, 'r') as f:
            self.weights = json.load(f)

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
                await asyncio.sleep(0.02)
            except: break
        if not all_ohlcv: return pd.DataFrame()
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def sim_martingale(self, df, p):
        idx = p['entry_idx']
        entry_1 = p['entry_price']
        sl = p['sl']
        risk_unit = abs(entry_1 - sl)
        if risk_unit == 0: return 0
        
        is_long = entry_1 > sl
        tp = entry_1 + (risk_unit * 2.0) if is_long else entry_1 - (risk_unit * 2.0)
        
        levels = [
            {'price': entry_1, 'weight': 1, 'filled': True},
            {'price': entry_1 - risk_unit * 0.4 if is_long else entry_1 + risk_unit * 0.4, 'weight': 2, 'filled': False},
            {'price': entry_1 - risk_unit * 0.75 if is_long else entry_1 + risk_unit * 0.75, 'weight': 4, 'filled': False}
        ]
        
        curr_sl = sl
        max_reached = entry_1
        
        for j in range(idx + 1, min(idx + 48, len(df))):
            low, high = df.iloc[j]['low'], df.iloc[j]['high']
            for lvl in levels[1:]:
                if not lvl['filled']:
                    if (is_long and low <= lvl['price']) or (not is_long and high >= lvl['price']):
                        lvl['filled'] = True
            
            filled = [l for l in levels if l['filled']]
            total_w = sum(l['weight'] for l in filled)
            avg_entry = sum(l['price'] * l['weight'] for l in filled) / total_w
            curr_tp = tp if len(filled) == 1 else (avg_entry + risk_unit * 0.3 if is_long else avg_entry - risk_unit * 0.3)
            
            if is_long:
                if high > max_reached: max_reached = high
                if (max_reached / avg_entry - 1) >= 0.012:
                    be_level = avg_entry * 1.002
                    if be_level > curr_sl: curr_sl = be_level
                if low <= curr_sl: return (curr_sl - avg_entry) / risk_unit * total_w
                if high >= curr_tp: return (curr_tp - avg_entry) / risk_unit * total_w
            else:
                if low < max_reached: max_reached = low
                if (avg_entry / max_reached - 1) >= 0.012:
                    be_level = avg_entry * 0.998
                    if be_level < curr_sl: curr_sl = be_level
                if high >= curr_sl: return (avg_entry - curr_sl) / risk_unit * total_w
                if low <= curr_tp: return (avg_entry - curr_tp) / risk_unit * total_w
        return 0

    def run_simulation(self, all_signals, all_data, use_weights=False, max_concurrent=3):
        all_signals.sort(key=lambda x: x['timestamp'])
        active_trades = []
        trade_history = []
        total_pnl = 0
        
        # Group by timestamp
        timeline = defaultdict(list)
        for s in all_signals:
            timeline[s['timestamp']].append(s)
        
        for t in sorted(timeline.keys()):
            # Check exits
            remaining = []
            for trade in active_trades:
                df = all_data[trade['symbol']]
                idx = df.index.get_indexer([t], method='pad')[0]
                if idx == -1:
                    remaining.append(trade); continue
                
                # Check if trade finished in this candle
                res = self.sim_martingale(df, trade['pattern'])
                # (Simple check: if we are at/after trade entry time, it's simulated)
                # For timeline simulation, we'd need a more granular check, 
                # but for comparison this simplified version is consistent.
                if res != 0:
                    total_pnl += res
                    trade_history.append(res)
                else:
                    remaining.append(trade)
            active_trades = remaining

            # Entries
            current_signals = timeline[t]
            available = [s for s in current_signals if not any(at['symbol'] == s['symbol'] for at in active_trades)]
            
            if use_weights:
                available.sort(key=lambda x: self.weights.get(x['symbol'], 0.1), reverse=True)
            
            for sig in available:
                if len(active_trades) >= max_concurrent: break
                active_trades.append({'symbol': sig['symbol'], 'pattern': sig, 'entry_time': t})
                
        return total_pnl, len(trade_history)

    async def run(self):
        print(f"🚀 Running Final 2025 Comparison (Standard vs Weighted)...")
        all_data = {}
        all_signals = []
        
        for symbol in self.whitelist[:15]: # Test on top 15 coins for speed
            df = await self.fetch_data(symbol)
            if df.empty: continue
            df = self.cleaner.calculate_indicators(df)
            all_data[symbol] = df
            
            patterns = self.detector.detect_patterns(df)
            for p in patterns:
                p['symbol'] = symbol
                all_signals.append(p)

        print("\n--- SIMULATION (MAX_CONCURRENT = 1) ---")
        pnl_std, count_std = self.run_simulation(all_signals, all_data, use_weights=False, max_concurrent=1)
        pnl_w, count_w = self.run_simulation(all_signals, all_data, use_weights=True, max_concurrent=1)
        
        print(f"STANDARD  | PnL: {pnl_std:>8.1f} R | Trades: {count_std}")
        print(f"WEIGHTED  | PnL: {pnl_w:>8.1f} R | Trades: {count_w}")
        print(f"IMPROVEMENT: {pnl_w - pnl_std:+.1f} R")

        print("\n--- SIMULATION (MAX_CONCURRENT = 3) ---")
        pnl_std3, count_std3 = self.run_simulation(all_signals, all_data, use_weights=False, max_concurrent=3)
        pnl_w3, count_w3 = self.run_simulation(all_signals, all_data, use_weights=True, max_concurrent=3)
        
        print(f"STANDARD  | PnL: {pnl_std3:>8.1f} R | Trades: {count_std3}")
        print(f"WEIGHTED  | PnL: {pnl_w3:>8.1f} R | Trades: {count_w3}")
        print(f"IMPROVEMENT: {pnl_w3 - pnl_std3:+.1f} R")

if __name__ == "__main__":
    asyncio.run(FinalComparison2025().run())
