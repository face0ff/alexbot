
import asyncio
import ccxt
import pandas as pd
import numpy as np
import os
import json
import joblib
from datetime import datetime, timedelta
from typing import List, Dict

# Import existing modules
try:
    from pattern.combined_detector import CombinedTASFibDetector
    from data.cleaner import DataCleaner
except ImportError:
    from impulse_fib_trader.pattern.combined_detector import CombinedTASFibDetector
    from impulse_fib_trader.data.cleaner import DataCleaner

# Configuration
TIMEFRAME = '1h' 
YEAR = 2025
WHITELIST_PATH = 'impulse_fib_trader/config/whitelist.json'

class MartingaleComparison:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.cleaner = DataCleaner()
        self.detector = CombinedTASFibDetector()
        
        with open(WHITELIST_PATH, 'r') as f:
            self.whitelist = json.load(f)[:20]

    async def fetch_year_data(self, symbol: str, year: int):
        print(f"Fetching {symbol} for {year}...")
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
                await asyncio.sleep(0.05)
            except: break
        if not all_ohlcv: return pd.DataFrame()
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def run_sim(self, signals, all_data, weights=None, max_concurrent=1):
        signals.sort(key=lambda x: x['timestamp'])
        active_trades = []
        trade_history = []
        
        from collections import defaultdict
        timeline = defaultdict(list)
        for sig in signals:
            timeline[sig['timestamp']].append(sig)
            
        sorted_times = sorted(timeline.keys())
        
        total_pnl = 0
        for t in sorted_times:
            # 1. Check exits
            remaining = []
            for trade in active_trades:
                df = all_data[trade['symbol']]
                idx = df.index.get_indexer([t], method='pad')[0]
                if idx == -1:
                    remaining.append(trade)
                    continue
                
                low, high = df.iloc[idx]['low'], df.iloc[idx]['high']
                res = 0
                if low <= trade['sl']: res = -1.2 
                elif high >= trade['tp']: res = 1.0 
                
                if res != 0:
                    total_pnl += res
                    trade_history.append(res)
                else:
                    remaining.append(trade)
            active_trades = remaining
            
            # 2. Entries
            current_signals = timeline[t]
            available = [s for s in current_signals if not any(at['symbol'] == s['symbol'] for at in active_trades)]
            
            if weights:
                available.sort(key=lambda x: weights.get(x['symbol'], 0), reverse=True)
            
            for sig in available:
                if len(active_trades) >= max_concurrent: break
                
                risk = sig['entry_price'] - sig['sl']
                if risk <= 0: continue
                
                active_trades.append({
                    'symbol': sig['symbol'],
                    'sl': sig['sl'],
                    'tp': sig['entry_price'] + risk * 1.5,
                    'entry_time': t
                })
                
        return total_pnl, len(trade_history)

    async def run(self):
        all_data = {}
        for s in self.whitelist:
            df = await self.fetch_year_data(s, 2025)
            if not df.empty:
                all_data[s] = self.cleaner.calculate_indicators(df)
        
        all_signals = []
        for symbol, df in all_data.items():
            patterns = self.detector.detect_patterns(df)
            for p in patterns:
                p['symbol'] = symbol
                all_signals.append(p)
        
        # Standard Martingale (No Weights)
        pnl_std, trades_std = self.run_sim(all_signals, all_data, weights=None)
        
        # Load weights from 2024 report logic
        weights_2024 = {
            "BTC/USDT": 1.0,
            "DOGE/USDT": 0.95,
            "LTC/USDT": 0.92,
            "BNB/USDT": 0.8,
            "XRP/USDT": 0.7,
            "ADA/USDT": 0.5,
            "AVAX/USDT": 0.4
        }
        
        pnl_weighted, trades_weighted = self.run_sim(all_signals, all_data, weights=weights_2024)
        
        print("\n" + "="*50)
        print(f"COMPARISON 2025: STANDARD VS WEIGHTED")
        print("="*50)
        print(f"STANDARD  | PnL: {pnl_std:>8.1f} R | Trades: {trades_std}")
        print(f"WEIGHTED  | PnL: {pnl_weighted:>8.1f} R | Trades: {trades_weighted}")
        print("="*50)
        print("Best Coins in 2024 (Weights Source): BTC, DOGE, LTC, BNB")
        print("="*50)

if __name__ == "__main__":
    asyncio.run(MartingaleComparison().run())
