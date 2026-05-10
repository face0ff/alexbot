
import asyncio
import ccxt
import pandas as pd
import numpy as np
import os
import json
import joblib
from datetime import datetime
from typing import List, Dict
from collections import defaultdict

# Import existing modules
try:
    from pattern.combined_detector import CombinedTASFibDetector
    from pattern.tas_detector import ImpulseRejectionDetector as TASDetector
    from pattern.fib75_detector import ImpulseFib75Detector
    from pattern.market_structure_smc import SMCMarketStructure
    from data.cleaner import DataCleaner
except ImportError:
    from impulse_fib_trader.pattern.combined_detector import CombinedTASFibDetector
    from impulse_fib_trader.pattern.tas_detector import ImpulseRejectionDetector as TASDetector
    from impulse_fib_trader.pattern.fib75_detector import ImpulseFib75Detector
    from impulse_fib_trader.pattern.market_structure_smc import SMCMarketStructure
    from impulse_fib_trader.data.cleaner import DataCleaner

class FullMarketBacktest2025:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.cleaner = DataCleaner()
        self.ifpc_detector = CombinedTASFibDetector()
        self.tas_detector = TASDetector(config={})
        self.fib75_detector = ImpulseFib75Detector()
        self.smc = SMCMarketStructure(window=3)
        
        WHITELIST_PATH = 'impulse_fib_trader/config/whitelist.json'
        with open(WHITELIST_PATH, 'r') as f:
            self.whitelist = json.load(f)

    async def fetch_2025_data(self, symbol: str):
        # Fetch H1 data for 2025
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

    def simulate_trade(self, df, entry_idx, entry_price, sl, tp, side='long'):
        risk = abs(entry_price - sl)
        if risk == 0: return 0
        
        # Limit to 48 candles
        end_idx = min(entry_idx + 48, len(df) - 1)
        for i in range(entry_idx + 1, end_idx + 1):
            low, high = df.iloc[i]['low'], df.iloc[i]['high']
            if side == 'long':
                if low <= sl: return -1.0
                if high >= tp: return 2.0
            else:
                if high >= sl: return -1.0
                if low <= tp: return 2.0
        return 0

    async def run(self):
        print(f"🚀 Starting Full 2025 Backtest for {len(self.whitelist)} coins...")
        results = defaultdict(lambda: defaultdict(float))
        trade_counts = defaultdict(lambda: defaultdict(int))
        
        # For display
        print(f"{'Symbol':<12} | {'IFPC (R)':<8} | {'Fib75 (R)':<8} | {'SMC (R)':<8} | {'TAS (R)':<8}")
        print("-" * 65)

        for symbol in self.whitelist:
            df = await self.fetch_2025_data(symbol)
            if df.empty or len(df) < 200: continue
            
            df = self.cleaner.calculate_indicators(df)
            
            # 1. IFPC
            ifpc_p = self.ifpc_detector.detect_patterns(df)
            for p in ifpc_p:
                res = self.simulate_trade(df, p['entry_idx'], p['entry_price'], p['sl'], p['tp'])
                results[symbol]['IFPC'] += res
                trade_counts[symbol]['IFPC'] += 1 if res != 0 else 0

            # 2. Fib75
            fib_p = self.fib75_detector.detect_patterns(df)
            for p in fib_p:
                res = self.simulate_trade(df, p['entry_idx'], p['entry_price'], p['sl'], p['tp'])
                results[symbol]['Fib75'] += res
                trade_counts[symbol]['Fib75'] += 1 if res != 0 else 0

            # 3. TAS
            tas_p = self.tas_detector.detect_patterns(df)
            for p in tas_p:
                # Standard TAS SL is candle low, TP set to RR 2.0
                sl = p['sl']
                tp = p['entry_price'] + (p['entry_price'] - sl) * 2.0
                res = self.simulate_trade(df, p['entry_idx'], p['entry_price'], sl, tp)
                results[symbol]['TAS'] += res
                trade_counts[symbol]['TAS'] += 1 if res != 0 else 0

            # 4. SMC
            df_smc = self.smc.detect_bos(df.copy())
            for i in range(len(df_smc)):
                if df_smc['bos_signal'].iloc[i] == 1:
                    price = df_smc['close'].iloc[i]
                    sl = df_smc['low'].iloc[max(0, i-5):i].min()
                    if np.isnan(sl) or price == sl: continue
                    tp = price + (price - sl) * 2.0
                    res = self.simulate_trade(df, i, price, sl, tp)
                    results[symbol]['SMC'] += res
                    trade_counts[symbol]['SMC'] += 1 if res != 0 else 0
            
            print(f"{symbol:<12} | {results[symbol]['IFPC']:>8.1f} | {results[symbol]['Fib75']:>8.1f} | {results[symbol]['SMC']:>8.1f} | {results[symbol]['TAS']:>8.1f}")

        # Final Totals
        total_ifpc = sum(results[s]['IFPC'] for s in results)
        total_fib = sum(results[s]['Fib75'] for s in results)
        total_smc = sum(results[s]['SMC'] for s in results)
        total_tas = sum(results[s]['TAS'] for s in results)
        
        print("-" * 65)
        print(f"{'TOTAL':<12} | {total_ifpc:>8.1f} | {total_fib:>8.1f} | {total_smc:>8.1f} | {total_tas:>8.1f}")
        
        # Save detailed stats
        final_report = {
            'year': 2025,
            'symbols': {s: dict(results[s]) for s in results},
            'totals': {
                'IFPC': total_ifpc,
                'Fib75': total_fib,
                'SMC': total_smc,
                'TAS': total_tas
            }
        }
        with open('total_analysis_2025.json', 'w') as f:
            json.dump(final_report, f, indent=4)

if __name__ == "__main__":
    asyncio.run(FullMarketBacktest2025().run())
