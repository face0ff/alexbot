
import asyncio
import ccxt
import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict

try:
    from pattern.combined_detector import CombinedTASFibDetector
    from data.cleaner import DataCleaner
except ImportError:
    from impulse_fib_trader.pattern.combined_detector import CombinedTASFibDetector
    from impulse_fib_trader.data.cleaner import DataCleaner

logger = logging.getLogger(__name__)

class WeightCalculator:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.cleaner = DataCleaner()
        self.detector = CombinedTASFibDetector()
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.WHITELIST_PATH = os.path.join(current_dir, 'config', 'whitelist.json')
        self.OUTPUT_PATH = os.path.join(current_dir, 'config', 'coin_weights_2024_2025.json')

        if not os.path.exists(self.WHITELIST_PATH):
            self.WHITELIST_PATH = os.path.join(current_dir, 'impulse_fib_trader', 'config', 'whitelist.json')
            self.OUTPUT_PATH = os.path.join(current_dir, 'impulse_fib_trader', 'config', 'coin_weights_2024_2025.json')

        with open(self.WHITELIST_PATH, 'r') as f:
            self.whitelist = json.load(f)

    async def fetch_historical_data(self, symbol, days=90):
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=days)
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
        
        if all_ohlcv:
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        return pd.DataFrame()

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
            {'price': entry_1 - risk_unit * 0.45 if is_long else entry_1 + risk_unit * 0.45, 'weight': 1.5, 'filled': False},
            {'price': entry_1 - risk_unit * 0.8 if is_long else entry_1 + risk_unit * 0.8, 'weight': 2.5, 'filled': False}
        ]
        curr_sl, max_reached = sl, entry_1
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
                    be = avg_entry * 1.002
                    if be > curr_sl: curr_sl = be
                if low <= curr_sl: return (curr_sl - avg_entry) / risk_unit * total_w
                if high >= curr_tp: return (curr_tp - avg_entry) / risk_unit * total_w
        return 0

    async def run(self):
        total_symbols = len(self.whitelist)
        print(f"🚀 Starting weights calculation for {total_symbols} coins...")
        results = {}
        detailed_stats = {}

        for i, symbol in enumerate(self.whitelist, 1):
            print(f"[{i}/{total_symbols}] Analyzing {symbol}...", end='\r')
            df = await self.fetch_historical_data(symbol)
            if df.empty or len(df) < 200: 
                results[symbol] = 0
                continue
                
            df = self.cleaner.calculate_indicators(df)
            patterns = self.detector.detect_patterns(df)
            pnl_total, trades, wins = 0, 0, 0
            for p in patterns:
                res = self.sim_martingale(df, p)
                if res != 0:
                    pnl_total += res
                    trades += 1
                    if res > 0: wins += 1
            
            results[symbol] = pnl_total
            detailed_stats[symbol] = {'pnl_r': round(pnl_total, 2), 'trades': trades, 'winrate': round(wins/trades, 2) if trades > 0 else 0}

        print(f"\n✅ Scan complete. Normalizing weights...")
        max_pnl = max(results.values()) if results else 1
        final_weights = {s: round(0.2 + (p / max_pnl) * 0.8, 2) if p > 0 else 0.1 for s, p in results.items()}
        sorted_symbols = sorted(final_weights.items(), key=lambda x: x[1], reverse=True)
        
        report_msg = "📈 <b>WEIGHTS UPDATE REPORT (90D)</b>\n──────────────────\n"
        for s, w in sorted_symbols[:5]:
            stat = detailed_stats[s]
            report_msg += f"🔸 <b>{s}</b>: W={w} | {stat['pnl_r']:+.1f}R | WR: {stat['winrate']:.0%}\n"
        
        with open(self.OUTPUT_PATH, 'w') as f:
            json.dump({'weights': dict(sorted_symbols), 'stats': detailed_stats, 'updated': str(datetime.now())}, f, indent=4)
        return report_msg

if __name__ == "__main__":
    asyncio.run(WeightCalculator().run())
