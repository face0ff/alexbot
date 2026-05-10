
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

class FiboVsMartiComparison:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.cleaner = DataCleaner()
        self.detector = CombinedTASFibDetector()
        
        WHITELIST_PATH = 'impulse_fib_trader/config/whitelist.json'
        with open(WHITELIST_PATH, 'r') as f:
            self.whitelist = json.load(f)[:10] # Top 10 for speed

    async def fetch_data(self, symbol, years=[2024, 2025]):
        all_dfs = []
        for year in years:
            start_dt = datetime(year, 1, 1)
            end_dt = datetime(year, 12, 31, 23, 59)
            since = int(start_dt.timestamp() * 1000)
            year_ohlcv = []
            while since < int(end_dt.timestamp() * 1000):
                try:
                    ohlcv = await asyncio.to_thread(self.exchange.fetch_ohlcv, symbol, '1h', since, 1000)
                    if not ohlcv: break
                    year_ohlcv.extend(ohlcv)
                    since = ohlcv[-1][0] + 3600000
                    if len(ohlcv) < 1000: break
                    await asyncio.sleep(0.02)
                except: break
            if year_ohlcv:
                df = pd.DataFrame(year_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                all_dfs.append(df)
        return pd.concat(all_dfs).sort_index() if all_dfs else pd.DataFrame()

    def sim_fib75(self, df, p):
        """Реалистичная симуляция Fib75 с RR ~3.0 и тренд-фильтром."""
        idx = p['entry_idx']
        # Тренд-фильтр: входим только по тренду (EMA 200)
        ema_200 = df.iloc[idx]['ema_200']
        is_long = p['entry_price'] > p['sl']
        if (is_long and p['entry_price'] < ema_200) or (not is_long and p['entry_price'] > ema_200):
            return 0 # Фильтр тренда отклонил вход

        risk = abs(p['entry_price'] - p['sl'])
        reward = abs(p['tp'] - p['entry_price'])
        rr = reward / risk if risk > 0 else 0
        
        for j in range(idx + 1, min(idx + 48, len(df))):
            l, h = df.iloc[j]['low'], df.iloc[j]['high']
            if is_long:
                if l <= p['sl']: return -1.0
                if h >= p['tp']: return rr
            else:
                if h >= p['sl']: return -1.0
                if l <= p['tp']: return rr
        return 0

    def sim_martingale(self, df, p):
        """Логика Мартингейла (как в предыдущем тесте)."""
        idx = p['entry_idx']
        entry_1 = p['entry_price']
        sl = p['sl']
        risk_unit = abs(entry_1 - sl)
        if risk_unit == 0: return 0
        is_long = entry_1 > sl
        
        levels = [
            {'price': entry_1, 'weight': 1, 'filled': True},
            {'price': entry_1 - risk_unit * 0.4 if is_long else entry_1 + risk_unit * 0.4, 'weight': 2, 'filled': False},
            {'price': entry_1 - risk_unit * 0.75 if is_long else entry_1 + risk_unit * 0.75, 'weight': 4, 'filled': False}
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
            curr_tp = (entry_1 + risk_unit * 2.0 if is_long else entry_1 - risk_unit * 2.0) if len(filled) == 1 else (avg_entry + risk_unit * 0.3 if is_long else avg_entry - risk_unit * 0.3)
            
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

    async def run(self):
        print(f"🚀 Ultimate Strategy Showdown (2025)...")
        all_data = {}
        all_patterns = []
        
        # 1. Fetch data and detect patterns
        for symbol in self.whitelist:
            df = await self.fetch_data(symbol, years=[2024, 2025])
            if df.empty: continue
            df = self.cleaner.calculate_indicators(df)
            all_data[symbol] = df
            
            patterns = self.detector.detect_patterns(df)
            for p in patterns:
                p['symbol'] = symbol
                all_patterns.append(p)

        # 2. Calculate Weights for Fib75 and Martingale separately (based on 2024-2025)
        w_fib = defaultdict(float)
        w_marti = defaultdict(float)
        
        for p in all_patterns:
            # We use only 2024 to calculate weights, and then test on 2025
            if p['timestamp'].year == 2024:
                w_fib[p['symbol']] += self.sim_fib75(all_data[p['symbol']], p)
                w_marti[p['symbol']] += self.sim_martingale(all_data[p['symbol']], p)

        # 3. Final Simulation on 2025 only
        signals_2025 = [p for p in all_patterns if p['timestamp'].year == 2025]
        signals_2025.sort(key=lambda x: x['timestamp'])

        def run_sim(signals, use_weights, weight_dict, sim_func, max_concurrent=1):
            active_trades = []
            total_pnl = 0
            timeline = defaultdict(list)
            for s in signals: timeline[s['timestamp']].append(s)
            
            for t in sorted(timeline.keys()):
                # Exits
                remaining = []
                for trade in active_trades:
                    res = sim_func(all_data[trade['symbol']], trade['pattern'])
                    if res != 0: total_pnl += res
                    else: remaining.append(trade)
                active_trades = remaining
                # Entries
                avail = [s for s in timeline[t] if not any(at['symbol'] == s['symbol'] for at in active_trades)]
                if use_weights:
                    avail.sort(key=lambda x: weight_dict.get(x['symbol'], 0), reverse=True)
                for sig in avail:
                    if len(active_trades) >= max_concurrent: break
                    active_trades.append({'symbol': sig['symbol'], 'pattern': sig})
            return total_pnl

        print("\n--- RESULTS FOR 2025 (MAX_CONCURRENT = 1) ---")
        p_m_std = run_sim(signals_2025, False, {}, self.sim_martingale)
        p_m_w   = run_sim(signals_2025, True, w_marti, self.sim_martingale)
        p_f_std = run_sim(signals_2025, False, {}, self.sim_fib75)
        p_f_w   = run_sim(signals_2025, True, w_fib, self.sim_fib75)

        print(f"Martingale Standard: {p_m_std:>8.1f} R")
        print(f"Martingale Weighted: {p_m_w:>8.1f} R")
        print(f"Fib75 Standard     : {p_f_std:>8.1f} R")
        print(f"Fib75 Weighted     : {p_f_w:>8.1f} R")
        
        print("\n" + "="*50)
        best = max([('Marti Std', p_m_std), ('Marti Weighted', p_m_w), ('Fib75 Std', p_f_std), ('Fib75 Weighted', p_f_w)], key=lambda x: x[1])
        print(f"WINNER: {best[0]} with {best[1]:.1f} R")
        print("="*50)

if __name__ == "__main__":
    asyncio.run(FiboVsMartiComparison().run())
