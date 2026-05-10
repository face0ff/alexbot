
import asyncio
import ccxt
import pandas as pd
import numpy as np
import os
import json
import joblib
import random
from datetime import datetime, timedelta
try:
    from pattern.combined_detector import CombinedTASFibDetector
    from data.cleaner import DataCleaner
    from features.engineer import FeatureEngineer
except ModuleNotFoundError:
    from impulse_fib_trader.pattern.combined_detector import CombinedTASFibDetector
    from impulse_fib_trader.data.cleaner import DataCleaner
    from impulse_fib_trader.features.engineer import FeatureEngineer

async def fetch_ohlcv_year(exchange, symbol, year=2025):
    start_dt = datetime(year, 1, 1)
    end_dt = datetime(year, 12, 31, 23, 59)
    since = int(start_dt.timestamp() * 1000)
    all_ohlcv = []
    while since < int(end_dt.timestamp() * 1000):
        try:
            ohlcv = await asyncio.to_thread(exchange.fetch_ohlcv, symbol, '1h', since, 1000)
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

def simulate_grid(is_long, entry_1, sl, tp, risk_unit, df_slice, weights, dist_factors):
    """
    Simulates a grid strategy with given weights and distance factors.
    dist_factors: [0, factor_2, factor_3] where factor is multiplier of risk_unit
    """
    levels = []
    for i in range(len(weights)):
        price = entry_1 - risk_unit * dist_factors[i] if is_long else entry_1 + abs(risk_unit) * dist_factors[i]
        levels.append({'price': price, 'weight': weights[i], 'filled': False})
    
    res = 0
    current_sl = sl
    
    for i in range(len(df_slice)):
        low, high = df_slice.iloc[i]['low'], df_slice.iloc[i]['high']
        
        for lvl in levels:
            if not lvl['filled']:
                if (is_long and low <= lvl['price']) or (not is_long and high >= lvl['price']):
                    lvl['filled'] = True
        
        filled_lvls = [l for l in levels if l['filled']]
        if not filled_lvls: continue
        
        total_w = sum(l['weight'] for l in filled_lvls)
        avg_entry = sum(l['price'] * l['weight'] for l in filled_lvls) / total_w
        
        # Dynamic TP logic:
        # If 1 lvl: use original TP
        # If >1 lvl: use avg_entry + small profit (0.3 of risk)
        if len(filled_lvls) == 1:
            curr_tp = tp
        else:
            curr_tp = avg_entry + abs(risk_unit) * 0.3 if is_long else avg_entry - abs(risk_unit) * 0.3
            
        if is_long:
            if low <= current_sl:
                return ((current_sl - avg_entry) / abs(risk_unit)) * total_w
            if high >= curr_tp:
                return ((curr_tp - avg_entry) / abs(risk_unit)) * total_w
        else:
            if high >= current_sl:
                return ((avg_entry - current_sl) / abs(risk_unit)) * total_w
            if low <= curr_tp:
                return ((avg_entry - curr_tp) / abs(risk_unit)) * total_w
    return 0

async def run_optimization():
    exchange = ccxt.binance({'enableRateLimit': True})
    MODEL_PATH = 'super_model_combined.joblib'
    detector = CombinedTASFibDetector()
    cleaner = DataCleaner()
    fe = FeatureEngineer()
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

    markets = await asyncio.to_thread(exchange.load_markets)
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 'AVAX/USDT', 'LINK/USDT', 'LTC/USDT']
    
    strategies = {
        'single':   {'weights': [1],          'dists': [0],          'stats': []},
        'marti_aggr': {'weights': [1, 2, 4],    'dists': [0, 0.4, 0.75], 'stats': []},
        'marti_cons': {'weights': [1, 1.5, 2.5], 'dists': [0, 0.45, 0.8], 'stats': []},
        'fib_grid':  {'weights': [1, 2, 3],    'dists': [0, 0.38, 0.62], 'stats': []}
    }

    print(f"🔬 Оптимизация стратегий сетки по {len(symbols)} монетам (2025)...")

    for symbol in symbols:
        df = await fetch_ohlcv_year(exchange, symbol, 2025)
        if df.empty or len(df) < 500: continue
        df = cleaner.calculate_indicators(df)
        patterns = detector.detect_patterns(df)
        
        for p in patterns:
            idx = p['entry_idx']
            if idx >= len(df) - 1: continue
            if model:
                try:
                    feat = fe.extract_features([p], df.iloc[:idx+1])
                    if model.predict_proba(feat)[0][1] < 0.52: continue
                except: continue

            entry_1 = p['entry_price']
            sl = p['sl']
            risk_unit = entry_1 - sl
            if risk_unit == 0: continue
            tp = entry_1 + (risk_unit * 2.0)
            is_long = entry_1 > sl
            df_slice = df.iloc[idx+1 : idx+50]

            for name, config in strategies.items():
                res = simulate_grid(is_long, entry_1, sl, tp, risk_unit, df_slice, config['weights'], config['dists'])
                if res != 0:
                    config['stats'].append(res)

    print("\n" + "="*80)
    print(f"{'STRATEGY':<12} | {'WR':<7} | {'Total R':<10} | {'Avg R':<7} | {'Max Loss':<8} | {'Max Drawdown (R)':<10}")
    print("-" * 80)
    
    for name, config in strategies.items():
        res_arr = np.array(config['stats'])
        if len(res_arr) == 0: continue
        
        wr = len(res_arr[res_arr > 0]) / len(res_arr)
        total_r = res_arr.sum()
        avg_r = res_arr.mean()
        max_loss = res_arr.min()
        
        # Max DD calculation (simplified on R sequence)
        cum_r = np.cumsum(res_arr)
        peak = np.maximum.accumulate(cum_r)
        drawdown = peak - cum_r
        max_dd = np.max(drawdown)

        print(f"{name.upper():<12} | {wr:<7.1%} | {total_r:<10.2f} | {avg_r:<7.3f} | {max_loss:<8.2f} | {max_dd:<10.2f}")

if __name__ == "__main__":
    asyncio.run(run_optimization())
