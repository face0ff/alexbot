
import asyncio
import ccxt
import pandas as pd
import numpy as np
import os
import json
import joblib
from datetime import datetime, timedelta

try:
    from pattern.combined_detector import CombinedTASFibDetector
    from data.cleaner import DataCleaner
    from features.engineer import FeatureEngineer
except:
    import sys
    sys.path.append(os.getcwd())
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
            await asyncio.sleep(0.02)
        except: break
    if not all_ohlcv: return pd.DataFrame()
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def simulate_grid_with_filter(is_long, entry_1, sl, tp, risk_unit, df_slice, weights, dist_factors, use_smart_filter=False):
    levels = []
    for i in range(len(weights)):
        price = entry_1 - risk_unit * dist_factors[i] if is_long else entry_1 + abs(risk_unit) * dist_factors[i]
        levels.append({'price': price, 'weight': weights[i], 'filled': i == 0}) # Level 0 always filled
    
    # Technical levels for Smart Filter
    # In a real bot we'd use rolling data, here we use slice context
    high_point = df_slice.iloc[0]['high'] # Approximate
    
    for i in range(len(df_slice)):
        row = df_slice.iloc[i]
        low, high, close, vol = row['low'], row['high'], row['close'], row['volume']
        
        # Grid Filling Logic
        for idx, lvl in enumerate(levels):
            if not lvl['filled']:
                # Basic condition: Price hit level
                hit_price = (is_long and low <= lvl['price']) or (not is_long and high >= lvl['price'])
                
                if hit_price:
                    if use_smart_filter and idx == len(levels) - 1: # Only for last (largest) level
                        # Smart Filter: Manipulation detection
                        vol_avg = df_slice.iloc[max(0, i-5):i+1]['volume'].mean()
                        is_climax = vol > vol_avg * 1.5
                        is_rejection = (close - low) > (high - low) * 0.6 if is_long else (high - close) > (high - low) * 0.6
                        
                        if is_climax or is_rejection:
                            lvl['filled'] = True
                    else:
                        # No filter or not the last level
                        lvl['filled'] = True
        
        filled_lvls = [l for l in levels if l['filled']]
        total_w = sum(l['weight'] for l in filled_lvls)
        avg_entry = sum(l['price'] * l['weight'] for l in filled_lvls) / total_w
        
        # Dynamic TP
        if len(filled_lvls) == 1:
            curr_tp = tp
        else:
            # Smart Exit: If filtered and 3rd fill, target 0.5 Fib of recent drop
            if use_smart_filter and len(filled_lvls) == len(weights):
                curr_tp = (high_point + low) / 2 if is_long else (high_point + high) / 2
                # Safety: ensure TP is in profit
                if is_long: curr_tp = max(curr_tp, avg_entry + abs(risk_unit) * 0.2)
                else: curr_tp = min(curr_tp, avg_entry - abs(risk_unit) * 0.2)
            else:
                curr_tp = avg_entry + abs(risk_unit) * 0.3 if is_long else avg_entry - abs(risk_unit) * 0.3
            
        if is_long:
            if low <= sl: return ((sl - avg_entry) / abs(risk_unit)) * total_w
            if high >= curr_tp: return ((curr_tp - avg_entry) / abs(risk_unit)) * total_w
        else:
            if high >= sl: return ((avg_entry - sl) / abs(risk_unit)) * total_w
            if low <= curr_tp: return ((avg_entry - curr_tp) / abs(risk_unit)) * total_w
            
    return 0

async def run_comparison():
    exchange = ccxt.binance({'enableRateLimit': True})
    detector = CombinedTASFibDetector()
    cleaner = DataCleaner()
    fe = FeatureEngineer()
    
    MODEL_PATH = 'super_model_combined.joblib'
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'PEPE/USDT', 'AVAX/USDT', 'XRP/USDT', 'LTC/USDT']
    
    base_strategies = {
        'MARTI_CONS': {'weights': [1, 1.5, 2.5], 'dists': [0, 0.45, 0.8]},
        'MARTI_AGGR': {'weights': [1, 2, 4],    'dists': [0, 0.4, 0.75]},
        'FIB_GRID':  {'weights': [1, 2, 3],    'dists': [0, 0.38, 0.62]}
    }

    results = []

    print(f"🔬 Сравнение стратегий с фильтром Smart Liquidity (2025)...")

    for symbol in symbols:
        df = await fetch_ohlcv_year(exchange, symbol, 2025)
        if df.empty or len(df) < 500: continue
        df = cleaner.calculate_indicators(df)
        patterns = detector.detect_patterns(df)
        
        for p in patterns:
            idx = p['entry_idx']
            if idx >= len(df) - 1: continue
            
            # ML Filter pre-check
            if model:
                try:
                    feat = fe.extract_features([p], df.iloc[:idx+1])
                    if model.predict_proba(feat)[0][1] < 0.52: continue
                except: continue

            entry_1, sl = p['entry_price'], p['sl']
            risk_unit = entry_1 - sl
            if risk_unit == 0: continue
            tp = entry_1 + (risk_unit * 2.0)
            is_long = entry_1 > sl
            df_slice = df.iloc[idx+1 : idx+60]

            for name, config in base_strategies.items():
                # 1. Without Filter
                res_raw = simulate_grid_with_filter(is_long, entry_1, sl, tp, risk_unit, df_slice, config['weights'], config['dists'], False)
                if res_raw != 0:
                    results.append({'symbol': symbol, 'strategy': name, 'smart': False, 'pnl': res_raw})
                
                # 2. With Smart Filter
                res_smart = simulate_grid_with_filter(is_long, entry_1, sl, tp, risk_unit, df_slice, config['weights'], config['dists'], True)
                if res_smart != 0:
                    results.append({'symbol': symbol, 'strategy': name, 'smart': True, 'pnl': res_smart})

    df_res = pd.DataFrame(results)
    
    print("\n" + "="*85)
    print(f"{'STRATEGY':<15} | {'SMART':<6} | {'WR':<7} | {'Total R':<10} | {'Avg R':<7} | {'Max Loss':<8}")
    print("-" * 85)
    
    for name in base_strategies.keys():
        for smart in [False, True]:
            sub = df_res[(df_res['strategy'] == name) & (df_res['smart'] == smart)]
            if sub.empty: continue
            
            wr = len(sub[sub['pnl'] > 0]) / len(sub)
            total_r = sub['pnl'].sum()
            avg_r = sub['pnl'].mean()
            max_loss = sub['pnl'].min()
            
            label = "YES" if smart else "NO"
            print(f"{name:<15} | {label:<6} | {wr:<7.1%} | {total_r:<10.2f} | {avg_r:<7.3f} | {max_loss:<8.2f}")
    print("="*85)

if __name__ == "__main__":
    asyncio.run(run_comparison())
