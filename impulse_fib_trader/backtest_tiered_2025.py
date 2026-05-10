
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
    """Скачивает данные H1 за указанный год."""
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

async def run_tiered_backtest():
    exchange = ccxt.binance({'enableRateLimit': True})
    MODEL_PATH = 'super_model_combined.joblib'
    
    detector = CombinedTASFibDetector()
    cleaner = DataCleaner()
    fe = FeatureEngineer()
    
    model = None
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)

    markets = await asyncio.to_thread(exchange.load_markets)
    all_symbols = [s for s in markets if s.endswith('/USDT') and markets[s]['active']]
    popular = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
    test_symbols = popular + random.sample([s for s in all_symbols if s not in popular], 5)
    
    print(f"🚀 Запуск TIERED (Сеткой) бектеста за 2025 год по {len(test_symbols)} монетам...")
    
    stats = {
        'single': {'pnl_r': 0, 'trades': 0, 'wins': 0},
        'tiered': {'pnl_r': 0, 'trades': 0, 'wins': 0, 'partial_fills': 0, 'full_fills': 0}
    }

    for symbol in test_symbols:
        df = await fetch_ohlcv_year(exchange, symbol, 2025)
        if df.empty or len(df) < 500: continue
        
        df = cleaner.calculate_indicators(df)
        patterns = detector.detect_patterns(df)
        
        for p in patterns:
            idx = p['entry_idx']
            if idx >= len(df) - 1: continue
            
            # ML Filter
            if model:
                try:
                    feat = fe.extract_features([p], df.iloc[:idx+1])
                    if model.predict_proba(feat)[0][1] < 0.52: continue
                except: continue

            # --- Общие параметры ---
            entry_1 = p['entry_price']
            sl = p['sl']
            risk_unit = entry_1 - sl # Risk of Entry 1
            if risk_unit == 0: continue
            
            tp = entry_1 + (risk_unit * 2.0)
            is_long = entry_1 > sl
            
            # --- SINGLE ENTRY SIM ---
            res_single = 0
            for j in range(idx + 1, min(idx + 49, len(df))):
                low, high = df.iloc[j]['low'], df.iloc[j]['high']
                if is_long:
                    if low <= sl: res_single = -1; break
                    if high >= tp: res_single = 2; break
                else: # Short (for completeness)
                    if high >= sl: res_single = -1; break
                    if low <= tp: res_single = 2; break
            
            if res_single != 0:
                stats['single']['trades'] += 1
                stats['single']['pnl_r'] += res_single
                if res_single > 0: stats['single']['wins'] += 1

            # --- MARTINGALE GRID SIM (Aggressive) ---
            # Weights: 1, 2, 4 (Total 7 units)
            # TP: Dynamic - aims for a small profit (e.g., 0.2R of the initial risk) from average entry
            m_levels = [
                {'price': entry_1, 'weight': 1, 'filled': False},
                {'price': entry_1 - risk_unit * 0.4, 'weight': 2, 'filled': False},
                {'price': entry_1 - risk_unit * 0.75, 'weight': 4, 'filled': False}
            ] if is_long else [
                {'price': entry_1, 'weight': 1, 'filled': False},
                {'price': entry_1 + abs(risk_unit) * 0.4, 'weight': 2, 'filled': False},
                {'price': entry_1 + abs(risk_unit) * 0.75, 'weight': 4, 'filled': False}
            ]
            
            res_martingale = 0
            m_current_sl = sl
            
            for j in range(idx + 1, min(idx + 49, len(df))):
                low, high = df.iloc[j]['low'], df.iloc[j]['high']
                
                # Update fills
                new_fill = False
                for lvl in m_levels:
                    if not lvl['filled']:
                        if (is_long and low <= lvl['price']) or (not is_long and high >= lvl['price']):
                            lvl['filled'] = True
                            new_fill = True
                
                # Recalculate Average Entry and Dynamic TP if new fill occurred
                filled_lvls = [l for l in m_levels if l['filled']]
                if not filled_lvls: continue
                
                total_w = sum(l['weight'] for l in filled_lvls)
                avg_entry = sum(l['price'] * l['weight'] for l in filled_lvls) / total_w
                
                # Dynamic TP: Average Entry + small offset (e.g., 0.3 of initial risk_unit)
                # This makes it very easy to exit in profit after averaging
                m_tp = avg_entry + abs(risk_unit) * 0.3 if is_long else avg_entry - abs(risk_unit) * 0.3
                
                # If only 1st level filled, keep original TP for better gains
                if len(filled_lvls) == 1:
                    m_tp = tp

                if is_long:
                    if low <= m_current_sl:
                        res_martingale = ((m_current_sl - avg_entry) / abs(risk_unit)) * (total_w / 1.0) # Normalized to initial unit
                        break
                    if high >= m_tp:
                        res_martingale = ((m_tp - avg_entry) / abs(risk_unit)) * (total_w / 1.0)
                        break
                else:
                    if high >= m_current_sl:
                        res_martingale = ((avg_entry - m_current_sl) / abs(risk_unit)) * (total_w / 1.0)
                        break
                    if low <= m_tp:
                        res_martingale = ((avg_entry - m_tp) / abs(risk_unit)) * (total_w / 1.0)
                        break

            if res_martingale != 0:
                if 'martingale' not in stats: stats['martingale'] = {'pnl_r': 0, 'trades': 0, 'wins': 0}
                stats['martingale']['trades'] += 1
                stats['martingale']['pnl_r'] += res_martingale
                if res_martingale > 0: stats['martingale']['wins'] += 1

    print("\n--- СРАВНЕНИЕ РЕЗУЛЬТАТОВ (2025) ---")
    
    for mode in ['single', 'martingale']:
        s = stats.get(mode, {})
        tr = s.get('trades', 0)
        if tr > 0:
            wr = s['wins'] / tr
            avg_r = s['pnl_r'] / tr
            print(f"{mode.upper():<10}: Trades: {tr:<4} | WR: {wr:.1%} | Total R: {s['pnl_r']:>8.2f} | Avg R: {avg_r:.3f}")
        else:
            print(f"{mode.upper():<10}: No trades.")

    if 'single' in stats and 'martingale' in stats:
        diff = stats['martingale']['pnl_r'] - stats['single']['pnl_r']
        print(f"\nРазница (MARTINGALE - SINGLE): {diff:+.2f} R")

if __name__ == "__main__":
    asyncio.run(run_tiered_backtest())
