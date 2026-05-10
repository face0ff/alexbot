
import asyncio
import ccxt
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta

# Mock or import project modules
try:
    from pattern.combined_detector import CombinedTASFibDetector
    from data.cleaner import DataCleaner
except:
    # Fallback for direct execution
    import sys
    sys.path.append(os.getcwd())
    from impulse_fib_trader.pattern.combined_detector import CombinedTASFibDetector
    from impulse_fib_trader.data.cleaner import DataCleaner

async def fetch_ohlcv(symbol, days=180):
    exchange = ccxt.binance()
    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    all_ohlcv = []
    while since < int(datetime.now().timestamp() * 1000):
        ohlcv = await asyncio.to_thread(exchange.fetch_ohlcv, symbol, '1h', since, 1000)
        if not ohlcv: break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 3600000
        if len(ohlcv) < 1000: break
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def simulate_marti_cons(df, start_idx, entry_price, sl, risk_unit):
    # Weights: 1, 1.5, 2.5
    levels = [
        {'price': entry_price, 'weight': 1, 'filled': True},
        {'price': entry_price - risk_unit * 0.45, 'weight': 1.5, 'filled': False},
        {'price': entry_price - risk_unit * 0.8, 'weight': 2.5, 'filled': False}
    ]
    
    for j in range(start_idx + 1, min(start_idx + 48, len(df))):
        low, high = df.iloc[j]['low'], df.iloc[j]['high']
        
        # Fill limits
        for lvl in levels[1:]:
            if not lvl['filled'] and low <= lvl['price']:
                lvl['filled'] = True
        
        filled = [l for l in levels if l['filled']]
        total_w = sum(l['weight'] for l in filled)
        avg_entry = sum(l['price'] * l['weight'] for l in filled) / total_w
        
        # TP logic
        if len(filled) == 1:
            tp = entry_price + risk_unit * 2.0
        else:
            tp = avg_entry + risk_unit * 0.3
            
        if low <= sl:
            return ((sl - avg_entry) / risk_unit) * total_w
        if high >= tp:
            return ((tp - avg_entry) / risk_unit) * total_w
    return 0

def simulate_smart_liquidity(df, start_idx, entry_price, sl, risk_unit):
    # Baseline entry 1
    total_w = 1
    avg_entry = entry_price
    
    # SSL Level (local low of last 48h)
    ssl_level = df.iloc[max(0, start_idx-48):start_idx]['low'].min()
    
    # Track the highest point before drop for Fib calculation
    high_point = df.iloc[max(0, start_idx-12):start_idx+1]['high'].max()
    
    manipulation_filled = False
    
    for j in range(start_idx + 1, min(start_idx + 48, len(df))):
        low, high, close, vol = df.iloc[j]['low'], df.iloc[j]['high'], df.iloc[j]['close'], df.iloc[j]['volume']
        vol_avg = df.iloc[j-20:j]['volume'].mean()
        
        # Check for manipulation near/below SSL
        if not manipulation_filled:
            is_below_ssl = low < ssl_level
            # Climax volume or Rejection wick
            is_climax = vol > vol_avg * 1.8
            is_rejection = (close - low) > (high - low) * 0.6 and (high - low) > 0
            
            if is_below_ssl and (is_climax or is_rejection):
                # Enter 4x at the close of this manipulation candle (Smart Entry)
                smart_qty = 4.0
                avg_entry = (avg_entry * total_w + close * smart_qty) / (total_w + smart_qty)
                total_w += smart_qty
                manipulation_filled = True
                # New TP: 0.5 Fib of the whole move down
                tp = (high_point + low) / 2
                # Ensure TP is at least 0.2R above avg entry
                tp = max(tp, avg_entry + risk_unit * 0.2)

        # Current TP/SL
        curr_tp = tp if manipulation_filled else (entry_price + risk_unit * 2.0)
        
        if low <= sl:
            return ((sl - avg_entry) / risk_unit) * total_w
        if high >= curr_tp:
            return ((curr_tp - avg_entry) / risk_unit) * total_w
    return 0

async def main():
    detector = CombinedTASFibDetector()
    cleaner = DataCleaner()
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'PEPE/USDT']
    
    results = {}
    
    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        try:
            df = await fetch_ohlcv(symbol, days=120)
            df = cleaner.calculate_indicators(df)
            patterns = detector.detect_patterns(df)
            
            res_marti = []
            res_smart = []
            
            for p in patterns:
                idx = p['entry_idx']
                if idx >= len(df) - 1: continue
                
                entry_p = p['entry_price']
                sl = p['sl']
                risk = entry_p - sl
                if risk <= 0: continue
                
                m_res = simulate_marti_cons(df, idx, entry_p, sl, risk)
                if m_res != 0: res_marti.append(m_res)
                
                s_res = simulate_smart_liquidity(df, idx, entry_p, sl, risk)
                if s_res != 0: res_smart.append(s_res)
            
            results[symbol] = {
                'marti': {'trades': len(res_marti), 'pnl': sum(res_marti), 'wr': len([r for r in res_marti if r > 0])/len(res_marti) if res_marti else 0},
                'smart': {'trades': len(res_smart), 'pnl': sum(res_smart), 'wr': len([r for r in res_smart if r > 0])/len(res_smart) if res_smart else 0}
            }
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")

    print("\n" + "="*70)
    print(f"{'SYMBOL':<12} | {'STRATEGY':<12} | {'TRADES':<7} | {'WINRATE':<8} | {'PNL (R)':<10}")
    print("-" * 70)
    for sym, res in results.items():
        m, s = res['marti'], res['smart']
        print(f"{sym:<12} | MARTI_CONS   | {m['trades']:<7} | {m['wr']:<8.1%} | {m['pnl']:<10.2f}")
        print(f"{sym:<12} | SMART_LIQ    | {s['trades']:<7} | {s['wr']:<8.1%} | {s['pnl']:<10.2f}")
        print("-" * 70)

if __name__ == "__main__":
    asyncio.run(main())
