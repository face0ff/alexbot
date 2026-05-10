
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
except ModuleNotFoundError:
    from impulse_fib_trader.pattern.combined_detector import CombinedTASFibDetector
    from impulse_fib_trader.data.cleaner import DataCleaner
    from impulse_fib_trader.features.engineer import FeatureEngineer

async def fetch_ohlcv_year(exchange, symbol, year=2024):
    """Скачивает данные за год."""
    start_dt = datetime(year, 1, 1)
    since = int(start_dt.timestamp() * 1000)
    all_ohlcv = []
    try:
        while True:
            ohlcv = await asyncio.to_thread(exchange.fetch_ohlcv, symbol, '1h', since, 1000)
            if not ohlcv or len(ohlcv) < 2: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 3600000
            if len(ohlcv) < 1000: break
            await asyncio.sleep(0.05) # Чуть быстрее
    except: pass
    
    if not all_ohlcv: return pd.DataFrame()
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

async def run_total_analysis():
    exchange = ccxt.binance({'enableRateLimit': True})
    MODEL_PATH = 'super_model_combined.joblib'
    
    detector = CombinedTASFibDetector()
    cleaner = DataCleaner()
    fe = FeatureEngineer()
    
    model = None
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)

    markets = await asyncio.to_thread(exchange.load_markets)
    symbols = [s for s in markets if s.endswith('/USDT') and markets[s]['active']]
    
    print(f"--- НАЧАЛО ТОТАЛЬНОГО АНАЛИЗА {len(symbols)} МОНЕТ ---")
    
    whitelist = []
    full_stats = []

    # Для ускорения берем пачками по 5 монет
    batch_size = 5
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        tasks = [fetch_ohlcv_year(exchange, s) for s in batch]
        dfs = await asyncio.gather(*tasks)
        
        for symbol, df in zip(batch, dfs):
            if df.empty or len(df) < 500: continue
            
            df = cleaner.calculate_indicators(df)
            patterns = detector.detect_patterns(df)
            
            trades = []
            for p in patterns:
                idx = p['entry_idx']
                if idx >= len(df) - 1: continue
                
                # ML
                prob = 0.5
                if model:
                    try:
                        feat = fe.extract_features([p], df.iloc[:idx+1])
                        prob = float(model.predict_proba(feat)[0][1])
                    except: continue
                
                if prob < 0.52: continue
                
                # Sim
                entry_p, sl = p['entry_price'], p['sl']
                risk = entry_p - sl
                tp = entry_p + (risk * 2.0)
                
                outcome = 0
                for j in range(idx + 1, min(idx + 49, len(df))):
                    if df.iloc[j]['low'] <= sl:
                        outcome = -1
                        break
                    if df.iloc[j]['high'] >= tp:
                        outcome = 2.0
                        break
                if outcome != 0: trades.append(outcome)

            if len(trades) >= 15:
                tr_arr = np.array(trades)
                wr = len(tr_arr[tr_arr > 0]) / len(tr_arr)
                wins = tr_arr[tr_arr > 0].sum()
                losses = abs(tr_arr[tr_arr < 0].sum())
                pf = wins / losses if losses > 0 else 10.0
                net_r = tr_arr.sum()
                
                stat = {'symbol': symbol, 'trades': len(trades), 'wr': wr, 'pf': pf, 'net_r': net_r}
                full_stats.append(stat)
                
                # Критерии для Whitelist
                if wr >= 0.48 and pf >= 1.6:
                    whitelist.append(symbol)
                    print(f"✅ {symbol:<12} | WR: {wr:.1%} | PF: {pf:.2f} | R: {net_r:.1f} -- В СПИСКЕ")
                else:
                    print(f"❌ {symbol:<12} | WR: {wr:.1%} | PF: {pf:.2f} | R: {net_r:.1f}")

    # Сохраняем результат
    with open('impulse_fib_trader/config/whitelist.json', 'w') as f:
        json.dump({'whitelist': whitelist, 'updated_at': str(datetime.now())}, f, indent=4)
    
    # Также сохраним полный отчет для анализа
    with open('total_analysis_2024.json', 'w') as f:
        json.dump(full_stats, f, indent=4)

    print(f"\n--- АНАЛИЗ ЗАВЕРШЕН ---")
    print(f"Из {len(symbols)} монет отобрано: {len(whitelist)}")
    print(f"Список сохранен в config/whitelist.json")

if __name__ == "__main__":
    asyncio.run(run_total_analysis())
