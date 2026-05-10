
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
    from reporting_plots import plot_tas_pattern
except ModuleNotFoundError:
    from impulse_fib_trader.pattern.combined_detector import CombinedTASFibDetector
    from impulse_fib_trader.data.cleaner import DataCleaner
    from impulse_fib_trader.features.engineer import FeatureEngineer
    from impulse_fib_trader.reporting_plots import plot_tas_pattern


async def fetch_ohlcv_year(exchange, symbol, year=2024):
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
            since = ohlcv[-1][0] + 3600000 # + 1 hour
            if len(ohlcv) < 1000: break
            await asyncio.sleep(0.1) # Rate limiting
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            break
            
    if not all_ohlcv: return pd.DataFrame()
    
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

async def run_massive_backtest():
    exchange = ccxt.binance({'enableRateLimit': True})
    MODEL_PATH = 'super_model_combined.joblib'
    
    detector = CombinedTASFibDetector()
    cleaner = DataCleaner()
    fe = FeatureEngineer()
    
    model = None
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ ML Модель загружена.")

    # Получаем все USDT пары
    markets = exchange.load_markets()
    symbols = [s for s in markets if s.endswith('/USDT') and markets[s]['active']]
    
    # Ограничим до 50 случайных монет для скорости в этом тесте, 
    # или можно все 400+, если у нас есть время. 
    # Но для демонстрации возьмем 100 самых популярных и случайных.
    popular = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT', 'AVAX/USDT', 'DOGE/USDT', 'DOT/USDT', 'LINK/USDT', 'MATIC/USDT', 'LTC/USDT']
    others = [s for s in symbols if s not in popular]
    random.shuffle(others)
    test_symbols = popular + others[:88] # Итого 100 монет
    
    print(f"Начинаю бэктест по {len(test_symbols)} монетам за 2024 год...")
    
    all_trade_logs = []
    summary_stats = {}

    for i, symbol in enumerate(test_symbols):
        print(f"[{i+1}/{len(test_symbols)}] Скачивание и анализ {symbol}...")
        df = await fetch_ohlcv_year(exchange, symbol, 2024)
        if df.empty or len(df) < 500: continue
        
        df = cleaner.calculate_indicators(df)
        patterns = detector.detect_patterns(df)
        
        symbol_trades = []
        for p in patterns:
            idx = p['entry_idx']
            if idx >= len(df) - 1: continue
            
            # ML Фильтр
            prob = 0.5
            if model:
                try:
                    feat = fe.extract_features([p], df.iloc[:idx+1])
                    prob = float(model.predict_proba(feat)[0][1])
                except: continue
            
            if prob < 0.52: continue
            
            # Симуляция
            entry_p = p['entry_price']
            sl = p['sl']
            risk = entry_p - sl
            tp = entry_p + (risk * 2.0)
            
            outcome = 0
            exit_price = 0
            exit_time = None
            max_p = entry_p
            curr_sl = sl
            
            # Настройки защиты
            BE_TRIGGER = 0.012
            BE_LEVEL = 0.002
            TRAILING_TRIGGER = 0.02
            TRAILING_DIST = 0.012

            for j in range(idx + 1, min(idx + 49, len(df))):
                low, high = df.iloc[j]['low'], df.iloc[j]['high']
                if high > max_p: max_p = high

                # БУ
                if (max_p / entry_p - 1) >= BE_TRIGGER:
                    be_sl = entry_p * (1 + BE_LEVEL)
                    if be_sl > curr_sl: curr_sl = be_sl
                
                # Трейлинг
                if (max_p / entry_p - 1) >= TRAILING_TRIGGER:
                    t_sl = max_p * (1 - TRAILING_DIST)
                    if t_sl > curr_sl: curr_sl = t_sl

                if low <= curr_sl:
                    outcome = (curr_sl - entry_p) / risk
                    exit_price = curr_sl
                    exit_time = df.index[j]
                    break
                if high >= tp:
                    outcome = (tp - entry_p) / risk
                    exit_price = tp
                    exit_time = df.index[j]
                    break
            
            if outcome != 0:
                trade_data = {
                    'symbol': symbol,
                    'entry_time': str(df.index[idx]),
                    'exit_time': str(exit_time),
                    'entry_price': entry_p,
                    'exit_price': exit_price,
                    'sl': sl,
                    'tp': tp,
                    'prob': prob,
                    'result_r': outcome,
                    'pattern_type': p['type']
                }
                symbol_trades.append(trade_data)
                all_trade_logs.append(trade_data)

        if symbol_trades:
            tr = np.array([t['result_r'] for t in symbol_trades])
            summary_stats[symbol] = {
                'trades': len(symbol_trades),
                'winrate': float(len(tr[tr > 0]) / len(tr)),
                'profit_r': float(tr.sum())
            }

    # Сохраняем отчет
    report = {
        'year': 2024,
        'total_coins': len(test_symbols),
        'total_trades': len(all_trade_logs),
        'summary': summary_stats,
        'all_trades': all_trade_logs
    }
    
    with open('full_backtest_report_2024.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"\n✅ Бэктест завершен. Всего сделок: {len(all_trade_logs)}")
    
    # Генерация 12 случайных графиков
    if all_trade_logs:
        os.makedirs('backtest_plots', exist_ok=True)
        winners = [t for t in all_trade_logs if t['result_r'] > 0]
        sample_size = min(12, len(winners))
        plot_samples = random.sample(winners, sample_size)
        
        print(f"Генерация {sample_size} примеров графиков...")
        for j, t in enumerate(plot_samples):
            # Нам нужен DF для отрисовки, скачаем кусочек вокруг сделки
            dt_obj = datetime.fromisoformat(t['entry_time'])
            start_plot = (dt_obj - timedelta(days=5)).timestamp() * 1000
            end_plot = (dt_obj + timedelta(days=3)).timestamp() * 1000
            
            try:
                df_plot = await asyncio.to_thread(exchange.fetch_ohlcv, t['symbol'], '1h', int(start_plot), 200)
                pdf = pd.DataFrame(df_plot, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                pdf['timestamp'] = pd.to_datetime(pdf['timestamp'], unit='ms')
                pdf.set_index('timestamp', inplace=True)
                pdf = cleaner.calculate_indicators(pdf)
                
                # Ищем индекс входа в этом кусочке
                entry_idx = pdf.index.get_indexer([pd.to_datetime(t['entry_time'])], method='nearest')[0]
                
                pattern_for_plot = {
                    'entry_idx': entry_idx,
                    'entry_price': t['entry_price'],
                    'sl': t['sl'],
                    'prob': t['prob'],
                    'p0': t['entry_price'] * 0.98, # Заглушка для уровней, если p0/p1 не сохранены
                    'p1': t['tp']
                }
                
                filename = f"backtest_plots/plot_{j+1}_{t['symbol'].replace('/', '_')}_{dt_obj.strftime('%Y%m')}.png"
                plot_tas_pattern(pdf, pattern_for_plot, t['symbol'], filename)
            except Exception as pe:
                print(f"Plot error for {t['symbol']}: {pe}")

if __name__ == "__main__":
    asyncio.run(run_massive_backtest())
