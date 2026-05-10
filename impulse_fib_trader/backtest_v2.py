import pandas as pd
import joblib
import numpy as np
from data.fetcher import DataFetcher
from data.cleaner import DataCleaner
from pattern.combined_detector import CombinedTASFibDetector
from features.engineer import FeatureEngineer

def simulate_trade(df, p, max_bars=48): # На M15 48 свечей = 12 часов
    entry_idx = p['entry_idx']
    entry_price = p['entry_price']
    sl = p['sl']
    tp = p['tp']
    
    for i in range(entry_idx + 1, min(entry_idx + max_bars + 1, len(df))):
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]
        
        if p['type'] == 'IFPC_STRICT':
            if low <= sl: return -1.0
            if high >= tp: return (tp - entry_price) / (entry_price - sl)
        else:
            if high >= sl: return -1.0
            if low <= tp: return (entry_price - tp) / (sl - entry_price)
            
    return 0 

def run_new_backtest():
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'LINK/USDT', 'MATIC/USDT', 'ADA/USDT', 'DOT/USDT']
    start_date = '2024-01-01'
    end_date = '2026-04-06'
    
    fetcher = DataFetcher()
    cleaner = DataCleaner()
    detector = CombinedTASFibDetector()
    fe = FeatureEngineer()
    model = joblib.load('super_model_v2.joblib')
    
    print(f"--- БЭКТЕСТ 2024-2026 (M15: 7 монет) ---")
    
    total_r = 0
    total_trades = 0
    wins = 0
    
    for symbol in symbols:
        try:
            df = fetcher.fetch_ohlcv(symbol, '15m', start_date, end_date)
            df = cleaner.validate_data(df)
            df = cleaner.calculate_indicators(df)
            
            patterns = detector.detect_patterns(df)
            if not patterns: continue
            
            X = fe.extract_features(patterns, df)
            predictions = model.predict(X)
            
            filtered_patterns = [p for i, p in enumerate(patterns) if predictions[i] == 1]
            
            symbol_r = 0
            symbol_wins = 0
            
            for p in filtered_patterns:
                res = simulate_trade(df, p)
                symbol_r += res
                if res > 0: symbol_wins += 1
                
            total_trades += len(filtered_patterns)
            total_r += symbol_r
            wins += symbol_wins
            
            wr = symbol_wins / len(filtered_patterns) if filtered_patterns else 0
            print(f"✅ {symbol}: Сделок: {len(filtered_patterns)}, WR: {wr:.1%}, R-Total: {symbol_r:.2f}")
            
        except Exception as e:
            print(f"❌ Ошибка {symbol}: {e}")

    if total_trades > 0:
        print(f"\n--- ИТОГО ПО ВСЕМ МОНЕТАМ (M15) ---")
        print(f"Всего сделок: {total_trades} (за 2.25 года)")
        print(f"Сделок в месяц: {total_trades / 27:.1f}")
        print(f"Общий Winrate: {wins/total_trades:.1%}")
        print(f"Общий профит (в R): {total_r:.2f}R")
        print(f"Профит в месяц (в R): {total_r / 27:.2f}R")

if __name__ == "__main__":
    run_new_backtest()
