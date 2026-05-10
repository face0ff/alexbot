import pandas as pd
import joblib
import numpy as np
import mplfinance as mpf
import os
from data.fetcher import DataFetcher
from data.cleaner import DataCleaner
from pattern.combined_detector import CombinedTASFibDetector
from features.engineer import FeatureEngineer

def plot_trade(df, p, res, symbol, idx_num):
    entry_idx = p['entry_idx']
    sl = p['sl']
    tp = p['tp']
    
    # Окно графика: 30 свечей до и до момента выхода или 100 свечей после
    start_idx = max(0, entry_idx - 30)
    end_idx = min(len(df), entry_idx + 100)
    
    plot_df = df.iloc[start_idx:end_idx].copy()
    plot_df.index = pd.to_datetime(plot_df.index)
    
    # Цвета
    res_text = "WIN" if res > 0 else "LOSS" if res < 0 else "TIME_EXIT"
    title = f"{symbol} Trade {idx_num} - {res_text} (R: {res:.2f})"
    
    # Линии
    hlines = [p['entry_price'], sl, tp]
    hcolors = ['orange', 'red', 'green']
    
    output_path = f"backtest_plots_v4/{symbol.replace('/', '_')}_{idx_num}_{res_text}.png"
    
    mpf.plot(plot_df, type='candle', style='charles',
             title=title,
             ylabel='Price',
             hlines=dict(hlines=hlines, colors=hcolors, linestyle='--'),
             savefig=output_path)
    return output_path

def main():
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    model = joblib.load('super_model_v2.joblib')
    detector = CombinedTASFibDetector()
    fe = FeatureEngineer()
    fetcher = DataFetcher()
    cleaner = DataCleaner()
    
    from backtest_v2 import simulate_trade
    
    for symbol in symbols:
        print(f"Обработка {symbol}...")
        df = fetcher.fetch_ohlcv(symbol, '1h', '2024-01-01', '2026-04-06')
        df = cleaner.validate_data(df)
        df = cleaner.calculate_indicators(df)
        
        patterns = detector.detect_patterns(df)
        if not patterns: continue
        
        X = fe.extract_features(patterns, df)
        preds = model.predict(X)
        
        filtered = [p for i, p in enumerate(patterns) if preds[i] == 1]
        
        # Берем первые 5 сделок для наглядности
        for i, p in enumerate(filtered[:5]):
            res = simulate_trade(df, p)
            path = plot_trade(df, p, res, symbol, i+1)
            print(f"✅ Сохранен график: {path}")

if __name__ == "__main__":
    main()
