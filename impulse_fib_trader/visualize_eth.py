import pandas as pd
import joblib
import matplotlib.pyplot as plt
import mplfinance as mpf
from data.fetcher import DataFetcher
from data.cleaner import DataCleaner
from pattern.combined_detector import CombinedTASFibDetector

def plot_eth_failure():
    symbol = 'ETH/USDT'
    start_date = '2025-01-01'
    end_date = '2026-04-06'
    
    fetcher = DataFetcher()
    cleaner = DataCleaner()
    detector = CombinedTASFibDetector()
    
    df = fetcher.fetch_ohlcv(symbol, '1h', start_date, end_date)
    df = cleaner.validate_data(df)
    df = cleaner.calculate_indicators(df)
    
    patterns = detector.detect_patterns(df)
    if not patterns:
        print("Паттерны не найдены")
        return
    
    # Берем самый свежий провальный паттерн (из последних 5)
    p = patterns[-2] 
    idx = p['entry_idx']
    
    start_idx = max(0, idx - 50)
    end_idx = min(len(df), idx + 100)
    
    plot_df = df.iloc[start_idx:end_idx].copy()
    
    # Добавляем уровни на график
    lines = [
        [(df.index[p['entry_idx']], p['entry_price']), (df.index[end_idx-1], p['entry_price'])],
        [(df.index[p['entry_idx']], p['sl']), (df.index[end_idx-1], p['sl'])],
        [(df.index[p['entry_idx']], p['tp']), (df.index[end_idx-1], p['tp'])]
    ]
    
    print(f"Отрисовка сделки по {symbol} от {p['timestamp']}")
    print(f"Вход: {p['entry_price']}, SL: {p['sl']}, TP: {p['tp']}")

    mpf.plot(plot_df, type='candle', style='charles',
             title=f"ETH Failure Analysis - {p['timestamp']}",
             ylabel='Price',
             hlines=dict(hlines=[p['entry_price'], p['sl'], p['tp']], 
                         colors=['orange', 'red', 'green'], 
                         linestyle='--'),
             savefig='eth_failure_plot.png')
    print("✅ График сохранен как eth_failure_plot.png")

if __name__ == "__main__":
    plot_eth_failure()
