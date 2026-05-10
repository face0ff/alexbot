import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.fetcher import DataFetcher
from data.cleaner import DataCleaner
from pattern.market_structure_smc import SMCMarketStructure

def simulate_smc(df_htf, df_entry, symbol):
    smc = SMCMarketStructure(window=5)
    
    # 1. HTF Bias
    df_htf = smc.detect_bos(df_htf)
    bias_map = df_htf['bos_signal'].replace(0, np.nan).ffill().reindex(df_entry.index, method='ffill')
    df_entry['htf_bias'] = bias_map
    
    # 2. Entry TF BOS
    df_entry = smc.detect_bos(df_entry)
    
    # 3. ATR
    high_low = df_entry['high'] - df_entry['low']
    high_close = np.abs(df_entry['high'] - df_entry['close'].shift())
    low_close = np.abs(df_entry['low'] - df_entry['close'].shift())
    atr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()

    trades = []
    active_trade = None
    
    for i in range(50, len(df_entry)):
        candle = df_entry.iloc[i]
        
        if active_trade:
            if active_trade['type'] == 'bullish':
                if candle['low'] <= active_trade['sl']:
                    trades.append(-1.0); active_trade = None
                elif candle['high'] >= active_trade['tp']:
                    trades.append(2.0); active_trade = None
            else:
                if candle['high'] >= active_trade['sl']:
                    trades.append(-1.0); active_trade = None
                elif candle['low'] <= active_trade['tp']:
                    trades.append(2.0); active_trade = None
            
            if active_trade and (i - active_trade['entry_idx'] > 200): # Ограничение по времени
                trades.append(0.0); active_trade = None
            continue

        bias = candle['htf_bias']
        bos = candle['bos_signal']
        
        if bias == 1 and bos == 1:
            entry_p = candle['close']
            sl_p = df_entry['swing_low'].iloc[max(0, i-30):i].min()
            if np.isnan(sl_p) or sl_p >= entry_p: sl_p = entry_p - (1.5 * atr.iloc[i])
            tp_p = entry_p + 2.0 * (entry_p - sl_p)
            if entry_p - sl_p > 0:
                active_trade = {'type': 'bullish', 'sl': sl_p, 'tp': tp_p, 'entry_idx': i}
        
        elif bias == -1 and bos == -1:
            entry_p = candle['close']
            sl_p = df_entry['swing_high'].iloc[max(0, i-30):i].max()
            if np.isnan(sl_p) or sl_p <= entry_p: sl_p = entry_p + (1.5 * atr.iloc[i])
            tp_p = entry_p - 2.0 * (sl_p - entry_p)
            if sl_p - entry_p > 0:
                active_trade = {'type': 'bearish', 'sl': sl_p, 'tp': tp_p, 'entry_idx': i}

    if not trades: return 0, 0, 0
    winrate = len([t for t in trades if t > 0]) / len(trades)
    return len(trades), winrate, sum(trades)

def main():
    symbol = 'BTC/USDT'
    start = '2024-01-01'
    end = '2026-04-06'
    fetcher = DataFetcher()
    
    print(f"--- СРАВНЕНИЕ ТАЙМФРЕЙМОВ SMC ({symbol}) ---")
    
    # Загружаем все нужные данные сразу
    df_1d = fetcher.fetch_ohlcv(symbol, '1d', start, end)
    df_4h = fetcher.fetch_ohlcv(symbol, '4h', start, end)
    df_15m = fetcher.fetch_ohlcv(symbol, '15m', start, end)
    
    # Тест 1: 1d -> 15m
    n1, wr1, r1 = simulate_smc(df_1d, df_15m.copy(), symbol)
    print(f"\n📊 [HTF: 1d -> Entry: 15m]")
    print(f"Сделок: {n1}, Winrate: {wr1:.1%}, Profit: {r1:.2f}R, Exp: {r1/n1 if n1>0 else 0:.2f}R")
    
    # Тест 2: 4h -> 15m
    n2, wr2, r2 = simulate_smc(df_4h, df_15m.copy(), symbol)
    print(f"\n📊 [HTF: 4h -> Entry: 15m]")
    print(f"Сделок: {n2}, Winrate: {wr2:.1%}, Profit: {r2:.2f}R, Exp: {r2/n2 if n2>0 else 0:.2f}R")

if __name__ == "__main__":
    main()
