import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from data.fetcher import DataFetcher
from data.cleaner import DataCleaner
from pattern.market_structure_smc import SMCMarketStructure
from features.engineer_smc import SMCFeatureEngineer

def run_final_backtest():
    symbol = 'BTC/USDT'
    start_date = '2024-01-01'
    end_date = '2026-04-06'
    
    fetcher = DataFetcher()
    cleaner = DataCleaner()
    smc = SMCMarketStructure(window=5)
    fe = SMCFeatureEngineer()
    model = joblib.load('smc_model_v1.joblib')
    
    print(f"--- FINAL SMC + ML BACKTEST (1d Bias -> 15m BOS) ---")
    
    df_1d = fetcher.fetch_ohlcv(symbol, '1d', start_date, end_date)
    df_15m = fetcher.fetch_ohlcv(symbol, '15m', start_date, end_date)
    
    # 1. Bias
    df_1d = smc.detect_bos(df_1d)
    bias_map = df_1d['bos_signal'].replace(0, np.nan).ffill().reindex(df_15m.index, method='ffill')
    df_15m['htf_bias'] = bias_map
    
    # 2. 15m Structure
    df_15m = cleaner.calculate_indicators(df_15m)
    df_15m = smc.detect_bos(df_15m)
    
    # 3. Features for ML
    bos_df = df_15m[df_15m['bos_signal'] != 0].copy()
    if bos_df.empty: return
    
    X = fe.extract_features(df_15m)
    preds = model.predict(X)
    probs = model.predict_proba(X)
    
    # 4. Simulation
    trades = []
    active_trade = None
    
    for i in range(50, len(df_15m)):
        candle = df_15m.iloc[i]
        idx = df_15m.index[i]
        
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
            continue

        if idx in X.index:
            p_idx = list(X.index).index(idx)
            if preds[p_idx] == 1: # ML одобрила сделку
                bias = candle['htf_bias']
                bos = candle['bos_signal']
                
                # LONG: Bias + BOS + ML
                if bias == 1 and bos == 1:
                    sl = df_15m['swing_low'].iloc[max(0, i-30):i].min()
                    if np.isnan(sl) or sl >= candle['close']: sl = candle['close'] * 0.99
                    tp = candle['close'] + 2.0 * (candle['close'] - sl)
                    active_trade = {'type': 'bullish', 'sl': sl, 'tp': tp}
                
                # SHORT
                elif bias == -1 and bos == -1:
                    sl = df_15m['swing_high'].iloc[max(0, i-30):i].max()
                    if np.isnan(sl) or sl <= candle['close']: sl = candle['close'] * 1.01
                    tp = candle['close'] - 2.0 * (sl - candle['close'])
                    active_trade = {'type': 'bearish', 'sl': sl, 'tp': tp}

    if trades:
        winrate = len([t for t in trades if t > 0]) / len(trades)
        total_r = sum(trades)
        print(f"🚀 Сделок: {len(trades)}")
        print(f"🔥 Winrate: {winrate:.1%}")
        print(f"💰 Профит: {total_r:.2f}R")
        print(f"📈 Мат. ожидание: {total_r/len(trades):.2f}R")

if __name__ == "__main__":
    run_final_backtest()
