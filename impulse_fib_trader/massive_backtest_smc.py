import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime, timedelta
from data.fetcher import DataFetcher
from data.cleaner import DataCleaner
from pattern.market_structure_smc import SMCMarketStructure
from features.engineer_smc import SMCFeatureEngineer

# Отключаем лишние логи
logging.getLogger('ccxt').setLevel(logging.WARNING)

def run_massive_backtest():
    fetcher = DataFetcher()
    cleaner = DataCleaner()
    smc = SMCMarketStructure(window=5)
    fe = SMCFeatureEngineer()
    
    # Загружаем модель
    try:
        model = joblib.load('smc_model_v1.joblib')
        print("✅ Модель SMC v1 загружена.")
    except:
        print("❌ Модель не найдена!")
        return

    # 1. Получаем список Топ-50 USDT пар
    all_symbols = fetcher.get_active_symbols()
    usdt_pairs = [s for s in all_symbols if s.endswith('/USDT')][:50]
    
    start_date = '2024-01-01'
    end_date = '2026-04-06'
    
    print(f"--- MASSIVE SMC BACKTEST (50 COINS) ---")
    
    total_trades_all = []
    symbol_results = []

    for symbol in usdt_pairs:
        try:
            # Загружаем данные (Bias на 1d, входы на 15m)
            df_1d = fetcher.fetch_ohlcv(symbol, '1d', start_date, end_date)
            df_15m = fetcher.fetch_ohlcv(symbol, '15m', start_date, end_date)
            
            if df_1d.empty or len(df_15m) < 200: continue
            
            # 2. Bias & Structure
            df_1d = smc.detect_bos(df_1d)
            bias_map = df_1d['bos_signal'].replace(0, np.nan).ffill().reindex(df_15m.index, method='ffill')
            df_15m['htf_bias'] = bias_map
            df_15m = cleaner.calculate_indicators(df_15m)
            df_15m = smc.detect_bos(df_15m)
            
            # 3. Features & ML
            X = fe.extract_features(df_15m)
            if X.empty: continue
            
            preds = model.predict(X)
            
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
                    if preds[p_idx] == 1:
                        bias = candle['htf_bias']
                        bos = candle['bos_signal']
                        if bias == 1 and bos == 1:
                            sl = df_15m['swing_low'].iloc[max(0, i-20):i].min()
                            if np.isnan(sl) or sl >= candle['close']: sl = candle['close'] * 0.99
                            tp = candle['close'] + 2.0 * (candle['close'] - sl)
                            active_trade = {'type': 'bullish', 'sl': sl, 'tp': tp}
                        elif bias == -1 and bos == -1:
                            sl = df_15m['swing_high'].iloc[max(0, i-20):i].max()
                            if np.isnan(sl) or sl <= candle['close']: sl = candle['close'] * 1.01
                            tp = candle['close'] - 2.0 * (sl - candle['close'])
                            active_trade = {'type': 'bearish', 'sl': sl, 'tp': tp}

            if trades:
                wr = len([t for t in trades if t > 0]) / len(trades)
                pnl = sum(trades)
                symbol_results.append({'symbol': symbol, 'trades': len(trades), 'wr': wr, 'pnl': pnl})
                total_trades_all.extend(trades)
                print(f"✅ {symbol}: {len(trades)} trades, PnL: {pnl:.1f}R, WR: {wr:.1%}")
                
        except Exception as e:
            continue

    if total_trades_all:
        final_wr = len([t for t in total_trades_all if t > 0]) / len(total_trades_all)
        final_pnl = sum(total_trades_all)
        print(f"\n" + "="*50)
        print(f"ИТОГО ПО 50 МОНЕТАМ:")
        print(f"Всего сделок: {len(total_trades_all)}")
        print(f"Общий Winrate: {final_wr:.1%}")
        print(f"Общий Профит: {final_pnl:.2f}R")
        print(f"Ср. профит на монету: {final_pnl/len(symbol_results):.2f}R")
        print("="*50)

if __name__ == "__main__":
    run_massive_backtest()
