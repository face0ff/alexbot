import torch
import pandas as pd
import numpy as np
import os
from data.fetcher import DataFetcher
from data.cleaner import DataCleaner
from pattern.market_structure_smc import SMCMarketStructure
from ml.transformer_smc import SMCTransformer

def run_transformer_backtest():
    fetcher = DataFetcher()
    cleaner = DataCleaner()
    smc = SMCMarketStructure(window=5)
    
    model = SMCTransformer(input_dim=4, d_model=64, nhead=4, num_layers=3)
    model.load_state_dict(torch.load('smc_transformer_v1.pth'))
    model.eval()
    
    # Берем ТОП-10 самых ликвидных монет для скорости
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'LINK/USDT', 'ADA/USDT', 'DOT/USDT', 'XRP/USDT', 'LTC/USDT', 'MATIC/USDT']
    
    start_date = '2024-01-01'
    end_date = '2026-04-06'
    
    print(f"--- TRANSFORMER BACKTEST (TOP 10 COINS) ---")
    
    total_trades_all = []
    
    for symbol in symbols:
        try:
            df_1d = fetcher.fetch_ohlcv(symbol, '1d', start_date, end_date)
            df_15m = fetcher.fetch_ohlcv(symbol, '15m', start_date, end_date)
            if df_1d.empty or len(df_15m) < 200: continue
            
            df_1d = smc.detect_bos(df_1d)
            bias_map = df_1d['bos_signal'].replace(0, np.nan).ffill().reindex(df_15m.index, method='ffill')
            df_15m['htf_bias'] = bias_map
            df_15m = cleaner.calculate_indicators(df_15m)
            df_15m = smc.detect_bos(df_15m)
            
            ohlc = df_15m[['open', 'high', 'low', 'close']].values
            atr = df_15m['atr'].values.reshape(-1, 1)
            atr[atr == 0] = np.nanmean(atr)
            norm_data = np.nan_to_num((ohlc[1:] - ohlc[:-1]) / (atr[1:] + 1e-9))
            norm_data = np.clip(norm_data, -5.0, 5.0)
            
            trades = []
            active_trade = None
            seq_len = 50
            bos_indices = df_15m[df_15m['bos_signal'] != 0].index
            
            # Векторизуем предсказания (Batch processing) для скорости
            bos_pos = [df_15m.index.get_loc(idx) for idx in bos_indices if df_15m.index.get_loc(idx) >= seq_len]
            if not bos_pos: continue
            
            sequences = [norm_data[p-seq_len : p] for p in bos_pos]
            x_tensor = torch.from_numpy(np.array(sequences)).float()
            with torch.no_grad():
                probs = model(x_tensor).squeeze().tolist()
            if isinstance(probs, float): probs = [probs]
            
            prob_map = dict(zip([df_15m.index[p] for p in bos_pos], probs))

            for i in range(seq_len + 1, len(df_15m)):
                candle = df_15m.iloc[i]
                idx = df_15m.index[i]
                
                if active_trade:
                    if active_trade['type'] == 'bullish':
                        if candle['low'] <= active_trade['sl']: trades.append(-1.0); active_trade = None
                        elif candle['high'] >= active_trade['tp']: trades.append(2.0); active_trade = None
                    else:
                        if candle['high'] >= active_trade['sl']: trades.append(-1.0); active_trade = None
                        elif candle['low'] <= active_trade['tp']: trades.append(2.0); active_trade = None
                    continue

                if idx in prob_map and prob_map[idx] > 0.52:
                    bias = candle['htf_bias']
                    bos = candle['bos_signal']
                    if bias == 1 and bos == 1:
                        sl = df_15m['swing_low'].iloc[i-20:i].min()
                        if np.isnan(sl) or sl >= candle['close']: sl = candle['close'] * 0.99
                        tp = candle['close'] + 2.0 * (candle['close'] - sl)
                        active_trade = {'type': 'bullish', 'sl': sl, 'tp': tp}
                    elif bias == -1 and bos == -1:
                        sl = df_15m['swing_high'].iloc[i-20:i].max()
                        if np.isnan(sl) or sl <= candle['close']: sl = candle['close'] * 1.01
                        tp = candle['close'] - 2.0 * (sl - candle['close'])
                        active_trade = {'type': 'bearish', 'sl': sl, 'tp': tp}

            if trades:
                wr = len([t for t in trades if t > 0]) / len(trades)
                pnl = sum(trades)
                total_trades_all.extend(trades)
                print(f"✅ {symbol}: {len(trades)} trades, PnL: {pnl:.1f}R, WR: {wr:.1%}")
                
        except Exception as e:
            print(f"❌ {symbol} Error: {e}")

    if total_trades_all:
        print(f"\nИТОГО (TRANSFORMER TOP-10):")
        print(f"Всего сделок: {len(total_trades_all)}")
        print(f"Winrate: {len([t for t in total_trades_all if t > 0])/len(total_trades_all):.1%}")
        print(f"Общий Профит: {sum(total_trades_all):.2f}R")

if __name__ == "__main__":
    run_transformer_backtest()
