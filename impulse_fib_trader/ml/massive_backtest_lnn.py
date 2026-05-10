import torch
import pandas as pd
import numpy as np
import os
from data.fetcher import DataFetcher
from data.cleaner import DataCleaner
from ml.lnn_model import LiquidNet

# --- CONFIG ---
START_DATE = '2025-01-01'
END_DATE = '2026-04-06'
SEQ_LEN = 60
MODEL_PATH = 'lnn_filter.pth'
THRESHOLD = 0.60
HIDDEN_SIZE = 128

def run_massive_lnn_test():
    fetcher = DataFetcher()
    cleaner = DataCleaner()
    
    # Загрузка модели
    model = LiquidNet(5, HIDDEN_SIZE, 1)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
    else:
        print("Модель не найдена!")
        return

    # Получаем Топ-50 монет
    all_symbols = fetcher.get_active_symbols()
    test_symbols = [s for s in all_symbols if s.endswith('/USDT')][:50]
    
    print(f"--- MASSIVE LNN FILTER TEST (50 COINS, 2025-2026) ---")
    
    overall_trades = 0
    overall_wins = 0
    overall_pnl = 0
    results = []

    for symbol in test_symbols:
        try:
            df = fetcher.fetch_ohlcv(symbol, '15m', START_DATE, END_DATE)
            if len(df) < 200: continue
            
            df = cleaner.validate_data(df)
            df = cleaner.calculate_indicators(df)
            
            df['returns'] = df['close'].pct_change()
            df['hl_range'] = (df['high'] - df['low']) / df['close']
            df['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            df['dist_ema'] = (df['close'] / df['ema_200']) - 1
            df = df.dropna()
            
            features = ['returns', 'hl_range', 'vol_ratio', 'dist_ema', 'rsi']
            X_raw = df[features].values
            
            # Предсказания пакетами
            sequences = [X_raw[i - SEQ_LEN : i] for i in range(SEQ_LEN, len(df) - 5)]
            if not sequences: continue
            
            X_tensor = torch.from_numpy(np.array(sequences)).float()
            all_probs = []
            with torch.no_grad():
                for i in range(0, len(X_tensor), 1024):
                    batch = X_tensor[i : i + 1024]
                    probs = torch.sigmoid(model(batch)).squeeze().tolist()
                    if isinstance(probs, float): probs = [probs]
                    all_probs.extend(probs)

            # Симуляция для монеты
            s_trades = 0
            s_wins = 0
            s_pnl = 1.0 # Multiplier
            
            for i, prob in enumerate(all_probs):
                if prob >= THRESHOLD:
                    real_idx = i + SEQ_LEN
                    change = (df['close'].iloc[real_idx + 5] / df['close'].iloc[real_idx]) - 1
                    s_pnl *= (1 + change)
                    s_trades += 1
                    if change > 0: s_wins += 1
            
            if s_trades > 0:
                overall_trades += s_trades
                overall_wins += s_wins
                overall_pnl += (s_pnl - 1)
                wr = s_wins / s_trades
                print(f"✅ {symbol}: {s_trades} trades, WR: {wr:.1%}, PnL: {(s_pnl-1)*100:+.2f}%")
                results.append({'symbol': symbol, 'trades': s_trades, 'wr': wr, 'pnl': s_pnl-1})
                
        except Exception as e:
            continue

    if overall_trades > 0:
        print(f"\n--- ИТОГО ПО ВСЕМ МОНЕТАМ ---")
        print(f"Всего сделок: {overall_trades}")
        print(f"Средний Winrate: {overall_wins/overall_trades:.1%}")
        print(f"Суммарная доходность (avg per coin): {overall_pnl/len(test_symbols)*100:.2f}%")
        
        # Топ монет для фильтра
        best = sorted(results, key=lambda x: x['pnl'], reverse=True)[:5]
        print("\nЛучшие монеты для этого фильтра:")
        for b in best:
            print(f"- {b['symbol']}: {b['pnl']*100:.1f}% profit")

if __name__ == "__main__":
    run_massive_lnn_test()
