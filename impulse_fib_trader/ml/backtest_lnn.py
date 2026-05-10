import torch
import pandas as pd
import numpy as np
import os
from data.fetcher import DataFetcher
from data.cleaner import DataCleaner
from ml.lnn_model import LiquidNet

SYMBOL = 'BTC/USDT'
TIMEFRAME = '15m'
START_DATE = '2024-01-01'
END_DATE = '2026-04-06'
SEQ_LEN = 60
MODEL_PATH = 'lnn_filter.pth'
THRESHOLD = 0.60
HIDDEN_SIZE = 128

def run_backtest():
    fetcher = DataFetcher()
    cleaner = DataCleaner()
    
    df = fetcher.fetch_ohlcv(SYMBOL, TIMEFRAME, START_DATE, END_DATE)
    df = cleaner.validate_data(df)
    df = cleaner.calculate_indicators(df)
    
    df['returns'] = df['close'].pct_change()
    df['hl_range'] = (df['high'] - df['low']) / df['close']
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['dist_ema'] = (df['close'] / df['ema_200']) - 1
    df = df.dropna()
    
    features = ['returns', 'hl_range', 'vol_ratio', 'dist_ema', 'rsi']
    X_raw = df[features].values
    
    model = LiquidNet(len(features), HIDDEN_SIZE, 1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    print("--- ВЕКТОРНЫЙ БЭКТЕСТ LNN ---")
    
    # Готовим все последовательности
    sequences = []
    for i in range(SEQ_LEN, len(df) - 5):
        sequences.append(X_raw[i - SEQ_LEN : i])
    
    X_tensor = torch.from_numpy(np.array(sequences)).float()
    
    # Предсказания пакетами по 512 для скорости
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), 512):
            batch = X_tensor[i : i + 512]
            probs = torch.sigmoid(model(batch)).squeeze().tolist()
            if isinstance(probs, float): probs = [probs]
            all_probs.extend(probs)
            if i % 5120 == 0: print(f"Progress: {i}/{len(X_tensor)}")

    # Симуляция
    equity = 1000
    trades = 0
    wins = 0
    
    for i, prob in enumerate(all_probs):
        if prob >= THRESHOLD:
            # Свеча входа - i + SEQ_LEN
            # Изменение цены через 5 свечей
            real_idx = i + SEQ_LEN
            price_change = (df['close'].iloc[real_idx + 5] / df['close'].iloc[real_idx]) - 1
            equity *= (1 + price_change)
            trades += 1
            if price_change > 0: wins += 1

    print(f"\nИТОГИ LNN ФИЛЬТРА:")
    print(f"Всего сделок: {trades}")
    print(f"Winrate: {wins/trades:.1%}" if trades > 0 else "0%")
    print(f"Финальный баланс: {equity:.2f} USDT")
    print(f"Чистая доходность: {(equity/1000 - 1)*100:.2f}%")

if __name__ == "__main__":
    run_backtest()
