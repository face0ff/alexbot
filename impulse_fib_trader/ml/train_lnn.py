import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from data.fetcher import DataFetcher
from data.cleaner import DataCleaner
from ml.lnn_model import LiquidNet
from sklearn.metrics import accuracy_score, precision_score, recall_score

# --- CONFIG ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '15m'
START_DATE = '2021-01-01'
END_DATE = '2023-12-31'
SEQ_LEN = 60
PREDICT_DISTANCE = 5
MODEL_PATH = 'lnn_filter.pth'
HIDDEN_SIZE = 128
BATCH_SIZE = 128
EPOCHS = 30
LR = 0.0005

def prepare_data():
    fetcher = DataFetcher()
    cleaner = DataCleaner()
    
    df = fetcher.fetch_ohlcv(SYMBOL, TIMEFRAME, START_DATE, END_DATE)
    df = cleaner.validate_data(df)
    df = cleaner.calculate_indicators(df)
    
    # Расширенные признаки
    df['returns'] = df['close'].pct_change()
    df['hl_range'] = (df['high'] - df['low']) / df['close']
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['dist_ema'] = (df['close'] / df['ema_200']) - 1
    
    df = df.dropna()
    
    # Цель: существенный рост (> 0.2%)
    df['target'] = (df['close'].shift(-PREDICT_DISTANCE) / df['close'] - 1 > 0.002).astype(float)
    df = df.dropna()
    
    features = ['returns', 'hl_range', 'vol_ratio', 'dist_ema', 'rsi']
    X_raw = df[features].values
    y_raw = df['target'].values
    
    X_seq, y_seq = [], []
    for i in range(len(df) - SEQ_LEN):
        X_seq.append(X_raw[i : i + SEQ_LEN])
        y_seq.append(y_raw[i + SEQ_LEN])
        
    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32).reshape(-1, 1)
    
    # БАЛАНСИРОВКА
    pos_idx = np.where(y_seq == 1)[0]
    neg_idx = np.where(y_seq == 0)[0]
    n = min(len(pos_idx), len(neg_idx))
    
    balanced_idx = np.concatenate([pos_idx[:n], neg_idx[:n]])
    np.random.shuffle(balanced_idx)
    
    X_balanced = X_seq[balanced_idx]
    y_balanced = y_seq[balanced_idx]
    
    # Split
    split = int(len(X_balanced) * 0.8)
    return X_balanced[:split], X_balanced[split:], y_balanced[:split], y_balanced[split:], len(features)

def train():
    X_train, X_test, y_train, y_test, input_dim = prepare_data()
    
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)), batch_size=BATCH_SIZE)
    
    model = LiquidNet(input_dim, HIDDEN_SIZE, 1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"--- ОБУЧЕНИЕ СБАЛАНСИРОВАННОЙ LNN ({len(X_train)} примеров) ---")
    
    for epoch in range(EPOCHS):
        model.train()
        for b_x, b_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(b_x), b_y)
            loss.backward()
            optimizer.step()
            
        # Eval
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for b_x, b_y in test_loader:
                p = (torch.sigmoid(model(b_x)) > 0.5).float()
                preds.extend(p.numpy()); targets.extend(b_y.numpy())
        
        acc = accuracy_score(targets, preds)
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1:2d} | Test Acc: {acc:.2%}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Модель сохранена: {MODEL_PATH} (Acc: {acc:.2%})")

if __name__ == "__main__":
    train()
