import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from data.fetcher import DataFetcher
from data.cleaner import DataCleaner
from pattern.market_structure_smc import SMCMarketStructure
from features.labels_smc import SMCLabeler
from ml.transformer_smc import SMCTransformer

def collect_sequences(symbols, timeframe='15m', seq_len=50):
    fetcher = DataFetcher()
    cleaner = DataCleaner()
    smc = SMCMarketStructure(window=5)
    labeler = SMCLabeler()
    
    all_X = []
    all_y = []
    
    print(f"--- СБОР ДАННЫХ ДЛЯ ПРАВИЛЬНОГО ОБУЧЕНИЯ ---")
    
    for symbol in symbols:
        try:
            df = fetcher.fetch_ohlcv(symbol, timeframe, '2021-01-01', '2023-12-31')
            if df.empty or len(df) < seq_len + 100: continue
            
            df = cleaner.validate_data(df)
            df = cleaner.calculate_indicators(df)
            df = smc.detect_bos(df)
            y_df = labeler.create_labels(df)
            
            ohlc = df[['open', 'high', 'low', 'close']].values
            atr = df['atr'].values.reshape(-1, 1)
            atr[atr == 0] = np.nanmean(atr)
            norm_data = np.nan_to_num((ohlc[1:] - ohlc[:-1]) / (atr[1:] + 1e-9))
            norm_data = np.clip(norm_data, -5.0, 5.0)
            
            bos_indices = df[df['bos_signal'] != 0].index
            for idx in bos_indices:
                pos = df.index.get_loc(idx)
                if pos < seq_len or idx not in y_df.index: continue
                all_X.append(norm_data[pos - seq_len : pos])
                all_y.append(float(y_df.loc[idx, 'target']))
            print(f"✅ {symbol} processed")
        except: continue
            
    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.float32)
    
    # --- БАЛАНСИРОВКА ---
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    
    print(f"До балансировки: Profit={len(pos_idx)}, Loss={len(neg_idx)}")
    
    # Берем равное количество примеров
    n_samples = min(len(pos_idx), len(neg_idx))
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    
    balanced_idx = np.concatenate([pos_idx[:n_samples], neg_idx[:n_samples]])
    np.random.shuffle(balanced_idx)
    
    return X[balanced_idx], y[balanced_idx].reshape(-1, 1)

def train():
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'LINK/USDT', 'ADA/USDT', 'DOT/USDT']
    X_train, y_train = collect_sequences(symbols)
    
    if len(X_train) == 0: return

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    
    model = SMCTransformer(input_dim=4, d_model=64, nhead=4, num_layers=3)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    print(f"--- ОБУЧЕНИЕ НА СБАЛАНСИРОВАННЫХ ДАННЫХ ({len(X_train)} примеров) ---")
    
    for epoch in range(30):
        model.train()
        total_loss = 0
        correct = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += ((output > 0.5).float() == batch_y).sum().item()
        
        acc = correct / len(X_train)
        print(f"Epoch {epoch+1:2d} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.2%}")

    torch.save(model.state_dict(), 'smc_transformer_v1.pth')
    print("✅ Сбалансированная модель сохранена!")

if __name__ == "__main__":
    train()
