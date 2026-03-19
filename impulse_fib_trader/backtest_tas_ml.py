import pandas as pd
import numpy as np
import json
import os
from pattern.tas_detector import TASDetector
from data.storage import DataStorage
from data.cleaner import DataCleaner
from ml.train import MLTrainer
from features.engineer import FeatureEngineer

def run_backtest_ml():
    config_path = 'impulse_fib_trader/config/pattern_spec_tas.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    detector = TASDetector(config)
    cleaner = DataCleaner()
    fe = FeatureEngineer()
    trainer = MLTrainer()
    
    model_path = 'trained_model_tas.joblib'
    if os.path.exists(model_path):
        trainer.load_model(model_path)
    else:
        print("Model not found!")
        return

    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT']
    results = []

    print(f"--- Бэктест TAS_v1 + ML (Threshold 0.60) ---")
    print(f"{'Symbol':<10} | {'Trades':<7} | {'Winrate':<8} | {'Profit (R)':<10} | {'PF':<5}")
    print("-" * 55)

    for symbol in symbols:
        data_path = f"data_{symbol.replace('/', '_')}_1h_tas.parquet"
        if not os.path.exists(data_path): continue
        
        df = DataStorage.load_from_parquet(data_path)
        df = cleaner.calculate_indicators(df)
        
        patterns = detector.detect_patterns(df)
        if not patterns: continue
        
        # Получаем предсказания ML для всех паттернов сразу
        X = fe.extract_features(patterns, df)
        probs = trainer.model.predict_proba(X)
        
        trades = []
        for idx, p in enumerate(patterns):
            prob = float(probs[idx][1])
            
            # Фильтр ML: только сделки с вероятностью > 60%
            if prob < 0.60:
                continue
                
            entry_idx = p['entry_idx']
            entry_price = p['entry_price']
            sl = p['tail_low']
            risk = entry_price - sl
            if risk <= 0: continue
            
            tp = entry_price + (risk * 2.0)
            
            outcome = 0
            end_search = min(entry_idx + 48, len(df) - 1)
            for i in range(entry_idx + 1, end_search + 1):
                low = df.iloc[i]['low']
                high = df.iloc[i]['high']
                if low <= sl:
                    outcome = -1
                    break
                if high >= tp:
                    outcome = 2
                    break
            
            if outcome != 0:
                trades.append(outcome)
        
        if not trades: continue
        
        trades_arr = np.array(trades)
        wins = len(trades_arr[trades_arr > 0])
        wr = wins / len(trades_arr)
        total_r = trades_arr.sum()
        
        gross_profit = trades_arr[trades_arr > 0].sum()
        gross_loss = abs(trades_arr[trades_arr < 0].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        print(f"{symbol:<10} | {len(trades):<7} | {wr:<8.2%} | {total_r:<10.1f} | {pf:<5.2f}")
        results.extend(trades)

    if results:
        res_arr = np.array(results)
        print("-" * 55)
        print(f"ИТОГО С ML: {len(res_arr)} сделок")
        print(f"Общий профит: {res_arr.sum():.1f} R")
        print(f"Winrate: {len(res_arr[res_arr > 0])/len(res_arr):.2%}")
        print(f"Profit Factor: {res_arr[res_arr > 0].sum()/abs(res_arr[res_arr < 0].sum()):.2f}")

if __name__ == "__main__":
    run_backtest_ml()
