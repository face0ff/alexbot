import pandas as pd
import numpy as np
import json
import os
from pattern.tas_detector import ImpulseRejectionDetector
from data.fetcher import DataFetcher
from data.cleaner import DataCleaner
from ml.train import MLTrainer
from features.engineer import FeatureEngineer

def test_rejection_oos():
    config_path = 'impulse_fib_trader/config/pattern_spec_tas.json'
    with open(config_path, 'r') as f: config = json.load(f)
    detector = ImpulseRejectionDetector(config)
    cleaner = DataCleaner()
    fe = FeatureEngineer()
    trainer = MLTrainer()
    
    model_path = 'trained_model_tas_2023.joblib'
    if not os.path.exists(model_path): return
    trainer.load_model(model_path)

    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT']
    fetcher = DataFetcher()
    results = []

    print(f"--- HYBRID (Trend + Rejection) TEST 2024 ---")
    print(f"{'Symbol':<10} | {'Trades':<7} | {'Winrate':<8} | {'Profit (R)':<10} | {'PF':<5}")
    print("-" * 60)

    for symbol in symbols:
        df = fetcher.fetch_ohlcv(symbol, '1h', '2024-01-01', '2024-12-31')
        if df.empty: continue
        df = cleaner.validate_data(df)
        df = cleaner.calculate_indicators(df)
        patterns = detector.detect_patterns(df)
        if not patterns: continue
        
        X = fe.extract_features(patterns, df)
        probs = trainer.model.predict_proba(X)
        
        trades = []
        for idx, p in enumerate(patterns):
            if float(probs[idx][1]) < 0.55: continue
            
            entry_idx = p['entry_idx']
            entry_price = p['entry_price']
            sl = p['sl']
            risk = entry_price - sl
            if risk <= 0: continue
            
            tp = entry_price + (risk * 2.0)
            
            outcome = 0
            end_search = min(entry_idx + 48, len(df) - 1)
            for i in range(entry_idx + 1, end_search + 1):
                if df.iloc[i]['low'] <= sl: outcome = -1; break
                if df.iloc[i]['high'] >= tp: outcome = 2; break
            
            if outcome != 0: trades.append(outcome)
        
        if not trades: continue
        t_arr = np.array(trades)
        wr = len(t_arr[t_arr > 0]) / len(t_arr)
        pf = t_arr[t_arr > 0].sum() / abs(t_arr[t_arr < 0].sum()) if (t_arr < 0).any() else float('inf')
        print(f"{symbol:<10} | {len(trades):<7} | {wr:<8.2%} | {t_arr.sum():<10.1f} | {pf:<5.2f}")
        results.extend(trades)

    if results:
        res_arr = np.array(results)
        print("-" * 60)
        print(f"ИТОГО: {len(res_arr)} сделок | Профит: {res_arr.sum():.1f} R | Winrate: {len(res_arr[res_arr > 0])/len(res_arr):.2%} | PF: {res_arr[res_arr > 0].sum()/abs(res_arr[res_arr < 0].sum()):.2f}")

if __name__ == "__main__":
    test_rejection_oos()
