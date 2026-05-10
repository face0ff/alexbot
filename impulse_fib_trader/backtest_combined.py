
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
try:
    from pattern.combined_detector import CombinedTASFibDetector
    from data.cleaner import DataCleaner
    from features.engineer import FeatureEngineer
except ModuleNotFoundError:
    from impulse_fib_trader.pattern.combined_detector import CombinedTASFibDetector
    from impulse_fib_trader.data.cleaner import DataCleaner
    from impulse_fib_trader.features.engineer import FeatureEngineer


def run_combined_backtest():
    # Настройки
    MODEL_PATH = 'super_model_combined.joblib'
    detector = CombinedTASFibDetector()
    cleaner = DataCleaner()
    fe = FeatureEngineer()
    
    model = None
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print(f"✅ ML Модель загружена: {MODEL_PATH}")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки модели: {e}")
    else:
        print("⚠️ Модель не найдена, бэктест без ML фильтра.")

    # Поиск файлов данных
    data_files = [f for f in os.listdir('.') if f.endswith('.parquet') and '_1h_tas.parquet' in f]
    if not data_files:
        print("Файлы данных не найдены в корневой папке.")
        return

    all_trades = []
    print(f"{'Symbol':<12} | {'Trades':<7} | {'Winrate':<8} | {'Profit (R)':<10} | {'PF':<5}")
    print("-" * 55)

    for file in data_files:
        symbol = file.replace('data_', '').replace('_1h_tas.parquet', '').replace('_', '/')
        df = pd.read_parquet(file)
        
        # Расчет индикаторов (RSI, EMA и др.)
        df = cleaner.calculate_indicators(df)
        
        # Поиск паттернов
        patterns = detector.detect_patterns(df)
        
        symbol_trades = []
        for p in patterns:
            entry_idx = p['entry_idx']
            if entry_idx >= len(df) - 1: continue
            
            entry_price = p['entry_price']
            sl = p['sl']
            tp = p['tp'] # В детекторе это p1 (максимум импульса)
            
            # В боте тейк считается как RR 2.0 от риска, если риск адекватный
            risk = entry_price - sl
            if risk <= 0: continue
            
            # Тейк в боте часто пересчитывается (см. telegram_bot.py)
            tp = entry_price + (risk * 2.0)
            
            # ML Фильтрация
            if model:
                try:
                    # Для извлечения фичей нужна история до входа
                    features = fe.extract_features([p], df.iloc[:entry_idx+1])
                    prob = model.predict_proba(features)[0][1]
                    if prob < 0.52: continue # Тот же порог, что в боте
                except:
                    continue
            
            # Симуляция сделки
            outcome = 0 
            # Ограничение по времени - 48 свечей (2 суток на H1)
            end_idx = min(entry_idx + 48, len(df) - 1)
            
            max_p = entry_price
            curr_sl = sl
            
            # Настройки защиты
            BE_TRIGGER = 0.012
            BE_LEVEL = 0.002
            TRAILING_TRIGGER = 0.02
            TRAILING_DIST = 0.012

            for i in range(entry_idx + 1, end_idx + 1):
                low = df.iloc[i]['low']
                high = df.iloc[i]['high']
                if high > max_p: max_p = high
                
                # БУ
                if (max_p / entry_price - 1) >= BE_TRIGGER:
                    be_sl = entry_price * (1 + BE_LEVEL)
                    if be_sl > curr_sl: curr_sl = be_sl
                
                # Трейлинг
                if (max_p / entry_price - 1) >= TRAILING_TRIGGER:
                    t_sl = max_p * (1 - TRAILING_DIST)
                    if t_sl > curr_sl: curr_sl = t_sl

                if low <= curr_sl:
                    outcome = (curr_sl - entry_price) / risk
                    break
                if high >= tp:
                    outcome = (tp - entry_price) / risk
                    break
            
            if outcome != 0:
                symbol_trades.append(outcome)
        
        if not symbol_trades: continue
        
        trades_arr = np.array(symbol_trades)
        wins = len(trades_arr[trades_arr > 0])
        wr = wins / len(trades_arr)
        total_r = trades_arr.sum()
        
        gross_profit = trades_arr[trades_arr > 0].sum()
        gross_loss = abs(trades_arr[trades_arr < 0].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        print(f"{symbol:<12} | {len(symbol_trades):<7} | {wr:<8.2%} | {total_r:<10.1f} | {pf:<5.2f}")
        all_trades.extend(symbol_trades)

    if all_trades:
        res_arr = np.array(all_trades)
        print("-" * 55)
        print(f"ИТОГО ПО ВСЕМ ПАРАМ: {len(res_arr)} сделок")
        print(f"Общий профит: {res_arr.sum():.1f} R")
        print(f"Средний Winrate: {len(res_arr[res_arr > 0])/len(res_arr):.2%}")
        print(f"Profit Factor: {res_arr[res_arr > 0].sum()/abs(res_arr[res_arr < 0].sum()):.2f}")
    else:
        print("Сделок по заданным критериям не найдено.")

if __name__ == "__main__":
    run_combined_backtest()
