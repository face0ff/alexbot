import logging
import json
import pandas as pd
import os
from data.storage import DataStorage
from pattern.detector import PatternDetector
from backtest.engine import BacktestEngine
from backtest.metrics import MetricsCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_comparison():
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    timeframe = '1h'
    config_path = 'config/pattern_spec.json'
    
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    detector = PatternDetector(config_path)
    bt_engine = BacktestEngine(config)
    metrics_calc = MetricsCalculator()
    
    all_results_breakout = []
    all_results_limit = []
    
    for symbol in symbols:
        data_path = f"data_{symbol.replace('/', '_')}_{timeframe}.parquet"
        if not os.path.exists(data_path):
            logger.warning(f"Data for {symbol} not found. Skipping.")
            continue
            
        df = DataStorage.load_from_parquet(data_path)
        patterns = detector.detect_patterns(df)
        
        res_breakout = bt_engine.run_backtest(patterns, df, entry_mode='BREAKOUT')
        all_results_breakout.append(res_breakout)
        
        res_limit = bt_engine.run_backtest(patterns, df, entry_mode='LIMIT')
        all_results_limit.append(res_limit)
        
    if not all_results_breakout:
        print("No data found to compare. Please run main.py first.")
        return

    df_breakout = pd.concat(all_results_breakout)
    df_limit = pd.concat(all_results_limit)
    
    metrics_breakout = metrics_calc.calculate(df_breakout)
    metrics_limit = metrics_calc.calculate(df_limit)
    
    print("\n" + "="*60)
    print("STRATEGY COMPARISON: BREAKOUT VS LIMIT (BTC, ETH, SOL)")
    print("="*60)
    print(f"{'Metric':<20} | {'BREAKOUT':<15} | {'LIMIT (0.618)':<15}")
    print("-" * 60)
    for key in ['total_trades', 'win_rate', 'expectancy', 'profit_factor', 'net_profit_r']:
        v1 = f"{metrics_breakout[key]:.3f}" if isinstance(metrics_breakout[key], float) else str(metrics_breakout[key])
        v2 = f"{metrics_limit[key]:.3f}" if isinstance(metrics_limit[key], float) else str(metrics_limit[key])
        print(f"{key:<20} | {v1:<15} | {v2:<15}")
    print("="*60)

if __name__ == "__main__":
    run_comparison()
