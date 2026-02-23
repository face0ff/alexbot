# Impulse-Fib618-Pullback-Continuation HTF Trader

Algorithmic trading system for cryptocurrency futures on High Timeframes (H1+).

## Features
- **Data Engine**: Automated OHLCV fetching from Binance Futures via CCXT.
- **Pattern Detector**: Detects strong impulsive moves followed by Fibonacci retracements (0.5 - 0.705).
- **ML Classifier**: XGBoost-based filter to improve trade quality by analyzing impulse/pullback characteristics.
- **Backtest Engine**: Rule-based simulation with R-multiple metrics.

## Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the full pipeline (Data -> Patterns -> Backtest -> ML):
```bash
python main.py
```

## Project Structure
- `data/`: Data fetching and cleaning.
- `pattern/`: Core detection logic (Impulse, Pullback, Structure).
- `features/`: Feature engineering for ML.
- `backtest/`: Simulation and performance metrics.
- `ml/`: Training and prediction pipelines.
- `config/`: JSON/YAML configuration for pattern parameters.

## Results (ETH/USDT H1 2023)
- Total patterns: ~350
- Rule-based Win Rate: ~78%
- ML-Filtered Win Rate: ~97% (on training/validation set)
