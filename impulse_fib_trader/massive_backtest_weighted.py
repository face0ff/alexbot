
import asyncio
import ccxt
import pandas as pd
import numpy as np
import os
import json
import joblib
from datetime import datetime, timedelta
from typing import List, Dict

# Import existing modules
try:
    from pattern.combined_detector import CombinedTASFibDetector
    from pattern.tas_detector import ImpulseRejectionDetector as TASDetector
    from pattern.fib75_detector import ImpulseFib75Detector
    from pattern.market_structure_smc import SMCMarketStructure
    from data.cleaner import DataCleaner
    from features.engineer import FeatureEngineer
except ImportError:
    from impulse_fib_trader.pattern.combined_detector import CombinedTASFibDetector
    from impulse_fib_trader.pattern.tas_detector import ImpulseRejectionDetector as TASDetector
    from impulse_fib_trader.pattern.fib75_detector import ImpulseFib75Detector
    from impulse_fib_trader.pattern.market_structure_smc import SMCMarketStructure
    from impulse_fib_trader.data.cleaner import DataCleaner
    from impulse_fib_trader.features.engineer import FeatureEngineer

# Configuration
TIMEFRAME = '15m'
DAYS_BACK = 14
MAX_CONCURRENT_TRADES = 3
WHITELIST_PATH = 'impulse_fib_trader/config/whitelist.json'
MODEL_PATH = 'super_model_combined.joblib'

class WeightedBacktest:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.cleaner = DataCleaner()
        self.fe = FeatureEngineer()
        self.smc = SMCMarketStructure(window=3)
        self.ifpc_detector = CombinedTASFibDetector()
        self.tas_detector = TASDetector(config={})
        self.fib75_detector = ImpulseFib75Detector()
        
        self.model = None
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            print(f"✅ ML Model loaded: {MODEL_PATH}")

        with open(WHITELIST_PATH, 'r') as f:
            self.whitelist = json.load(f)

    async def fetch_data(self, symbol: str, timeframe: str, days: int):
        print(f"Fetching {symbol} ({timeframe})...")
        since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        all_ohlcv = []
        while since < int(datetime.now().timestamp() * 1000):
            try:
                ohlcv = await asyncio.to_thread(self.exchange.fetch_ohlcv, symbol, timeframe, since, 1000)
                if not ohlcv: break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                if len(ohlcv) < 1000: break
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                break
        
        if not all_ohlcv: return pd.DataFrame()
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def run_standard_weighting(self, all_data: Dict[str, pd.DataFrame]):
        """Phase 1: Calculate weights based on IFPC performance."""
        print("\n--- PHASE 1: CALCULATING WEIGHTS (STANDARD IFPC) ---")
        weights = {}
        for symbol, df in all_data.items():
            if df.empty: continue
            df_with_ind = self.cleaner.calculate_indicators(df.copy())
            patterns = self.ifpc_detector.detect_patterns(df_with_ind)
            
            pnl_r = 0
            trades = 0
            for p in patterns:
                idx = p['entry_idx']
                if idx >= len(df_with_ind) - 1: continue
                
                # Simulation
                entry_p = p['entry_price']
                sl = p['sl']
                risk = abs(entry_p - sl)
                if risk == 0: continue
                tp = entry_p + (2.0 * (entry_p - sl)) if entry_p > sl else entry_p - (2.0 * (sl - entry_p))
                
                res = 0
                for j in range(idx + 1, min(idx + 50, len(df_with_ind))):
                    low, high = df_with_ind.iloc[j]['low'], df_with_ind.iloc[j]['high']
                    if entry_p > sl: # Long
                        if low <= sl: res = -1; break
                        if high >= tp: res = 2; break
                    else: # Short
                        if high <= sl: res = -1; break
                        if low >= tp: res = 2; break
                
                if res != 0:
                    pnl_r += res
                    trades += 1
            
            # Simple weighting formula
            if trades > 0:
                weights[symbol] = max(0.1, pnl_r / trades + (trades * 0.05)) # Reward both winrate and frequency
            else:
                weights[symbol] = 0.1
        
        # Normalize weights
        max_w = max(weights.values()) if weights else 1
        for s in weights:
            weights[s] = weights[s] / max_w
            
        print("Top 5 weighted coins:")
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        for s, w in sorted_weights[:5]:
            print(f"- {s}: {w:.2f}")
            
        return weights

    def detect_all_signals(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Detect signals from all 4 strategies."""
        signals = []
        
        # 1. IFPC
        ifpc_p = self.ifpc_detector.detect_patterns(df)
        for p in ifpc_p:
            p['strategy'] = 'IFPC'
            p['symbol'] = symbol
            signals.append(p)
            
        # 2. TAS
        tas_p = self.tas_detector.detect_patterns(df)
        for p in tas_p:
            p['strategy'] = 'TAS'
            p['symbol'] = symbol
            signals.append(p)
            
        # 3. Fib75
        fib75_p = self.fib75_detector.detect_patterns(df)
        for p in fib75_p:
            p['strategy'] = 'Fib75'
            p['symbol'] = symbol
            signals.append(p)
            
        # 4. SMC
        df_smc = self.smc.detect_bos(df.copy())
        for i in range(len(df_smc)):
            if df_smc['bos_signal'].iloc[i] == 1:
                signals.append({
                    'strategy': 'SMC',
                    'symbol': symbol,
                    'type': 'SMC_BOS',
                    'entry_idx': i,
                    'entry_price': df_smc['close'].iloc[i],
                    'sl': df_smc['low'].iloc[max(0, i-5):i].min(),
                    'tp': df_smc['close'].iloc[i] * 1.02, # Target 2%
                    'timestamp': df_smc.index[i]
                })
        
        return signals

    async def run_weighted_backtest(self):
        # 1. Fetch data for all coins
        all_data = {}
        tasks = [self.fetch_data(s, TIMEFRAME, DAYS_BACK) for s in self.whitelist]
        results = await asyncio.gather(*tasks)
        for s, df in zip(self.whitelist, results):
            if not df.empty:
                all_data[s] = df

        if not all_data:
            print("No data fetched.")
            return

        # 2. Phase 1: Get Weights
        weights = self.run_standard_weighting(all_data)

        # 3. Phase 2: Multi-Strategy Backtest
        print("\n--- PHASE 2: MULTI-STRATEGY WEIGHTED BACKTEST (Top 20 coins) ---")
        
        # Pre-detect all signals to simulate timeline
        all_signals = []
        top_symbols = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:20]
        top_symbols = [s[0] for s in top_symbols]
        
        for symbol in top_symbols:
            df = all_data[symbol]
            df_with_ind = self.cleaner.calculate_indicators(df.copy())
            signals = self.detect_all_signals(df_with_ind, symbol)
            all_signals.extend(signals)
            
        # Sort signals by timestamp
        all_signals.sort(key=lambda x: x['timestamp'])
        
        active_trades = []
        trade_history = []
        balance_r = 0
        
        # Group signals by timestamp
        from collections import defaultdict
        timeline = defaultdict(list)
        for sig in all_signals:
            timeline[sig['timestamp']].append(sig)
            
        sorted_times = sorted(timeline.keys())
        
        for t in sorted_times:
            current_signals = timeline[t]
            
            # Check for exits of active trades
            remaining_trades = []
            for trade in active_trades:
                df = all_data[trade['symbol']]
                try:
                    # Find candle at or after signal
                    candle_idx = df.index.get_indexer([t], method='pad')[0]
                    if candle_idx == -1: 
                        remaining_trades.append(trade)
                        continue
                    
                    low, high = df.iloc[candle_idx]['low'], df.iloc[candle_idx]['high']
                    
                    exit_r = 0
                    if trade['side'] == 'long':
                        if low <= trade['sl']: exit_r = -1
                        elif high >= trade['tp']: exit_r = 2
                    else:
                        if high >= trade['sl']: exit_r = -1
                        elif low <= trade['tp']: exit_r = 2
                        
                    if exit_r != 0:
                        balance_r += exit_r
                        trade['exit_time'] = t
                        trade['result_r'] = exit_r
                        trade_history.append(trade)
                    else:
                        remaining_trades.append(trade)
                except:
                    remaining_trades.append(trade)
            active_trades = remaining_trades

            # Filter signals (don't enter if already in trade for that symbol)
            available_signals = [s for s in current_signals if not any(at['symbol'] == s['symbol'] for at in active_trades)]
            
            if available_signals and len(active_trades) < MAX_CONCURRENT_TRADES:
                # SELECT BEST SIGNAL(S) based on coin weights
                available_signals.sort(key=lambda x: weights.get(x['symbol'], 0), reverse=True)
                
                for sig in available_signals:
                    if len(active_trades) >= MAX_CONCURRENT_TRADES: break
                    
                    # Entry logic
                    new_trade = {
                        'symbol': sig['symbol'],
                        'strategy': sig['strategy'],
                        'entry_time': t,
                        'entry_price': sig['entry_price'],
                        'sl': sig['sl'],
                        'tp': sig['tp'] if 'tp' in sig else sig['entry_price'] * 1.02,
                        'side': 'long' if sig['entry_price'] > sig['sl'] else 'short'
                    }
                    active_trades.append(new_trade)

        # Final Report
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Total Trades: {len(trade_history)}")
        print(f"Total Profit (R): {balance_r:.2f}")
        
        if trade_history:
            wins = len([t for t in trade_history if t['result_r'] > 0])
            print(f"Winrate: {wins/len(trade_history):.2%}")
            
            # Strategy breakdown
            strat_stats = defaultdict(lambda: {'r': 0, 'count': 0})
            for t in trade_history:
                strat_stats[t['strategy']]['r'] += t['result_r']
                strat_stats[t['strategy']]['count'] += 1
            
            print("\nStrategy Breakdown:")
            for s, st in strat_stats.items():
                print(f"- {s}: {st['count']} trades, {st['r']:.1f} R")

        # Save weights to file for bot use
        with open('impulse_fib_trader/config/coin_weights.json', 'w') as f:
            json.dump(weights, f, indent=4)
        print("\n✅ Weights saved to coin_weights.json")

if __name__ == "__main__":
    bt = WeightedBacktest()
    asyncio.run(bt.run_weighted_backtest())
