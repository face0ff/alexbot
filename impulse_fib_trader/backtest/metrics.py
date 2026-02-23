import pandas as pd
import numpy as np
from typing import Dict

class MetricsCalculator:
    @staticmethod
    def calculate(df: pd.DataFrame) -> Dict:
        """
        Calculates key trading metrics from backtest results.
        """
        if df.empty:
            return {}
            
        r_multiples = df['r_multiple']
        win_rate = len(r_multiples[r_multiples > 0]) / len(r_multiples)
        expectancy = r_multiples.mean()
        
        # Profit Factor
        gross_profit = r_multiples[r_multiples > 0].sum()
        gross_loss = abs(r_multiples[r_multiples < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Drawdown
        equity_curve = r_multiples.cumsum()
        running_max = equity_curve.cummax()
        drawdown = equity_curve - running_max
        max_drawdown = drawdown.min()
        
        # Sharpe (simplified for R-multiples)
        sharpe = r_multiples.mean() / r_multiples.std() if r_multiples.std() > 0 else 0
        
        return {
            'total_trades': len(df),
            'win_rate': win_rate,
            'expectancy': expectancy,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'net_profit_r': r_multiples.sum()
        }
