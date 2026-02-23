import pandas as pd
import matplotlib.pyplot as plt
import os

class Reporter:
    @staticmethod
    def print_full_report(metrics: dict, name: str = "Strategy"):
        print(f"
--- {name} Report ---")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"{k:<20}: {v:.4f}")
            else:
                print(f"{k:<20}: {v}")

    @staticmethod
    def save_equity_curve(results: pd.DataFrame, path: str):
        """
        Saves equity curve plot to a file (if possible in this environment).
        """
        if results.empty:
            return
            
        equity = results['r_multiple'].cumsum()
        # In a real environment we would save the plot
        # For CLI, we can just log the final equity
        print(f"Final Equity (R): {equity.iloc[-1]:.2f}")
