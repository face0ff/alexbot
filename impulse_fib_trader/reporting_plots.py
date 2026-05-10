
import os
import matplotlib
matplotlib.use('Agg')
import mplfinance as mpf
import pandas as pd
import numpy as np
from datetime import datetime

def plot_backtest_trade(df, trade, output_path):
    """
    Рисует подробный график сделки из бэктеста с входом, выходом и результатом.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty")

    try:
        entry_time = pd.to_datetime(trade['entry_time'])
        exit_time = pd.to_datetime(trade['exit_time'])
        
        entry_idx = df.index.get_indexer([entry_time], method='nearest')[0]
        exit_idx = df.index.get_indexer([exit_time], method='nearest')[0]

        start_idx = max(0, entry_idx - 40)
        end_idx = min(len(df), exit_idx + 20)
        
        plot_df = df.iloc[start_idx:end_idx].copy()
        if plot_df.empty: return None
        plot_df.index = pd.to_datetime(plot_df.index)
        
        added_plots = []
        if 'ema200' in df.columns:
            ema_data = df['ema200'].iloc[start_idx:end_idx]
            if not ema_data.empty and not ema_data.isna().all():
                added_plots.append(mpf.make_addplot(ema_data, color='blue', width=0.7, alpha=0.3))

        entry_markers = pd.Series(np.nan, index=plot_df.index)
        if entry_time in entry_markers.index:
            entry_markers.loc[entry_time] = trade['entry_price'] * 0.992
        if not entry_markers.isna().all():
            added_plots.append(mpf.make_addplot(entry_markers, type='scatter', markersize=120, marker='^', color='orange'))

        exit_markers = pd.Series(np.nan, index=plot_df.index)
        exit_color = 'green' if trade['result_r'] > 0 else 'red'
        if exit_time in exit_markers.index:
            exit_markers.loc[exit_time] = trade['exit_price']
        if not exit_markers.isna().all():
            added_plots.append(mpf.make_addplot(exit_markers, type='scatter', markersize=120, marker='o', color=exit_color))

        duration = (exit_time - entry_time).total_seconds() / 3600
        res_text = "PROFIT" if trade['result_r'] > 0 else "LOSS"
        info_str = (f"{trade['symbol']} | {res_text} ({trade['result_r']:.1f}R)\n"
                    f"ML Prob: {trade['prob']:.1%}\n"
                    f"Time: {duration:.1f}h")

        mpf.plot(
            plot_df, type='candle', addplot=added_plots,
            hlines=dict(hlines=[trade['sl'], trade['tp']], colors=['red', 'green'], linestyle='--', alpha=0.4),
            style='charles', title=f"\n{info_str}", savefig=output_path, tight_layout=True, figratio=(12, 8)
        )
        return output_path
    except Exception:
        return None

def plot_tas_pattern(df, pattern, symbol, output_path):
    """
    Версия для живых сигналов (совместима с ботом).
    """
    if df is None or df.empty: return None
    
    try:
        idx = pattern['entry_idx']
        start_idx = max(0, idx - 60)
        end_idx = min(len(df), idx + 15)
        
        plot_df = df.iloc[start_idx:end_idx].copy()
        if plot_df.empty: return None
        plot_df.index = pd.to_datetime(plot_df.index)
        
        added_plots = []
        
        # EMA200
        if 'ema200' in df.columns:
            ema_data = df['ema200'].iloc[start_idx:end_idx]
            if not ema_data.empty and not ema_data.isna().all():
                added_plots.append(mpf.make_addplot(ema_data, color='blue', width=0.8, alpha=0.4))
        
        # Маркер входа
        markers = pd.Series(np.nan, index=plot_df.index)
        entry_time = df.index[idx]
        if entry_time in markers.index:
            markers.loc[entry_time] = plot_df.loc[entry_time, 'low'] * 0.995
        
        if not markers.isna().all():
            added_plots.append(mpf.make_addplot(markers, type='scatter', markersize=100, marker='^', color='orange'))
        
        # Линии уровней
        h_vals = [pattern['sl'], pattern['entry_price']]
        h_cols = ['red', 'green']
        
        # Если есть уровни импульса
        if 'p0' in pattern and 'p1' in pattern:
            h_vals.extend([pattern['p0'], pattern['p1']])
            h_cols.extend(['gray', 'gray'])

        mpf.plot(
            plot_df, type='candle', addplot=added_plots,
            hlines=dict(hlines=h_vals, colors=h_cols, linestyle='--', alpha=0.5),
            style='charles', title=f"\nTAS Signal: {symbol} (Prob: {pattern.get('prob', 0.5):.1%})",
            savefig=output_path, tight_layout=True, figratio=(12, 7)
        )
        return output_path
    except Exception:
        return None
