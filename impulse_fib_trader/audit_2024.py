
import json
import numpy as np

with open('full_backtest_report_2024.json', 'r') as f:
    report = json.load(f)

trades = report['all_trades']
if not trades:
    print("No trades found.")
    exit()

wins = [t for t in trades if t['result_r'] > 0]
losses = [t for t in trades if t['result_r'] < 0]

total_trades = len(trades)
wr = len(wins) / total_trades
gross_profit = sum(t['result_r'] for t in wins)
gross_loss = abs(sum(t['result_r'] for t in losses))
net_profit = gross_profit - gross_loss
pf = gross_profit / gross_loss if gross_loss > 0 else 0

print(f"--- АУДИТ БЭКТЕСТА 2024 ---")
print(f"Всего сделок: {total_trades}")
print(f"Винрейт (WR): {wr:.2%}")
print(f"Прибыльных (Wins): {len(wins)}")
print(f"Убыточных (Losses): {len(losses)}")
print(f"Валовая прибыль (Gross Profit): {gross_profit:.1f} R")
print(f"Валовый убыток (Gross Loss): {gross_loss:.1f} R")
print(f"ЧИСТЫЙ ПРОФИТ (Net Profit): {net_profit:.1f} R")
print(f"Профит фактор (PF): {pf:.2f}")

# Анализ по монетам
coin_stats = {}
for t in trades:
    s = t['symbol']
    if s not in coin_stats: coin_stats[s] = []
    coin_stats[s].append(t['result_r'])

print("\n--- ТОП-5 МОНЕТ (по профиту) ---")
sorted_coins = sorted(coin_stats.items(), key=lambda x: sum(x[1]), reverse=True)
for s, tr in sorted_coins[:5]:
    w = len([x for x in tr if x > 0])
    l = len([x for x in tr if x < 0])
    print(f"{s:<12} | Profit: {sum(tr):>6.1f} R | WR: {w/(w+l):.1%}")

print("\n--- ХУДШИЕ-5 МОНЕТ ---")
for s, tr in sorted_coins[-5:]:
    w = len([x for x in tr if x > 0])
    l = len([x for x in tr if x < 0])
    print(f"{s:<12} | Profit: {sum(tr):>6.1f} R | WR: {w/(w+l):.1%}")
