import json

def fix_history_status():
    history_file = 'impulse_fib_trader/trade_history.json'
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    for trade in history:
        if 'exit_price' in trade:
            trade['status'] = 'CLOSED'
            
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=4)
    print("Статусы в истории исправлены на CLOSED.")

if __name__ == "__main__":
    fix_history_status()
