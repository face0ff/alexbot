import ccxt
import json
import os
from datetime import datetime
from config.config import BINANCE_API_KEY, BINANCE_API_SECRET

def sync_trades():
    exchange = ccxt.binance({
        'apiKey': BINANCE_API_KEY,
        'secret': BINANCE_API_SECRET,
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })

    base_dir = os.path.dirname(os.path.abspath(__file__))
    state_file = os.path.join(base_dir, 'trade_state.json')
    history_file = os.path.join(base_dir, 'trade_history.json')

    if not os.path.exists(state_file):
        print("State file not found.")
        return

    with open(state_file, 'r') as f:
        try:
            active_trades = json.load(f)
        except:
            print("Error loading state file.")
            return

    if not active_trades:
        print("No active trades in state.")
        return

    print(f"Checking {len(active_trades)} active trades...")
    
    # Get current balance once for all assets
    try:
        balance = exchange.fetch_balance()
        total_balances = balance['total']
    except Exception as e:
        print(f"Error fetching balance: {e}")
        return

    updated_active = []
    closed_count = 0

    for trade in active_trades:
        symbol = trade['symbol']
        base_asset = symbol.split('/')[0]
        
        actual_qty = total_balances.get(base_asset, 0.0)
        
        # If actual quantity is significantly less than expected, consider it closed manually
        # Using 10% threshold to account for partial fills/dust
        if actual_qty < trade['amount'] * 0.1: 
            print(f"Trade for {symbol} seems CLOSED manually or via OCO (Expected ~{trade['amount']}, Found {actual_qty})")
            
            # Try to find the actual exit price from recent trades
            exit_price = None
            exit_time = None
            try:
                # Fetch more trades to be sure (limit=50)
                my_trades = exchange.fetch_my_trades(symbol, limit=50)
                # Find the last 'sell' trade
                for t in reversed(my_trades):
                    if t['side'] == 'sell':
                        exit_price = t['price']
                        exit_time = t['datetime']
                        break
                
                if exit_price:
                    trade['exit_price'] = exit_price
                    trade['exit_time'] = exit_time
                    trade['exit_reason'] = "MANUAL_EXIT_SYNC"
                    trade['pnl_usdt'] = (exit_price - trade['real_entry_price']) * trade['amount']
                    trade['status'] = 'CLOSED'
                    print(f"   ✅ Found exit at {exit_price}, PnL: {trade['pnl_usdt']:.2f} USDT")
                else:
                    # Fallback to current market price if no sell trade found in recent history
                    ticker = exchange.fetch_ticker(symbol)
                    exit_price = ticker['last']
                    trade['exit_price'] = exit_price
                    trade['exit_time'] = datetime.now().isoformat()
                    trade['exit_reason'] = "MANUAL_EXIT_UNKNOWN_PRICE_SYNC"
                    trade['pnl_usdt'] = (exit_price - trade['real_entry_price']) * trade['amount']
                    trade['status'] = 'CLOSED'
                    print(f"   ⚠️ No sell trade found. Using market price {exit_price} for PnL.")

                save_to_history(history_file, trade)
                closed_count += 1
            except Exception as e:
                print(f"   ❌ Error processing {symbol}: {e}")
                updated_active.append(trade)
        else:
            print(f"Trade for {symbol} is still OPEN (Qty: {actual_qty:.6g})")
            updated_active.append(trade)

    # Save updated state
    with open(state_file, 'w') as f:
        json.dump(updated_active, f, indent=4)
    
    print(f"\n--- Sync complete. Closed {closed_count} trades. ---")

def save_to_history(history_file, trade_data):
    history = []
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            try:
                history = json.load(f)
            except:
                pass
    history.append(trade_data)
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=4)

if __name__ == "__main__":
    sync_trades()
