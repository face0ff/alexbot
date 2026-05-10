import ccxt
import json
import os
import logging
from datetime import datetime, timedelta
from config.config import BINANCE_API_KEY, BINANCE_API_SECRET

logger = logging.getLogger(__name__)

class TradeManager:
    def __init__(self, state_file='trade_state.json', history_file='trade_history.json'):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.state_file = os.path.join(base_dir, state_file)
        self.history_file = os.path.join(base_dir, history_file)
        self.FEE_RATE = 0.001 
        
        # --- GRID STRATEGIES ---
        self.STRATEGIES = {
            'MARTI_CONS': {
                'weights': [1, 1.5, 2.5],
                'dists': [0, 0.45, 0.8] # Multiplier of initial risk
            },
            'FIB_GRID': {
                'weights': [1, 2, 3],
                'dists': [0, 0.38, 0.62]
            }
        }
        self.current_strategy = 'MARTI_CONS'
        self.MAX_ACTIVE_TRADES = 1 # Ограничение количества одновременно открытых сделок
        
        self.exchange = ccxt.binance({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_API_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        try:
            self.exchange.load_markets()
        except Exception as e:
            logger.error(f"Failed to load markets: {e}")
            
        self.active_trades = self._load_state()

    def set_strategy(self, name):
        if name in self.STRATEGIES:
            self.current_strategy = name
            return True
        return False

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    # Support legacy state and new current_strategy
                    if isinstance(data, dict):
                        self.current_strategy = data.get('strategy', 'MARTI_CONS')
                        return data.get('trades', [])
                    return data if isinstance(data, list) else []
            except Exception as e:
                logger.error(f"Error loading state: {e}")
                return []
        return []

    def _save_state(self):
        try:
            with open(self.state_file, 'w') as f:
                json.dump({
                    'strategy': self.current_strategy,
                    'trades': self.active_trades
                }, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _save_history(self, trade_data):
        history = []
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f: history = json.load(f)
            except: pass
        history.append(trade_data)
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=4)

    def get_balance(self, currency='USDT'):
        try:
            balance = self.exchange.fetch_balance()
            return balance['free'].get(currency, 0.0)
        except Exception as e:
            logger.error(f"Balance error: {e}")
            return 0.0

    def enter_trade(self, symbol, entry_price, sl, tp, side='buy', amount_usdt=10):
        if any(t['symbol'] == symbol for t in self.active_trades):
            return False, "Already in trade"
            
        if len(self.active_trades) >= self.MAX_ACTIVE_TRADES:
            return False, f"Limit reached ({self.MAX_ACTIVE_TRADES} trade active)"

        risk_unit = entry_price - sl
        if risk_unit <= 0: return False, "Invalid SL for Long"

        strat = self.STRATEGIES[self.current_strategy]
        
        try:
            # Entry 1 (Market)
            first_weight = strat['weights'][0]
            first_qty_usdt = amount_usdt * first_weight
            params = {'quoteOrderQty': self.exchange.cost_to_precision(symbol, first_qty_usdt)}
            order = self.exchange.create_order(symbol, 'market', 'buy', None, None, params)
            
            real_entry = order.get('average') or order.get('price') or entry_price
            filled_qty = order.get('filled') or order.get('amount') or 0
            
            if filled_qty == 0: return False, "Order failed"

            # Create Sub-orders for Grid (Limit Orders)
            grid_orders = []
            for i in range(1, len(strat['weights'])):
                price = real_entry - risk_unit * strat['dists'][i]
                qty_usdt = amount_usdt * strat['weights'][i]
                
                # We calculate qty based on price since it's a limit order
                qty = qty_usdt / price
                try:
                    limit_order = self.exchange.create_order(symbol, 'limit', 'buy', 
                                                           self.exchange.amount_to_precision(symbol, qty), 
                                                           self.exchange.price_to_precision(symbol, price))
                    grid_orders.append({
                        'id': limit_order['id'],
                        'price': price,
                        'weight': strat['weights'][i],
                        'filled': False
                    })
                except Exception as le:
                    logger.error(f"Limit order failed for {symbol}: {le}")

            new_trade = {
                'symbol': symbol,
                'strategy': self.current_strategy,
                'initial_risk': risk_unit,
                'sl': float(sl),
                'tp': float(tp), # Initial TP
                'entries': [{
                    'price': float(real_entry),
                    'qty': float(filled_qty),
                    'weight': first_weight,
                    'time': datetime.now().isoformat()
                }],
                'grid_orders': grid_orders,
                'status': 'OPEN'
            }
            
            self.active_trades.append(new_trade)
            self._save_state()
            return True, f"Entered {symbol} with {self.current_strategy} grid"

        except Exception as e:
            logger.error(f"Execution error: {e}")
            return False, str(e)

    def check_trade_exit(self):
        closed_messages = []
        remaining_trades = []
        
        for trade in self.active_trades:
            if 'grid_orders' not in trade:
                logger.warning(f"Legacy trade detected for {trade.get('symbol')}, skipping grid check.")
                remaining_trades.append(trade)
                continue

            try:
                # 1. Fetch data for RSI calculation
                ohlcv = self.exchange.fetch_ohlcv(trade['symbol'], timeframe='15m', limit=30)
                df_exit = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Simple RSI Calculation
                delta = df_exit['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df_exit['rsi'] = 100 - (100 / (1 + rs))
                curr_rsi = df_exit['rsi'].iloc[-1]
                current_price = df_exit['close'].iloc[-1]

                # 2. Check Grid Fills
                for g_order in trade['grid_orders']:
                    if not g_order['filled']:
                        o_status = self.exchange.fetch_order(g_order['id'], trade['symbol'])
                        if o_status['status'] == 'closed':
                            g_order['filled'] = True
                            trade['entries'].append({
                                'price': o_status['average'] or o_status['price'],
                                'qty': o_status['filled'],
                                'weight': g_order['weight'],
                                'time': datetime.now().isoformat()
                            })
                            logger.info(f"✅ GRID FILL: {trade['symbol']} level filled!")

                # 3. Update Stats
                total_qty = sum(e['qty'] for e in trade['entries'])
                total_cost = sum(e['price'] * e['qty'] for e in trade['entries'])
                avg_price = total_cost / total_qty
                
                # Dynamic TP Logic
                if len(trade['entries']) > 1:
                    new_tp = avg_price + trade['initial_risk'] * 0.3
                    trade['tp'] = new_tp

                # --- RSI PROFIT GUARD LOGIC ---
                curr_pnl_pct = (current_price / avg_price - 1) * 100
                if 'rsi_armed' not in trade: trade['rsi_armed'] = False
                
                exit_reason = None
                
                # Check for RSI Reversal if in profit
                if curr_pnl_pct > 0.3: # Only if at least slightly in profit
                    if curr_rsi >= 70: trade['rsi_armed'] = True
                    
                    if trade['rsi_armed'] and curr_rsi < 65:
                        exit_reason = f"RSI_GUARD (RSI:{curr_rsi:.1f})"
                
                if current_price >= trade['tp']: exit_reason = "TAKE_PROFIT"
                elif current_price <= trade['sl']: exit_reason = "STOP_LOSS"

                if exit_reason:
                    # Sell All
                    sell_order = self.exchange.create_order(trade['symbol'], 'market', 'sell', 
                                                           self.exchange.amount_to_precision(trade['symbol'], total_qty))
                    exit_p = sell_order.get('average') or current_price
                    pnl = (exit_p - avg_price) * total_qty
                    
                    # Cancel remaining grid orders
                    for g_order in trade['grid_orders']:
                        if not g_order['filled']:
                            try: self.exchange.cancel_order(g_order['id'], trade['symbol'])
                            except: pass

                    trade.update({
                        'exit_price': exit_p,
                        'avg_entry': avg_price,
                        'exit_time': datetime.now().isoformat(),
                        'exit_reason': exit_reason,
                        'pnl_usdt': pnl,
                        'status': 'CLOSED'
                    })
                    self._save_history(trade)
                    closed_messages.append(f"✅ <b>CLOSED: {trade['symbol']}</b> ({trade['strategy']})\nReason: {exit_reason}\nPnL: <b>{pnl:+.2f} USDT</b>")
                else:
                    remaining_trades.append(trade)
            except Exception as e:
                logger.error(f"Check error {trade['symbol']}: {e}")
                remaining_trades.append(trade)

        self.active_trades = remaining_trades
        self._save_state()
        return closed_messages

    def manual_market_exit(self, symbol):
        trade_idx = next((i for i, t in enumerate(self.active_trades) if t['symbol'] == symbol), None)
        if trade_idx is None: return False, "Trade not found"
        
        trade = self.active_trades.pop(trade_idx)
        try:
            total_qty = sum(e['qty'] for e in trade['entries'])
            self.exchange.create_order(symbol, 'market', 'sell', self.exchange.amount_to_precision(symbol, total_qty))
            
            # Cancel grid orders
            for g_order in trade['grid_orders']:
                if not g_order['filled']:
                    try: self.exchange.cancel_order(g_order['id'], symbol)
                    except: pass

            trade.update({'status': 'MANUAL_CLOSED', 'exit_time': datetime.now().isoformat()})
            self._save_history(trade)
            self._save_state()
            return True, f"Manual closed {symbol}"
        except Exception as e:
            self.active_trades.append(trade)
            return False, str(e)

    def get_stats(self):
        if not os.path.exists(self.history_file): return "No history yet."
        try:
            with open(self.history_file, 'r') as f: history = json.load(f)
            pnl = sum([t.get('pnl_usdt', 0) for t in history])
            return f"📊 <b>Total PnL: {pnl:.2f} USDT</b>\nTrades: {len(history)}"
        except: return "Error loading stats."
