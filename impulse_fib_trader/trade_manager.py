import ccxt
import json
import os
import logging
from datetime import datetime, timedelta
from config.config import BINANCE_API_KEY, BINANCE_API_SECRET

logger = logging.getLogger(__name__)

class TradeManager:
    def __init__(self, state_file='trade_state.json', history_file='trade_history.json'):
        # Используем абсолютные пути для надежности
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.state_file = os.path.join(base_dir, state_file)
        self.history_file = os.path.join(base_dir, history_file)
        
        self.exchange = ccxt.binance({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_API_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        self.active_trades = self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
            except Exception:
                return []
        return []

    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.active_trades, f, indent=4)

    def _save_history(self, trade_data):
        history = []
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
            except Exception:
                pass
        history.append(trade_data)
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=4)

    def get_cooldown_symbols(self, hours=4):
        """Возвращает список символов, по которым недавно были сделки."""
        cooldown_symbols = []
        if not os.path.exists(self.history_file):
            return cooldown_symbols
            
        try:
            with open(self.history_file, 'r') as f:
                history = json.load(f)
            
            cutoff = datetime.now() - timedelta(hours=hours)
            for t in history:
                entry_time = datetime.fromisoformat(t['entry_time'])
                if entry_time > cutoff:
                    if t['symbol'] not in cooldown_symbols:
                        cooldown_symbols.append(t['symbol'])
        except Exception as e:
            logger.error(f"Error checking cooldown: {e}")
            
        return cooldown_symbols

    def get_balance(self, currency='USDT'):
        try:
            balance = self.exchange.fetch_balance()
            return balance['free'].get(currency, 0.0)
        except Exception as e:
            logger.error(f"Ошибка баланса: {e}")
            return 0.0

    def enter_trade(self, symbol, entry_price, sl, tp, side, amount_usdt=10):
        if any(t['symbol'] == symbol for t in self.active_trades):
            return False, f"Сделка по {symbol} уже открыта."

        try:
            # 1. Покупка по маркету
            approx_amount = amount_usdt / entry_price
            params = {'quoteOrderQty': self.exchange.cost_to_precision(symbol, amount_usdt)}
            order = self.exchange.create_order(symbol, 'market', 'buy', approx_amount, None, params)
            
            real_entry = order['average'] if order.get('average') else order.get('price', entry_price)
            filled_amount = order['filled'] if order.get('filled') else order['amount']

            new_trade = {
                'symbol': symbol,
                'real_entry_price': real_entry,
                'sl': sl,
                'tp': tp,
                'amount': filled_amount,
                'entry_time': datetime.now().isoformat(),
                'status': 'OPEN'
            }
            
            self.active_trades.append(new_trade)
            self._save_state()
            return True, f"✅ Куплено {symbol} по {real_entry:.6g}. Бот следит за ценой для выхода."

        except Exception as e:
            logger.error(f"Trade entry failed: {e}")
            return False, str(e)

    def check_trade_exit(self):
        closed_messages = []
        remaining_trades = []

        self.exchange.load_markets()

        try:
            balance = self.exchange.fetch_balance()
            total_balances = balance['total']
        except Exception as e:
            logger.error(f"Не удалось получить баланс: {e}")
            return []

        for trade in self.active_trades:
            try:
                symbol = trade['symbol']
                base_asset = symbol.split('/')[0]
                actual_qty = total_balances.get(base_asset, 0.0)
                
                if actual_qty < trade['amount'] * 0.1:
                    logger.info(f"Обнаружено внешнее закрытие для {symbol}.")
                    exit_price = None
                    try:
                        my_trades = self.exchange.fetch_my_trades(symbol, limit=20)
                        for t in reversed(my_trades):
                            if t['side'] == 'sell':
                                exit_price = t['price']
                                break
                    except Exception: pass

                    if exit_price:
                        self._process_exit(trade, exit_price, "EXTERNAL_EXIT", closed_messages)
                    else:
                        ticker = self.exchange.fetch_ticker(symbol)
                        self._process_exit(trade, ticker['last'], "EXTERNAL_EXIT_UNKNOWN", closed_messages)
                    continue

                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                exit_reason = None
                if current_price >= trade['tp']:
                    exit_reason = "TAKE_PROFIT"
                elif current_price <= trade['sl']:
                    exit_reason = "STOP_LOSS"

                if exit_reason:
                    try:
                        open_orders = self.exchange.fetch_open_orders(symbol)
                        for order in open_orders:
                            self.exchange.cancel_order(order['id'], symbol)
                    except Exception: pass

                    try:
                        balance = self.exchange.fetch_balance()
                        free_qty = balance['free'].get(base_asset, 0.0)
                        qty_to_sell = self.exchange.amount_to_precision(symbol, free_qty)
                        
                        if free_qty > 0:
                            order = self.exchange.create_order(symbol, 'market', 'sell', qty_to_sell)
                            exit_p = order.get('average', current_price)
                            self._process_exit(trade, exit_p, exit_reason, closed_messages)
                        else:
                            self._process_exit(trade, current_price, "MANUAL_EXIT_SYNC", closed_messages)
                    except Exception as e:
                        logger.error(f"Sell failed for {symbol}: {e}")
                        remaining_trades.append(trade)
                else:
                    remaining_trades.append(trade)

            except Exception as e:
                logger.error(f"Error checking {trade['symbol']}: {e}")
                remaining_trades.append(trade)

        if len(remaining_trades) != len(self.active_trades):
            self.active_trades = remaining_trades
            self._save_state()
        return closed_messages

    def manual_market_exit(self, symbol):
        trade = next((t for t in self.active_trades if t['symbol'] == symbol), None)
        if not trade:
            return False, f"Сделка по {symbol} не найдена."

        try:
            self.exchange.load_markets()
            try:
                open_orders = self.exchange.fetch_open_orders(symbol)
                for order in open_orders:
                    self.exchange.cancel_order(order['id'], symbol)
            except Exception: pass

            balance = self.exchange.fetch_balance()
            base_asset = symbol.split('/')[0]
            free_qty = balance['free'].get(base_asset, 0.0)
            
            if free_qty > 0:
                qty_to_sell = self.exchange.amount_to_precision(symbol, free_qty)
                order = self.exchange.create_order(symbol, 'market', 'sell', qty_to_sell)
                exit_price = order.get('average', order.get('price'))
            else:
                ticker = self.exchange.fetch_ticker(symbol)
                exit_price = ticker['last']

            # _process_exit сам добавит в историю и изменит статус
            self._process_exit(trade, exit_price, "MANUAL_FIX_PROFIT", [])
            
            # Удаляем из активных и сохраняем стейт
            self.active_trades = [t for t in self.active_trades if t['symbol'] != symbol]
            self._save_state()
            
            return True, f"✅ {symbol} продан по {exit_price:.6g}."
        except Exception as e:
            return False, str(e)

    def _process_exit(self, trade, exit_price, reason, messages_list):
        pnl = (exit_price - trade['real_entry_price']) * trade['amount']
        trade['exit_price'] = exit_price
        trade['exit_time'] = datetime.now().isoformat()
        trade['exit_reason'] = reason
        trade['pnl_usdt'] = pnl
        trade['status'] = 'CLOSED'
        self._save_history(trade)
        messages_list.append(f"🔔 **Сделка закрыта ({reason})!**\nПара: {trade['symbol']}\nPnL: {pnl:.2f} USDT")

    def get_stats(self):
        if not os.path.exists(self.history_file): return "История пуста."
        try:
            with open(self.history_file, 'r') as f:
                history = json.load(f)
            pnl = sum([t.get('pnl_usdt', 0) for t in history])
            return f"📊 **Статистика**\nСделок: {len(history)}\nПрофит: {pnl:.2f} USDT"
        except Exception: return "Ошибка статистики."
