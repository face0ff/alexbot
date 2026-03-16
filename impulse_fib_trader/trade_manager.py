import ccxt
import json
import os
import logging
from datetime import datetime
from config.config import BINANCE_API_KEY, BINANCE_API_SECRET

logger = logging.getLogger(__name__)

class TradeManager:
    def __init__(self, state_file='trade_state.json', history_file='trade_history.json'):
        self.state_file = state_file
        self.history_file = history_file
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

            # Сохраняем данные сделки (БЕЗ OCO)
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

        # Загружаем рынки для точности округления
        self.exchange.load_markets()

        # Предварительно загружаем баланс, чтобы не дергать его в цикле
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
                
                # 1. Проверка на ручное закрытие (если количество монет < 10% от ожидаемого)
                if actual_qty < trade['amount'] * 0.1:
                    logger.info(f"Обнаружено внешнее закрытие для {symbol}. Синхронизация PnL.")
                    exit_price = None
                    exit_time = None
                    
                    try:
                        # Ищем последнюю сделку на продажу в истории
                        my_trades = self.exchange.fetch_my_trades(symbol, limit=20)
                        for t in reversed(my_trades):
                            if t['side'] == 'sell':
                                exit_price = t['price']
                                exit_time = t['datetime']
                                break
                    except Exception as e:
                        logger.error(f"Ошибка получения сделок для {symbol}: {e}")

                    if exit_price:
                        self._process_exit(trade, exit_price, "EXTERNAL_EXIT", closed_messages)
                    else:
                        # Фолбэк на текущую цену, если сделка не найдена
                        ticker = self.exchange.fetch_ticker(symbol)
                        self._process_exit(trade, ticker['last'], "EXTERNAL_EXIT_UNKNOWN_PRICE", closed_messages)
                    continue

                # 2. Обычная проверка по TP/SL
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                exit_reason = None
                if current_price >= trade['tp']:
                    exit_reason = "TAKE_PROFIT"
                elif current_price <= trade['sl']:
                    exit_reason = "STOP_LOSS"

                if exit_reason:
                    logger.info(f"Триггер {exit_reason} для {symbol}. Цена: {current_price}")
                    
                    # Отменяем любые открытые ордера (OCO)
                    try:
                        open_orders = self.exchange.fetch_open_orders(symbol)
                        for order in open_orders:
                            self.exchange.cancel_order(order['id'], symbol)
                    except Exception: pass

                    # Продаем ВЕСЬ свободный остаток по рынку
                    try:
                        # Снова берем свежий баланс для точности перед продажей
                        balance = self.exchange.fetch_balance()
                        free_qty = balance['free'].get(base_asset, 0.0)
                        
                        min_qty = self.exchange.markets[symbol]['limits']['amount']['min'] if symbol in self.exchange.markets else 0.0
                        
                        if free_qty <= (min_qty if min_qty else 0.0):
                            logger.info(f"Баланс {symbol} пуст. Вероятно, закрыто вручную прямо сейчас.")
                            self._process_exit(trade, current_price, "MANUAL_EXIT_SYNC", closed_messages)
                            continue

                        qty_to_sell = self.exchange.amount_to_precision(symbol, free_qty)
                        order = self.exchange.create_order(symbol, 'market', 'sell', qty_to_sell)
                        exit_price = order.get('average', order.get('price', current_price))
                        self._process_exit(trade, exit_price, exit_reason, closed_messages)
                    except Exception as e:
                        if "insufficient balance" in str(e).lower():
                            logger.warning(f"Недостаточно средств для {symbol} при выходе. Считаем закрытым.")
                            self._process_exit(trade, current_price, "MANUAL_EXIT_SYNC", closed_messages)
                        else:
                            logger.error(f"Рыночная продажа НЕ УДАЛАСЬ для {symbol}: {e}")
                            remaining_trades.append(trade)
                else:
                    remaining_trades.append(trade)

            except Exception as e:
                logger.error(f"Ошибка проверки выхода для {trade['symbol']}: {e}")
                remaining_trades.append(trade)

        if len(remaining_trades) != len(self.active_trades):
            self.active_trades = remaining_trades
            self._save_state()
        return closed_messages

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
