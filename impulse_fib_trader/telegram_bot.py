
import asyncio
import logging
import sys
import os
import json
import torch
import pandas as pd
import numpy as np
import mplfinance as mpf
from datetime import datetime, timedelta
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, FSInputFile, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from config.config import BOT_TOKEN, TELEGRAM_PRIVATE_CHAT_ID
from data.fetcher import DataFetcher
from data.cleaner import DataCleaner
from pattern.market_structure_smc import SMCMarketStructure
from trade_manager import TradeManager
from ml.transformer_smc import SMCTransformer
from ml.lnn_model import LiquidNet

# --- New: Import Weight Calculator Logic (Modular) ---
try:
    from calculate_coin_weights import WeightCalculator
except ImportError:
    class WeightCalculator:
        async def run(self): return "Error: Calculator module not found."

# State management
STATE_FILE = "telegram_state.json"
WEIGHTS_FILE = "impulse_fib_trader/config/coin_weights_2024_2025.json"

def get_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f: return json.load(f)
        except: pass
    return {'chat_id': TELEGRAM_PRIVATE_CHAT_ID, 'use_weights': False}

def save_state(state):
    with open(STATE_FILE, 'w') as f: json.dump(state, f, indent=4)

# Safe Logger
class SafeLogger(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            self.stream.write(msg.encode('ascii', 'replace').decode() + self.terminator)
            self.flush()
        except Exception: self.handleError(record)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[SafeLogger()])
logger = logging.getLogger(__name__)

# Init
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
trade_manager = TradeManager()
fetcher = DataFetcher()
cleaner = DataCleaner()
smc = SMCMarketStructure(window=3)

# Load Models
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WHITELIST_PATH = os.path.join(PROJECT_ROOT, 'impulse_fib_trader', 'config', 'whitelist.json')

transformer_model = SMCTransformer(input_dim=4, d_model=64, nhead=4, num_layers=3)
transformer_path = os.path.join(PROJECT_ROOT, 'smc_transformer_v1.pth')
if os.path.exists(transformer_path):
    transformer_model.load_state_dict(torch.load(transformer_path, map_location=torch.device('cpu')))
    transformer_model.eval()

lnn_model = LiquidNet(in_features=5, hidden_features=128, out_features=1)
lnn_path = os.path.join(PROJECT_ROOT, 'lnn_filter.pth')
if os.path.exists(lnn_path):
    lnn_model.load_state_dict(torch.load(lnn_path, map_location=torch.device('cpu')))
    lnn_model.eval()

auto_scan_active = False

def get_kb():
    state = get_state()
    w_status = "ON" if state.get('use_weights') else "OFF"
    return ReplyKeyboardMarkup(keyboard=[
        [KeyboardButton(text="🚀 START ENGINE"), KeyboardButton(text="🛑 STOP ENGINE")],
        [KeyboardButton(text=f"📊 Weights: {w_status}"), KeyboardButton(text="🔄 Refresh Weights")],
        [KeyboardButton(text="📊 Stats"), KeyboardButton(text="⚙️ Strategy")],
        [KeyboardButton(text="ℹ️ Status")]
    ], resize_keyboard=True)

async def notify(text, photo=None):
    state = get_state()
    chat_id = state.get('chat_id')
    try:
        if photo: await bot.send_photo(chat_id, FSInputFile(photo), caption=text, parse_mode="HTML")
        else: await bot.send_message(chat_id, text, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Notify error: {e}")

def plot_advanced_trade(symbol, is_new=True, trade_data=None):
    """
    Генерирует продвинутый график с RSI и всеми уровнями.
    is_new: если True, рисуем сигнал входа. False - рисуем активную сделку.
    """
    try:
        df = fetcher.fetch_ohlcv(symbol, '15m', (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d'))
        df = cleaner.calculate_indicators(df)
        
        # Берем последние 60 свечей
        plot_df = df.tail(60).copy()
        plot_df.index = pd.to_datetime(plot_df.index)
        
        apds = []
        # 1. RSI Panel
        apds.append(mpf.make_addplot(plot_df['rsi'], panel=1, color='purple', ylabel='RSI'))
        apds.append(mpf.make_addplot([70]*len(plot_df), panel=1, color='red', linestyle='--'))
        apds.append(mpf.make_addplot([30]*len(plot_df), panel=1, color='green', linestyle='--'))
        
        hlines_prices = []
        hlines_colors = []

        if is_new:
            # Для нового сигнала данные берем из аргументов (упрощенно для примера)
            # В реальности сюда нужно передать конкретные sl/tp
            pass
        else:
            # Для активной сделки рисуем средний вход и сетку
            total_qty = sum(e['qty'] for e in trade_data['entries'])
            total_cost = sum(e['price'] * e['qty'] for e in trade_data['entries'])
            avg_p = total_cost / total_qty
            
            hlines_prices = [avg_p, trade_data['sl'], trade_data['tp']]
            hlines_colors = ['orange', 'red', 'green']
            
            # Добавляем уровни сетки (усреднения)
            for g in trade_data.get('grid_orders', []):
                if not g['filled']:
                    hlines_prices.append(g['price'])
                    hlines_colors.append('gray')

        file_path = f"status_{symbol.replace('/', '_')}.png"
        mpf.plot(plot_df, type='candle', style='charles', 
                 addplot=apds, 
                 hlines=dict(hlines=hlines_prices, colors=hlines_colors, linestyle='--'),
                 title=f"{symbol} Status",
                 panel_ratios=(2, 1),
                 savefig=file_path)
        return file_path
    except Exception as e:
        logger.error(f"Plotting error: {e}")
        return None

# --- Handlers ---

@dp.message(F.text.startswith("📊 Weights:"))
async def toggle_weights(message: types.Message):
    state = get_state()
    state['use_weights'] = not state.get('use_weights', False)
    save_state(state)
    await message.answer(f"⚖️ Coin Weighting Filter: <b>{'ON' if state['use_weights'] else 'OFF'}</b>", reply_markup=get_kb(), parse_mode="HTML")

@dp.message(F.text == "🔄 Refresh Weights")
async def refresh_weights_handler(message: types.Message):
    await message.answer("⏳ Recalculating weights based on last 90 days... Please wait (approx 1-2 min).")
    try:
        calc = WeightCalculator()
        report = await calc.run() 
        await message.answer(report, parse_mode="HTML")
    except Exception as e:
        await message.answer(f"❌ Error updating weights: {e}")

@dp.message(F.text == "ℹ️ Status")
async def status_msg(message: types.Message):
    state = get_state()
    state['chat_id'] = message.chat.id
    save_state(state)
    
    balance = trade_manager.get_balance()
    active = trade_manager.active_trades
    weight_status = "ON" if state.get('use_weights') else "OFF"
    
    msg = f"ℹ️ <b>SYSTEM STATUS</b>\n──────────────────\n"
    msg += f"🚀 Engine: <b>{'RUNNING' if auto_scan_active else 'STOPPED'}</b>\n"
    msg += f"⚙️ Strategy: <b>{trade_manager.current_strategy}</b>\n"
    msg += f"⚖️ Weight Filter: <b>{weight_status}</b>\n"
    msg += f"💵 Balance: <b>{balance:.2f} USDT</b>\n"
    msg += f"📦 Active Trades: {len(active)}\n"
    await message.answer(msg, parse_mode="HTML")

    if active:
        for t in active:
            symbol = t['symbol']
            # Генерируем график для КАЖДОЙ сделки
            path = plot_advanced_trade(symbol, is_new=False, trade_data=t)
            
            total_qty = sum(e['qty'] for e in t['entries'])
            total_cost = sum(e['price'] * e['qty'] for e in t['entries'])
            avg_entry = total_cost / total_qty
            
            ticker = trade_manager.exchange.fetch_ticker(symbol)
            pnl_pct = (ticker['last'] / avg_entry - 1) * 100
            
            info = f"📊 <b>{symbol}</b>\n"
            info += f"💰 Avg: {avg_entry:.6g} | PnL: <b>{pnl_pct:+.2f}%</b>\n"
            info += f"📦 Fills: {len(t['entries'])}/3"
            
            kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=f"❌ Close {symbol}", callback_data=f"close_{symbol}")]])
            await notify(info, photo=path)
            if path and os.path.exists(path): os.remove(path)

@dp.message(F.text == "🚀 START ENGINE")
async def start_auto(message: types.Message):
    global auto_scan_active
    auto_scan_active = True
    await message.answer("🟢 ENGINE STARTED", reply_markup=get_kb())
    asyncio.create_task(auto_loop())

@dp.message(F.text == "🛑 STOP ENGINE")
async def stop_auto(message: types.Message):
    global auto_scan_active
    auto_scan_active = False
    await message.answer("🔴 ENGINE STOPPED", reply_markup=get_kb())

@dp.message(F.text == "📊 Stats")
async def stats_msg(message: types.Message):
    await message.answer(trade_manager.get_stats(), parse_mode="HTML")

@dp.message(F.text == "⚙️ Strategy")
async def strategy_settings(message: types.Message):
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🛡 Conservative (Marti)", callback_data="set_strat_MARTI_CONS")],
        [InlineKeyboardButton(text="🔥 Aggressive (Fib)", callback_data="set_strat_FIB_GRID")]
    ])
    await message.answer(f"Current Strategy: <b>{trade_manager.current_strategy}</b>\n\nChoose risk profile:", reply_markup=kb, parse_mode="HTML")

@dp.callback_query(F.data.startswith("set_strat_"))
async def cb_set_strat(callback: CallbackQuery):
    strat = callback.data.replace("set_strat_", "")
    if trade_manager.set_strategy(strat):
        trade_manager._save_state()
        await callback.answer(f"Strategy changed to {strat}")
        await callback.message.edit_text(f"✅ Strategy updated to: <b>{strat}</b>", parse_mode="HTML")

@dp.callback_query(F.data.startswith("close_"))
async def cb_close(callback: CallbackQuery):
    symbol = callback.data.replace("close_", "")
    success, res = trade_manager.manual_market_exit(symbol)
    await callback.answer(res)
    await callback.message.edit_text(f"🛑 <b>{symbol}</b>: {res}")

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    save_state({'chat_id': message.chat.id, 'use_weights': False})
    await message.answer("🤖 <b>SMC Bot v2.6.0</b>\nVisual Status + RSI Protection.", reply_markup=get_kb(), parse_mode="HTML")

# --- Logic ---

async def scan_and_trade():
    logger.info("--- [SCAN] START ---")
    state = get_state()
    try:
        with open(WHITELIST_PATH, 'r') as f: symbols = json.load(f)
    except: symbols = ['BTC/USDT', 'ETH/USDT']

    if state.get('use_weights') and os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE, 'r') as f: 
                data = json.load(f)
                weights = data.get('weights', {})
                symbols.sort(key=lambda x: weights.get(x, 0.1), reverse=True)
        except: pass

    for symbol in symbols:
        try:
            if not auto_scan_active: break
            if any(t['symbol'] == symbol for t in trade_manager.active_trades): continue
            
            df = fetcher.fetch_ohlcv(symbol, '15m', (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d'))
            if len(df) < 100: continue
            
            df_4h = fetcher.fetch_ohlcv(symbol, '4h', (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d'))
            df_4h = smc.detect_bos(df_4h)
            bias = df_4h['bos_signal'].replace(0, np.nan).ffill().iloc[-1]

            df = cleaner.calculate_indicators(df)
            df = smc.detect_bos(df)
            
            for l in [0, 1]:
                idx = len(df) - 1 - l
                if df['bos_signal'].iloc[idx] == 1 and bias == 1:
                    seq = np.nan_to_num((df[['open','high','low','close']].values[1:] - df[['open','high','low','close']].values[:-1]) / (df['atr'].values[1:].reshape(-1,1) + 1e-9))
                    x_t = torch.from_numpy(np.clip(seq[idx-50:idx], -5, 5)).unsqueeze(0).float()
                    with torch.no_grad(): prob_trans = transformer_model(x_t).item()
                    
                    if prob_trans > 0.55:
                        df['returns'] = df['close'].pct_change()
                        df['hl_range'] = (df['high'] - df['low']) / df['close']
                        df['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
                        df['dist_ema'] = (df['close'] / df['ema_200']) - 1
                        seq_l = df[['returns', 'hl_range', 'vol_ratio', 'dist_ema', 'rsi']].values[idx-60:idx]
                        prob_lnn = 0
                        if len(seq_l) == 60:
                            with torch.no_grad(): prob_lnn = torch.sigmoid(lnn_model(torch.from_numpy(seq_l).unsqueeze(0).float())).item()
                        
                        if prob_lnn >= 0.60:
                            price = df['close'].iloc[idx]
                            sl = df['swing_low'].iloc[idx-20:idx].min()
                            if np.isnan(sl) or abs(price-sl)/price > 0.05: sl = price * 0.992
                            tp = price + 2.0 * (price - sl)
                            
                            success, res = trade_manager.enter_trade(symbol, price, sl, tp, amount_usdt=10)
                            if success:
                                last_trade = trade_manager.active_trades[-1]
                                path = plot_advanced_trade(symbol, is_new=False, trade_data=last_trade)
                                await notify(f"🚀 <b>ENTER: {symbol}</b>\n🎯 Prob: {prob_trans:.1%}\n🛡 LNN: YES ({prob_lnn:.1%})", photo=path)
                                if path and os.path.exists(path): os.remove(path)
                            break
        except: continue

async def auto_loop():
    while auto_scan_active:
        await scan_and_trade()
        await asyncio.sleep(900)

async def exit_loop():
    last_alert = {}
    while True:
        try:
            if trade_manager.active_trades:
                msgs = trade_manager.check_trade_exit()
                for m in msgs: await notify(m)
                
                # --- PROFIT ALERTS (15 MIN) ---
                now = datetime.now()
                for t in trade_manager.active_trades:
                    symbol = t['symbol']
                    total_qty = sum(e['qty'] for e in t['entries'])
                    total_cost = sum(e['price'] * e['qty'] for e in t['entries'])
                    avg_entry = total_cost / total_qty
                    ticker = trade_manager.exchange.fetch_ticker(symbol)
                    pnl_pct = (ticker['last'] / avg_entry - 1) * 100
                    
                    threshold = 3.0 if len(t['entries']) == 1 else (2.0 if len(t['entries']) == 2 else 1.0)
                    if pnl_pct >= threshold:
                        if symbol not in last_alert or (now - last_alert[symbol]) >= timedelta(minutes=15):
                            await notify(f"🔔 <b>PROFIT ALERT: {symbol}</b>\n📈 Profit: <b>{pnl_pct:+.2f}%</b>\n🛡 RSI Guard Active")
                            last_alert[symbol] = now
        except Exception as e: logger.error(f"Exit error: {e}")
        await asyncio.sleep(30)

async def main():
    asyncio.create_task(exit_loop())
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
