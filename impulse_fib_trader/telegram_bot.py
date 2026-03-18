import asyncio
import logging
import sys
import os
import json
from datetime import datetime, timedelta
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from config.config import BOT_TOKEN, TELEGRAM_PRIVATE_CHAT_ID
from data.fetcher import DataFetcher
from data.cleaner import DataCleaner
from pattern.tas_detector import TASDetector
from trade_manager import TradeManager
from features.engineer import FeatureEngineer
from ml.train import MLTrainer

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# State
auto_trade_enabled = False
TELEGRAM_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'telegram_state.json')

def load_telegram_state():
    if os.path.exists(TELEGRAM_STATE_FILE):
        try:
            with open(TELEGRAM_STATE_FILE, 'r') as f:
                return json.load(f).get('current_chat_id', TELEGRAM_PRIVATE_CHAT_ID)
        except: pass
    return TELEGRAM_PRIVATE_CHAT_ID

def save_telegram_state(chat_id):
    try:
        with open(TELEGRAM_STATE_FILE, 'w') as f:
            json.dump({'current_chat_id': str(chat_id)}, f)
    except: pass

current_chat_id = load_telegram_state()

# Initialize TAS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'pattern_spec_tas.json')
MODEL_PATH = os.path.join(BASE_DIR, 'trained_model_tas.joblib')

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
trade_manager = TradeManager()

fetcher = DataFetcher()
cleaner = DataCleaner()
fe = FeatureEngineer()
trainer = MLTrainer()

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)
detector = TASDetector(config)

if os.path.exists(MODEL_PATH):
    trainer.load_model(MODEL_PATH)
    print(f"✅ Модель TAS загружена: {MODEL_PATH}")
else:
    print(f"⚠️ Модель {MODEL_PATH} не найдена. Бот будет работать без ML фильтра!")

# Keyboards
main_kb = ReplyKeyboardMarkup(keyboard=[
    [KeyboardButton(text="▶️ Старт Мониторинг"), KeyboardButton(text="⏹️ Стоп Мониторинг")],
    [KeyboardButton(text="📡 Обзор рынка (H1)")],
    [KeyboardButton(text="📊 Статистика"), KeyboardButton(text="ℹ️ Статус")]
], resize_keyboard=True)

async def send_notification(text):
    global current_chat_id
    try:
        return await bot.send_message(current_chat_id, text, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Ошибка отправки в ТГ: {e}")
        return None

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    global current_chat_id
    current_chat_id = message.chat.id
    save_telegram_state(current_chat_id)
    await message.answer(f"🤖 <b>TAS_v1 (Tails & Shelves) Bot</b>\n\nChat ID: <code>{message.chat.id}</code>\nБот готов к работе.", reply_markup=main_kb, parse_mode="HTML")

@dp.message(F.text == "▶️ Старт Мониторинг")
async def start_auto_monitor(message: types.Message):
    global auto_trade_enabled, current_chat_id
    current_chat_id = message.chat.id
    save_telegram_state(current_chat_id)
    if auto_trade_enabled:
        await message.answer("⚠️ Мониторинг уже запущен.")
        return
    
    auto_trade_enabled = True
    await message.answer("✅ <b>Мониторинг TAS_v1 ВКЛЮЧЕН.</b>\nИнтервал: 15 мин.", parse_mode="HTML")
    asyncio.create_task(perform_scan_and_trade(show_progress=True))

@dp.message(F.text == "⏹️ Стоп Мониторинг")
async def stop_auto_monitor(message: types.Message):
    global auto_trade_enabled
    auto_trade_enabled = False
    await message.answer("🛑 <b>Мониторинг ОСТАНОВЛЕН.</b>", parse_mode="HTML")

async def perform_scan_and_trade(show_progress=False):
    if show_progress:
        await send_notification("🔍 Сканирую рынок на наличие паттернов TAS...")

    try:
        symbols = fetcher.get_active_symbols()
        total = len(symbols)
        best_setup = None
        max_prob = 0
        
        cooldown_symbols = trade_manager.get_cooldown_symbols(hours=4)

        for i, symbol in enumerate(symbols):
            if any(t['symbol'] == symbol for t in trade_manager.active_trades):
                continue
            if symbol in cooldown_symbols:
                continue

            df = await asyncio.to_thread(fetcher.fetch_ohlcv, symbol, '1h', (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'))
            if df.empty or len(df) < 40: continue
            
            df = cleaner.validate_data(df)
            df = cleaner.calculate_indicators(df)
            
            patterns = detector.detect_patterns(df)
            # Берем только свежие пробои (последние 3 свечи)
            latest = [p for p in patterns if p['entry_idx'] >= len(df) - 3]
            
            if latest:
                # Проверка на "нож" (2 зеленые свечи)
                if df.iloc[-1]['close'] <= df.iloc[-1]['open'] or df.iloc[-2]['close'] <= df.iloc[-2]['open']:
                    continue

                ticker = await asyncio.to_thread(trade_manager.exchange.fetch_ticker, symbol)
                current_price = ticker['last']
                
                prob = 1.0 # По умолчанию если нет модели
                if trainer.model:
                    X = fe.extract_features(latest, df)
                    probs = trainer.model.predict_proba(X)
                    prob = float(probs[-1][1])
                
                p = latest[-1]
                if prob > max_prob and current_price < p['entry_price'] * 1.01:
                    max_prob = prob
                    best_setup = {'symbol': symbol, 'p': p, 'prob': prob, 'current_price': current_price}
            
            if i % 50 == 0: print(f"Scanned {i}/{total}...")

        if best_setup and max_prob >= 0.50:
            s = best_setup
            p = s['p']
            
            sl = p['tail_low']
            risk = s['current_price'] - sl
            tp = s['current_price'] + (risk * 2.0)
            
            msg = f"🎯 <b>СИГНАЛ TAS_v1: {s['symbol']}</b>\n"
            msg += f"Уверенность ML: {s['prob']:.1%}\n\n"
            msg += f"Вход (рынок): <code>{s['current_price']:.6g}</code>\n"
            msg += f"SL: <code>{sl:.6g}</code> | TP: <code>{tp:.6g}</code>\n"
            await send_notification(msg)
            
            success, res_msg = await asyncio.to_thread(
                trade_manager.enter_trade, s['symbol'], s['current_price'], sl, tp, 'TAS', 10
            )
            await send_notification(res_msg)
            
    except Exception as e:
        logger.error(f"Scan error: {e}")

@dp.message(F.text == "📡 Обзор рынка (H1)")
async def global_scan_no_trade(message: types.Message):
    status_msg = await message.answer("⏳ Ищу паттерны TAS (Tails & Shelves)...")
    try:
        symbols = fetcher.get_active_symbols()
        found = []
        for symbol in symbols[:100]:
            df = await asyncio.to_thread(fetcher.fetch_ohlcv, symbol, '1h', (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'))
            if df.empty: continue
            df = cleaner.validate_data(df)
            df = cleaner.calculate_indicators(df)
            patterns = detector.detect_patterns(df)
            latest = [p for p in patterns if p['entry_idx'] >= len(df) - 6]
            for p in latest:
                found.append((symbol, p))
        
        if not found:
            await status_msg.edit_text("Свежих паттернов TAS не найдено.")
            return

        res = "📡 <b>Актуальные паттерны TAS (H1):</b>\n"
        for symbol, p in found[:10]:
            res += f"\n🔹 <code>{symbol}</code>\nУровень пробоя: <code>{p['breakout_level']:.6g}</code>\n"
        await status_msg.edit_text(res, parse_mode="HTML")
    except Exception as e:
        await message.answer(f"Ошибка: {e}")

@dp.message(F.text == "📊 Статистика")
async def stats_handler(message: types.Message):
    stats = await asyncio.to_thread(trade_manager.get_stats)
    await message.answer(stats, parse_mode="HTML")

@dp.message(F.text == "ℹ️ Статус")
async def status_handler(message: types.Message):
    if not trade_manager.active_trades:
        await message.answer("⚪ Нет активных сделок.")
        return

    await message.answer("⏳ Запрашиваю текущие цены...")
    
    for t in trade_manager.active_trades:
        symbol = t['symbol']
        try:
            ticker = await asyncio.to_thread(trade_manager.exchange.fetch_ticker, symbol)
            curr_price = ticker['last']
            entry_price = t['real_entry_price']
            pnl_perc = ((curr_price / entry_price) - 1) * 100
            pnl_color = "🟢" if pnl_perc >= 0 else "🔴"
            
            tv_symbol = symbol.replace("/", "")
            tv_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{tv_symbol}"
            
            res = f"🔹 <b>{symbol} (TAS)</b>\n"
            res += f"Вход: <code>{entry_price:.6g}</code>\n"
            res += f"Текущая: <code>{curr_price:.6g}</code> ({pnl_color} {pnl_perc:+.2f}%)\n"
            res += f"TP: <code>{t['tp']:.6g}</code> | SL: <code>{t['sl']:.6g}</code>"
            
            sell_kb = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text=f"💰 Продать", callback_data=f"sell_{symbol}"),
                    InlineKeyboardButton(text="📈 График TV", url=tv_url)
                ]
            ])
            await message.answer(res, reply_markup=sell_kb, parse_mode="HTML")
        except Exception as e:
            logger.error(f"Error: {e}")

@dp.callback_query(F.data.startswith("sell_"))
async def process_manual_sell(callback: CallbackQuery):
    symbol = callback.data.replace("sell_", "")
    success, msg = await asyncio.to_thread(trade_manager.manual_market_exit, symbol)
    if success:
        await callback.message.edit_text(f"✅ Сделка по {symbol} закрыта.\n{msg}", parse_mode="HTML")
    else:
        await callback.message.answer(f"❌ {msg}")

async def monitor_trades():
    while True:
        try:
            if trade_manager.active_trades:
                msgs = await asyncio.to_thread(trade_manager.check_trade_exit)
                for m in msgs:
                    await send_notification(m)
        except Exception as e:
            logger.error(f"Monitor error: {e}")
        await asyncio.sleep(60)

async def auto_scan_task():
    while True:
        if auto_trade_enabled:
            await perform_scan_and_trade(show_progress=False)
        await asyncio.sleep(60 * 15)

async def main():
    asyncio.create_task(monitor_trades())
    asyncio.create_task(auto_scan_task())
    await dp.start_polling(bot)

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
