import asyncio
import logging
import sys
import os
import json
from datetime import datetime, timedelta
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from config.config import BOT_TOKEN, TELEGRAM_PRIVATE_CHAT_ID
from scanner import MarketScanner
from trade_manager import TradeManager
from features.engineer import FeatureEngineer

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

# Initialize
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'pattern_spec.json')
MODEL_PATH = os.path.join(BASE_DIR, 'trained_model.joblib')

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
trade_manager = TradeManager()

print("--- Инициализация сканера и загрузка модели... ---")
scanner = MarketScanner(CONFIG_PATH, MODEL_PATH)
fe = FeatureEngineer()
print(f"✅ Модель и конфиг загружены успешно.")
print(f"✅ Бот запускается... Напишите /start в Telegram.")

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
    await message.answer(f"🤖 <b>Fib 0.618 Market Trader</b>\n\nChat ID: <code>{message.chat.id}</code>\nБот готов к работе.", reply_markup=main_kb, parse_mode="HTML")

@dp.message(F.text == "▶️ Старт Мониторинг")
async def start_auto_monitor(message: types.Message):
    global auto_trade_enabled, current_chat_id
    current_chat_id = message.chat.id
    save_telegram_state(current_chat_id)
    if auto_trade_enabled:
        await message.answer("⚠️ Мониторинг уже запущен.")
        return
    
    auto_trade_enabled = True
    await message.answer("✅ <b>Мониторинг ВКЛЮЧЕН.</b>\nНачинаю поиск лучшей сделки каждые 15 минут...", parse_mode="HTML")
    asyncio.create_task(perform_scan_and_trade(show_progress=True))

@dp.message(F.text == "⏹️ Стоп Мониторинг")
async def stop_auto_monitor(message: types.Message):
    global auto_trade_enabled
    auto_trade_enabled = False
    await message.answer("🛑 <b>Мониторинг ОСТАНОВЛЕН.</b>", parse_mode="HTML")

async def perform_scan_and_trade(show_progress=False):
    progress_msg = None
    if show_progress:
        progress_msg = await send_notification("🔍 Сканирую Binance... (0%)")

    try:
        symbols = scanner.fetcher.get_active_symbols()
        total = len(symbols)
        best_setup = None
        max_prob = 0
        
        print(f"\n--- Начало сканирования ({total} пар) ---")

        for i, symbol in enumerate(symbols):
            if show_progress and i % 50 == 0 and i > 0:
                try: await progress_msg.edit_text(f"🔍 Прогресс сканирования: <code>{i}/{total}</code> монет...", parse_mode="HTML")
                except: pass

            if any(t['symbol'] == symbol for t in trade_manager.active_trades):
                continue

            df = await asyncio.to_thread(scanner.fetcher.fetch_ohlcv, symbol, '1h', (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'))
            if df.empty or len(df) < 40: continue
            
            df = scanner.cleaner.validate_data(df)
            df = scanner.cleaner.calculate_indicators(df)
            
            patterns = scanner.detector.detect_patterns(df)
            # Свежесть 12 часов для H1
            latest = [p for p in patterns if p['impulse']['type'] == 'bullish' and p['structure']['entry_idx'] >= len(df) - 12]
            
            if latest:
                ticker = await asyncio.to_thread(trade_manager.exchange.fetch_ticker, symbol)
                current_price = ticker['last']
                
                X = fe.extract_features(latest, df)
                probs = scanner.trainer.model.predict_proba(X)
                
                for idx, p in enumerate(latest):
                    prob = float(probs[idx][1])
                    entry_lvl = p['structure']['entry_price']
                    
                    print(f"FOUND {symbol}: Prob={prob:.2f}, Entry={entry_lvl:.6g}, Market={current_price:.6g}")
                    
                    if current_price > entry_lvl * 1.01:
                        print(f"   REJECTED: Price too high (+{((current_price/entry_lvl)-1)*100:.1f}%)")
                        continue
                        
                    if prob > max_prob:
                        max_prob = prob
                        best_setup = {'symbol': symbol, 'p': p, 'prob': prob, 'current_price': current_price}
                    elif prob < 0.50:
                        print(f"   REJECTED: Low probability ({prob:.2f} < 0.50)")
            
            if i % 20 == 0:
                print(f"[{i}/{total}] Checking {symbol}...")

        if show_progress and progress_msg: 
            await progress_msg.delete()

        if best_setup and max_prob >= 0.50:
            s = best_setup
            p = s['p']
            struct = p['structure']
            imp = p['impulse']
            
            entry = struct['entry_price']
            sl = struct.get('stop_loss')
            if not sl:
                sl = p['pullback']['low'] * 0.997 if imp['type'] == 'bullish' else p['pullback']['high'] * 1.003
            
            risk = abs(entry - sl)
            target_rr = scanner.config.get('risk_management', {}).get('take_profit', {}).get('rr_min', 2.0)
            tp = entry + (risk * target_rr) if imp['type'] == 'bullish' else entry - (risk * target_rr)
            
            side_str = "LONG 🟢"
            
            msg = f"🎯 <b>СИГНАЛ: {s['symbol']} ({side_str})</b>\n"
            msg += f"Уверенность ML: {s['prob']:.1%}\n"
            msg += f"Тип: {struct['confirmation']}\n\n"
            msg += f"Вход (уровень): <code>{entry:.6g}</code>\n"
            msg += f"Вход (рынок): <code>{s['current_price']:.6g}</code>\n"
            msg += f"SL: <code>{sl:.6g}</code> | TP: <code>{tp:.6g}</code>\n"
            msg += f"Риск/Награда: 1:{target_rr}"
            await send_notification(msg)
            
            success, res_msg = await asyncio.to_thread(
                trade_manager.enter_trade, s['symbol'], s['current_price'], sl, tp, 'bullish', 10
            )
            await send_notification(res_msg)
        else:
            msg_fail = "😴 Сигналов с вероятностью > 50% не найдено."
            print(msg_fail)
            if show_progress: await send_notification(msg_fail)
            
    except Exception as e:
        logger.error(f"Scan error: {e}")
        if show_progress: await send_notification(f"❌ Ошибка: {e}")

@dp.message(F.text == "📡 Обзор рынка (H1)")
async def global_scan_no_trade(message: types.Message):
    status_msg = await message.answer("⏳ Сканирую Топ-100 для обзора...")
    try:
        symbols = scanner.fetcher.get_active_symbols()
        found = []
        for symbol in symbols[:100]:
            df = await asyncio.to_thread(scanner.fetcher.fetch_ohlcv, symbol, '1h', (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'))
            if df.empty: continue
            df = scanner.cleaner.validate_data(df)
            df = scanner.cleaner.calculate_indicators(df)
            
            patterns = scanner.detector.detect_patterns(df)
            latest = [p for p in patterns if p['structure']['entry_idx'] >= len(df) - 12]
            for p in latest:
                found.append((symbol, p))
        
        if not found:
            await status_msg.edit_text("Свежих паттернов ложного пробоя не найдено.")
            return

        res = "📡 <b>Актуальные ложные пробои (H1):</b>\n"
        found.sort(key=lambda x: x[1]['timestamp'], reverse=True)
        for symbol, p in found[:10]:
            side = "🟢 LONG" if p['impulse']['type'] == 'bullish' else "🔴 SHORT"
            res += f"\n🔹 <code>{symbol}</code> | {side}\nВход: <code>{p['structure']['entry_price']:.6g}</code>\n"
        await status_msg.edit_text(res, parse_mode="HTML")
    except Exception as e:
        await message.answer(f"Ошибка: {e}")

@dp.message(F.text == "📊 Статистика")
async def stats_handler(message: types.Message):
    stats = await asyncio.to_thread(trade_manager.get_stats)
    await message.answer(stats, parse_mode="HTML")

@dp.message(F.text == "ℹ️ Статус")
async def status_handler(message: types.Message):
    if trade_manager.active_trades:
        res = "🟢 <b>Активные сделки:</b>\n"
        for t in trade_manager.active_trades:
            res += f"\n🔹 <code>{t['symbol']}</code>\nВход: <code>{t['real_entry_price']:.6g}</code>\nTP: <code>{t['tp']:.6g}</code> | SL: <code>{t['sl']:.6g}</code>\n"
        await message.answer(res, parse_mode="HTML")
    else:
        await message.answer("⚪ Нет активных сделок.")

async def monitor_trades():
    while True:
        try:
            if trade_manager.active_trades:
                msgs = await asyncio.to_thread(trade_manager.check_trade_exit)
                for m in msgs:
                    await send_notification(m)
        except Exception as e:
            logger.error(f"Monitor error: {e}")
        # Проверяем уровни каждую минуту
        await asyncio.sleep(60)

async def auto_scan_task():
    while True:
        if auto_trade_enabled:
            logger.info("Auto-scan triggered (15 min interval).")
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
