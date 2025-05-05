import asyncio
import logging
import sys
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from backend_client import BackendClient
from config import settings
from database import Database

# Проверяем значения настроек перед их использованием
print(f"Прямой доступ к settings: POSTGRES_DB={settings.POSTGRES_DB}")
print(f"Прямой доступ к DSN: {settings.postgres_dsn}")

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Инициализация клиента бэкенда
# Здесь можно передать URL для MCP клиента, если он не на localhost:8001
backend_client = BackendClient()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start"""
    user = update.effective_user
    
    # Сохраняем пользователя в БД
    await Database.save_user(
        user_id=user.id,
        username=user.username or "",
        first_name=user.first_name or "",
        last_name=user.last_name or "",
    )
    
    welcome_message = f"Привет, *{user.first_name}*! Я бот для работы с InfinityAgent."
    await update.message.reply_text(welcome_message, parse_mode=ParseMode.MARKDOWN)
    
    # Сохраняем ответное сообщение в БД
    await Database.save_message(
        user_id=user.id, 
        message_text=welcome_message, 
        is_from_user=False
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /help"""
    help_text = (
        "Доступные команды:\n"
        "*/start* - Начать работу с ботом\n"
        "*/help* - Показать справку\n"
        "_Просто напишите сообщение, чтобы отправить его на обработку._"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)
    
    # Сохраняем ответное сообщение в БД
    user = update.effective_user
    await Database.save_message(
        user_id=user.id, 
        message_text=help_text, 
        is_from_user=False
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик обычных текстовых сообщений от пользователя"""
    user = update.effective_user
    message_text = update.message.text
    
    # Сохраняем сообщение пользователя в БД
    await Database.save_message(
        user_id=user.id, 
        message_text=message_text, 
        is_from_user=True
    )
    
    # Отправляем индикатор набора текста
    await update.message.chat.send_action(action="typing")
    
    # Отправляем сообщение в бэкенд
    response_data = await backend_client.send_message(
        user_id=user.id,
        message=message_text
    )
    
    response_text = response_data.get("response", "Произошла ошибка при обработке запроса.")
    
    try:
        # Пробуем отправить сообщение с поддержкой Markdown
        await update.message.reply_text(response_text, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        # Если возникла ошибка (например, неправильные символы Markdown), отправляем без форматирования
        logger.warning(f"Ошибка при отправке сообщения с Markdown: {e}, отправляем без форматирования")
        await update.message.reply_text(response_text)
    
    # Сохраняем ответное сообщение в БД
    await Database.save_message(
        user_id=user.id, 
        message_text=response_text, 
        is_from_user=False
    )


async def main():
    """Основная функция запуска бота"""
    # Отладочная информация
    logger.info(f"Переменные окружения: POSTGRES_DB={settings.POSTGRES_DB}, PORT={settings.POSTGRES_PORT}")
    logger.info(f"Используется URL MCP клиента: {backend_client.api_url}")
    
    # Инициализация бота
    application = Application.builder().token(settings.TELEGRAM_TOKEN).build()
    logger.info(f"Using DSN: {settings.postgres_dsn}")
    
    # Регистрация обработчиков команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    
    # Регистрация обработчика текстовых сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Запуск бота
    logger.info("Запуск бота...")
    
    # Инициализируем приложение
    await application.initialize()
    # Запускаем бота
    await application.start()
    # Запускаем опрос обновлений
    await application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
    
    # Блокируем выполнение, чтобы бот продолжал работать
    logger.info("Бот успешно запущен и готов к работе")
    
    # Просто ждем вечно или до сигнала прерывания (Ctrl+C)
    try:
        # Бесконечный цикл ожидания
        await asyncio.Event().wait()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Получен сигнал завершения, останавливаем бота...")
    finally:
        # Корректное завершение работы при выходе
        logger.info("Останавливаем бота...")
        await application.updater.stop()
        await application.stop()
        await application.shutdown()
        logger.info("Бот остановлен")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Критическая ошибка при запуске бота: {e}")
        sys.exit(1)
