# MCP Client

Клиентская библиотека для взаимодействия с бэкендом InfinityAgent.

## Установка

```bash
cd mcp-client
pip install -e .
```

## Использование

```python
import asyncio
from api import MCPClient

async def main():
    # Инициализация клиента
    client = MCPClient(
        api_url="http://localhost:8000",
        api_key="ваш_api_ключ"
    )
    
    # Отправка сообщения
    response = await client.send_message(
        user_id=123,
        message="Привет, это тестовое сообщение!"
    )
    print(response)
    
    # Получение истории сообщений
    history = await client.get_user_history(
        user_id=123,
        limit=5
    )
    print(history)

if __name__ == "__main__":
    asyncio.run(main())
```

## Переменные окружения

Клиент может быть настроен с помощью следующих переменных окружения:

- `API_URL` - URL-адрес API (по умолчанию: "http://localhost:8000")
- `API_KEY` - API ключ для авторизации

Вы также можете передать эти значения непосредственно при создании экземпляра клиента.

## Функциональность

- Отправка сообщений в бэкенд
- Получение истории сообщений пользователя

## Разработка

Для настройки окружения разработки:

```bash
cd mcp-client
pip install -e ".[dev]"
``` 