import os
import httpx
from typing import Any, Dict, Optional, Union

class BackendClient:
    """Клиент для взаимодействия с бэкендом через HTTP API"""
    
    def __init__(self, api_url: Optional[str] = None):
        """Инициализация клиента API"""
        self.api_url = "http://localhost:8000"
    
    async def send_message(self, user_id: int, message: str) -> Dict[str, Any]:
        """
        Отправляет сообщение в бэкенд и возвращает ответ
        """
        try:
            # Отправляем запрос к MCP клиенту через HTTP
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.api_url}/query",
                    json={
                        "query": message
                    }
                )
                
                if response.status_code != 200:
                    raise Exception(f"MCP Client вернул код ошибки: {response.status_code}, {response.text}")
                
                result = response.json()
                return {
                    "status": "success",
                    "response": result.get("result", "Получен пустой ответ от сервера")
                }
                
        except Exception as e:
            print(f"Ошибка при обращении к MCP Client: {e}")
            return {
                "status": "error",
                "response": f"Произошла ошибка при обработке запроса: {str(e)}"
            }
    
    async def get_user_history(self, user_id: int, limit: int = 10) -> Dict[str, Any]:
        """
        Получает историю сообщений пользователя из бэкенда
        """
        # В текущей реализации возвращаем пустой список
        # TODO: Реализовать получение истории из хранилища
        return {
            "status": "success",
            "messages": []
        }
    
    async def get_user_info(self, user_id: int) -> Dict[str, Any]:
        """
        Получает информацию о пользователе из бэкенда
        
        В текущей реализации - простая заглушка
        """
        return {
            "status": "success",
            "user": {
                "id": user_id,
                "name": "Пользователь",
                "settings": {}
            }
        } 