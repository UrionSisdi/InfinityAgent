import os
from typing import Any, Dict, List, Optional, Union

import httpx
from dotenv import load_dotenv

load_dotenv()


class MCPClient:
    """Клиент для взаимодействия с бэкендом InfinityAgent"""
    
    def __init__(
        self,
        api_url: Optional[str] = None,
    ):
        """
        Инициализация клиента API
        
        Args:
            api_url: URL API бэкенда (если не указан, берется из переменной окружения API_URL)
            api_key: API ключ для авторизации (если не указан, берется из переменной окружения API_KEY)
        """
        self.api_url = api_url or os.getenv("API_URL", "http://localhost:8000")
    
    async def send_message(
        self, 
        user_id: Union[int, str], 
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Отправка сообщения бэкенду для обработки
        
        Args:
            user_id: ID пользователя
            message: Текст сообщения
            context: Дополнительный контекст (опционально)
            
        Returns:
            Dict[str, Any]: Ответ от бэкенда
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_url}/api/message",
                json={
                    "user_id": user_id,
                    "message": message,
                    "context": context or {},
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            
            if response.status_code != 200:
                raise Exception(f"Ошибка API: {response.status_code}, {response.text}")
            
            return response.json()
    
    async def get_user_history(
        self, 
        user_id: Union[int, str], 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Получение истории сообщений пользователя
        
        Args:
            user_id: ID пользователя
            limit: Максимальное количество сообщений (по умолчанию 10)
            
        Returns:
            List[Dict[str, Any]]: Список сообщений пользователя
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.api_url}/api/user/{user_id}/history",
                params={"limit": limit},
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            
            if response.status_code != 200:
                raise Exception(f"Ошибка API: {response.status_code}, {response.text}")
            
            return response.json().get("messages", []) 