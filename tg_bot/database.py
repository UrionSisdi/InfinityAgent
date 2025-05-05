import asyncio
from datetime import datetime
from typing import List, Optional

import asyncpg
from asyncpg import Pool

from config import settings


class Database:
    _pool: Optional[Pool] = None
    
    @classmethod
    async def get_pool(cls) -> Pool:
        if cls._pool is None:
            cls._pool = await asyncpg.create_pool(str(settings.postgres_dsn))
            await cls._init_db()
        return cls._pool
    
    @classmethod
    async def _init_db(cls) -> None:
        """Инициализация БД - создание таблиц при необходимости"""
        pool = cls._pool
        async with pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id BIGINT PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    last_name TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            ''')
            
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT REFERENCES users(id),
                    message_text TEXT,
                    is_from_user BOOLEAN,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            ''')
    
    @classmethod
    async def save_user(cls, user_id: int, username: str, first_name: str, last_name: str) -> None:
        """Сохранение пользователя в БД"""
        pool = await cls.get_pool()
        async with pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO users (id, username, first_name, last_name)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (id) DO UPDATE
                SET username = $2, first_name = $3, last_name = $4
            ''', user_id, username, first_name, last_name)
    
    @classmethod
    async def save_message(cls, user_id: int, message_text: str, is_from_user: bool) -> None:
        """Сохранение сообщения в БД"""
        pool = await cls.get_pool()
        async with pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO messages (user_id, message_text, is_from_user)
                VALUES ($1, $2, $3)
            ''', user_id, message_text, is_from_user)
    
    @classmethod
    async def get_user_messages(cls, user_id: int, limit: int = 10) -> List[dict]:
        """Получение истории сообщений пользователя"""
        pool = await cls.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT message_text, is_from_user, created_at
                FROM messages
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT $2
            ''', user_id, limit)
            
            return [dict(row) for row in rows] 