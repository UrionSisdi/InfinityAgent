from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from mcp_client import MCPClient
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pathlib import Path
import os
import re
load_dotenv()


# Функция для очистки ответа от служебных данных и размышлений
def clean_response(text: str) -> str:
    """
    Очищает ответ от служебных данных, размышлений и метаданных.
    Например, убирает блоки вида "**Analyzing Data**" или "I'll use tools to find information".
    """
    if not text:
        return text
        
    # Проверяем на незавершенные размышления без полезного содержимого
    if any(marker in text.lower() for marker in ["thinking", "expected outcome", "tool call", "tool: ", "query: "]):
        return "Для получения этой информации требуется поиск в интернете. Пожалуйста, уточните запрос."
    
    # Удаляем блоки размышлений в формате markdown
    text = re.sub(r'\*\*.*?\*\*\s*\n\s*\n', '', text)
    
    # Удаляем блоки с обсуждением инструментов
    lines = text.split('\n')
    filtered_lines = []
    skip_section = False
    
    for line in lines:
        lower_line = line.lower()
        
        # Проверяем наличие маркеров размышлений/служебной информации
        if any(marker in lower_line for marker in [
            'analyzing', 'thinking', 'considering', 'i\'ll use', 'let me', 'i need to', 
            'gathering', 'assessing', 'investigating', 'reviewing', 'i\'m', 'i will', 'i should', 
            'i must', 'the user wants', 'expected outcome', 'query:', 'tool:'
        ]):
            skip_section = True
            continue
        
        # Пустая строка может означать конец раздела размышлений
        if skip_section and not line.strip():
            skip_section = False
            continue
            
        # Добавляем строку только если она не в пропускаемом блоке
        if not skip_section:
            filtered_lines.append(line)
    
    text = '\n'.join(filtered_lines)
    
    # Удаляем объяснения и вступления
    start_phrases = [
        "вот информация", "согласно", "на основе", "исходя из", 
        "результат", "инструмент", "анализируя", "получен", "вот что"
    ]
    
    # Если текст начинается с одной из фраз, ищем пустую строку и берём только следующий блок
    for phrase in start_phrases:
        if text.lower().startswith(phrase):
            parts = text.split('\n\n', 1)
            if len(parts) > 1:
                text = parts[1]
                break
    
    # Удаляем любые начальные пустые строки
    text = text.lstrip()
    
    # Если текст стал пустым или слишком коротким, возвращаем сообщение по умолчанию
    if not text or len(text) < 20:
        return "Для получения этой информации требуется поиск в интернете. Пожалуйста, уточните запрос."
    
    return text


class Settings(BaseSettings):
    mcp_server_url: str = "http://localhost:8001"


settings = Settings()

# Глобальная переменная для хранения клиента MCP
mcp_client: Optional[MCPClient] = None


def get_mcp_client():
    if mcp_client is None:
        raise HTTPException(status_code=500, detail="MCP client not initialized")
    return mcp_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    global mcp_client
    client = MCPClient()
    try:
        # Подключаемся к серверу сразу при запуске
        connected = await client.connect_to_http_server(settings.mcp_server_url)
        if not connected:
            raise HTTPException(status_code=500, detail="Failed to connect to MCP server")
            
        mcp_client = client
        yield
    except Exception as e:
        print(f"Error during lifespan: {e}")
        raise HTTPException(status_code=500, detail=f"Error during lifespan: {str(e)}") from e
    finally:
        # shutdown
        await client.cleanup()


app = FastAPI(title="MCP Client API", lifespan=lifespan)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class QueryRequest(BaseModel):
    query: str


class Message(BaseModel):
    role: str
    content: Any


class ToolCall(BaseModel):
    name: str
    args: Dict[str, Any]


@app.post("/query")
async def process_query(request: QueryRequest, client: MCPClient = Depends(get_mcp_client)):
    """Process a query and return the response"""
    try:
        result = await client.process_query(request.query)
        
        # Проверяем тип возвращаемого результата
        if isinstance(result, str):
            # Если результат - строка, очищаем её от служебных данных и возвращаем
            cleaned_result = clean_response(result)
            return {"result": cleaned_result}
        elif isinstance(result, list):
            # Если это список сообщений, возвращаем последнее сообщение ассистента
            for message in reversed(result):
                if message["role"] == "assistant" and isinstance(message["content"], str) and message["content"].strip():
                    cleaned_content = clean_response(message["content"])
                    return {"result": cleaned_content}
            
            # Если не найдено подходящее сообщение ассистента
            return {"result": "Не удалось получить ответ на ваш запрос."}
        else:
            # Неизвестный тип результата
            return {"result": str(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools")
async def get_tools(client: MCPClient = Depends(get_mcp_client)):
    """Get the list of available tools"""
    try:
        tools = await client.get_mcp_tools()
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in tools
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
