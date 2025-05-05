from typing import Optional
from contextlib import AsyncExitStack
import traceback
import httpx
import subprocess
import time

# from utils.logger import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from datetime import datetime
from utils.logger import logger
import json
import os
from openai import OpenAI


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm = OpenAI(
            base_url="https://openrouter.ai/api/v1", 
            api_key="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",
        )
        self.tools = []
        self.messages = []
        self.logger = logger
        self.http_client = None
        self.http_base_url = None
        self.connect_mode = None

    # connect to the MCP server
    async def connect_to_server(self, server_script_path: str, mode="auto", http_url=None):
        try:
            # Если указан HTTP URL, используем его
            if http_url:
                await self.connect_to_http_server(http_url)
                return True
                
            # Автоматический выбор режима подключения или принудительный режим
            if mode == "auto" or mode == "http":
                # Проверяем, нужно ли запустить сервер в HTTP режиме
                if mode == "auto":
                    # Пробуем подключиться к локальному HTTP серверу
                    try:
                        await self.connect_to_http_server("http://127.0.0.1:8001")
                        return True
                    except:
                        # Если не удалось подключиться к существующему HTTP серверу, 
                        # запускаем новый в HTTP режиме
                        self.logger.info("Не удалось подключиться к HTTP серверу, запускаем новый.")
                
                # Запускаем сервер в HTTP режиме
                is_python = server_script_path.endswith(".py")
                is_js = server_script_path.endswith(".js")
                if not (is_python or is_js):
                    raise ValueError("Server script must be a .py or .js file")

                command = "python" if is_python else "node"
                self.logger.info(f"Launching HTTP server: {command} {server_script_path} --mode http")
                
                # Запуск процесса сервера в фоновом режиме
                subprocess.Popen(
                    [command, server_script_path, "--mode", "http"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
                
                # Ждем запуска сервера
                time.sleep(2)
                
                # Подключаемся к HTTP серверу
                await self.connect_to_http_server("http://127.0.0.1:8001")
                return True
            
            # Стандартное подключение по stdio
            is_python = server_script_path.endswith(".py")
            is_js = server_script_path.endswith(".js")
            if not (is_python or is_js):
                raise ValueError("Server script must be a .py or .js file")

            command = "python" if is_python else "node"
            server_params = StdioServerParameters(
                command=command, args=[server_script_path], env=None
            )

            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )

            await self.session.initialize()
            self.connect_mode = "stdio"

            self.logger.info("Connected to MCP server via stdio")

            mcp_tools = await self.get_mcp_tools()
            self.tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in mcp_tools
            ]

            self.logger.info(
                f"Available tools: {[tool['name'] for tool in self.tools]}"
            )

            return True

        except Exception as e:
            self.logger.error(f"Error connecting to MCP server: {e}")
            traceback.print_exc()
            raise
            
    # Подключение к HTTP серверу
    async def connect_to_http_server(self, base_url):
        try:
            self.http_client = httpx.AsyncClient(base_url=base_url, timeout=30.0)
            self.http_base_url = base_url
            self.connect_mode = "http"
            
            # Проверяем доступность сервера
            response = await self.http_client.get("/list-tools")
            if response.status_code != 200:
                raise Exception(f"HTTP server returned status code {response.status_code}")
                
            # Получаем список инструментов
            tools_data = response.json()
            self.tools = tools_data.get("tools", [])
            
            self.logger.info(f"Connected to MCP server via HTTP at {base_url}")
            self.logger.info(f"Available tools: {[tool['name'] for tool in self.tools]}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error connecting to HTTP server: {e}")
            raise

    # get mcp tool list
    async def get_mcp_tools(self):
        try:
            if self.connect_mode == "http":
                response = await self.http_client.get("/list-tools")
                response.raise_for_status()
                return response.json().get("tools", [])
            else:
                response = await self.session.list_tools()
                return response.tools
        except Exception as e:
            self.logger.error(f"Error getting MCP tools: {e}")
            raise

    # Общая функция для вызова инструмента независимо от типа
    async def execute_tool(self, tool_name, tool_args):
        try:
            self.logger.info(f"Вызов инструмента {tool_name} с аргументами {tool_args}")
            
            if self.connect_mode == "http":
                result_data = await self.http_client.post(
                    "/call-tool", 
                    json={"name": tool_name, "args": tool_args}
                )
                result_data.raise_for_status()
                result_json = result_data.json()
                if result_json.get("status") == "error":
                    raise Exception(result_json.get("message"))
                result_content = result_json.get("result", "No result")
            else:
                result = await self.session.call_tool(tool_name, tool_args)
                result_content = result.content
                
            # Безопасно логируем результаты, обрабатывая возможные проблемы с кодировкой
            try:
                log_preview = result_content[:100] + "..." if len(result_content) > 100 else result_content
                self.logger.info(f"Результат инструмента {tool_name}: {log_preview}")
            except UnicodeEncodeError:
                # Для случаев, когда в результате есть проблемные символы
                self.logger.info(f"Результат инструмента {tool_name} получен (не отображается из-за проблем кодировки)")
                
            return result_content
        except Exception as e:
            self.logger.error(f"Ошибка при вызове инструмента {tool_name}: {e}")
            raise

    # process query
    async def process_query(self, query: str):
        try:
            self.logger.info(f"Processing query: {query}")
            user_message = {"role": "user", "content": query}
            self.messages = [user_message]

            while True:
                response = await self.call_llm()

                # Проверяем, есть ли вызовы инструментов
                if not response.tool_calls:
                    # LLM не вызвал инструменты. Если запрос связан с определенными темами,
                    # мы сами можем решить вызвать соответствующие инструменты
                    
                    # Проверяем, содержит ли запрос ключевые слова для поиска документации
                    lower_query = query.lower()
                    
                    # Ключевые слова, которые указывают на необходимость поиска документации
                    doc_keywords = [
                        "докум", "документ", "документация", "как использовать", "пример", 
                        "доки", "синтаксис", "функция", "метод", "класс", "api"
                    ]
                    
                    # Ключевые слова для библиотек
                    library_keywords = {
                        "langchain": ["langchain", "ланчейн", "ланчин", "langсhain"],
                        "openai": ["openai", "опенай", "опенаи", "gpt", "chatgpt", "гпт"],
                        "llama-index": ["llama", "индекс", "индексация", "index", "llama-index"]
                    }
                    
                    # Слова, которые указывают на необходимость поиска в интернете
                    search_keywords = [
                        "найти", "поиск", "погугли", "загугли", "посмотри", "информация", 
                        "данные", "что такое", "как", "статья", "статьи", "статистика"
                    ]
                    
                    # Ключевые слова, указывающие на запрос погоды
                    weather_keywords = [
                        "погода", "температура", "градус", "осадки", "дождь", "снег", "ветер", 
                        "прогноз", "климат", "тепло", "холодно", "небо", "солнце", "облачность"
                    ]
                    
                    # Ключевые слова для спортивных запросов
                    sports_keywords = [
                        "спорт", "клуб", "команда", "игра", "матч", "хоккей", "футбол", 
                        "баскетбол", "чемпионат", "турнир", "лига", "игрок", "гол", 
                        "очко", "счет", "счёт", "статистика игр", "статистика матчей"
                    ]
                    
                    # Определяем наличие ключевых слов в запросе
                    is_doc_request = any(kw in lower_query for kw in doc_keywords)
                    
                    # Определяем библиотеку
                    target_library = None
                    for lib, keywords in library_keywords.items():
                        if any(kw in lower_query for kw in keywords):
                            target_library = lib
                            break
                    
                    # Проверяем, есть ли в запросе ключевые слова для поиска в интернете
                    is_search_request = any(kw in lower_query for kw in search_keywords)
                    
                    # Проверяем, есть ли запрос о погоде
                    is_weather_request = any(kw in lower_query for kw in weather_keywords)
                    
                    # Проверяем, есть ли запрос о спорте
                    is_sports_request = any(kw in lower_query for kw in sports_keywords)
                    
                    # Извлекаем местоположение из запроса о погоде
                    location = None
                    if is_weather_request:
                        # Список городов и регионов для проверки
                        cities_regions = [
                            "москва", "санкт-петербург", "питер", "спб", "новосибирск", "екатеринбург", 
                            "казань", "нижний новгород", "челябинск", "омск", "самара", "ростов", 
                            "уфа", "красноярск", "воронеж", "пермь", "волгоград", "краснодар", 
                            "россия", "беларусь", "казахстан", "украина", "киев", "минск", "астана", 
                            "нур-султан", "алматы", "париж", "лондон", "берлин", "нью-йорк", "токио",
                            "пекин", "сидней", "рим", "мадрид", "барселона", "амстердам", "вена"
                        ]
                        
                        words = lower_query.split()
                        for city in cities_regions:
                            if city in lower_query:
                                location = city
                                # Преобразуем первую букву в заглавную для корректного отображения
                                location = location[0].upper() + location[1:]
                                break
                        
                        # Если местоположение не найдено, попробуем найти слова после ключевых слов о погоде
                        if not location:
                            for keyword in weather_keywords:
                                if keyword in lower_query:
                                    # Находим индекс ключевого слова
                                    idx = lower_query.find(keyword)
                                    # Пытаемся извлечь слово после ключевого слова
                                    rest_of_query = lower_query[idx + len(keyword):].strip()
                                    words_after = rest_of_query.split()
                                    if words_after and len(words_after[0]) > 3:  # Предполагаем, что название города длиннее 3 символов
                                        location = words_after[0]
                                        # Преобразуем первую букву в заглавную
                                        location = location[0].upper() + location[1:]
                                        break
                        
                        # Если местоположение всё ещё не найдено, используем Москву по умолчанию
                        if not location:
                            location = "Москва"
                    
                    # Если запрос похож на запрос документации и определена библиотека
                    if is_doc_request and target_library:
                        self.logger.info(
                            f"Запрос похож на запрос документации для {target_library}, вызываем get_docs"
                        )
                        
                        # Вызываем инструмент get_docs
                        tool_result = await self.execute_tool(
                            "get_docs", 
                            {"query": query, "library": target_library}
                        )
                        
                        # Добавляем ответ инструмента и создаем новое сообщение от ассистента
                        tool_message = {"role": "tool", "tool_call_id": "1", "content": tool_result}
                        self.messages.append(tool_message)
                        
                        # Вызываем LLM еще раз с результатом инструмента
                        response = await self.call_llm()
                        
                        # Возвращаем финальный ответ пользователю
                        return response.content
                    elif is_weather_request and location:
                        self.logger.info(f"Запрос похож на запрос погоды для местоположения: {location}")
                        
                        # Вызываем инструмент get_weather
                        tool_result = await self.execute_tool("get_weather", {"location": location})
                        
                        # Добавляем ответ инструмента и создаем новое сообщение от ассистента
                        tool_message = {"role": "tool", "tool_call_id": "1", "content": tool_result}
                        self.messages.append(tool_message)
                        
                        # Вызываем LLM еще раз с результатом инструмента
                        response = await self.call_llm()
                        
                        # Возвращаем финальный ответ пользователю
                        return response.content
                    elif is_sports_request:
                        self.logger.info(f"Запрос спортивной информации, вызываем web_search напрямую")
                        
                        # Вызываем инструмент web_search
                        tool_result = await self.execute_tool("web_search", {"query": query})
                        
                        # Возвращаем результаты поиска напрямую, без LLM обработки
                        return tool_result
                    elif is_search_request and not (is_doc_request and target_library):
                        self.logger.info(f"Запрос похож на запрос поиска в интернете, вызываем web_search")
                        
                        # Определяем, нужно ли возвращать результаты напрямую
                        # Если запрос начинается с явных маркеров поиска, возвращаем напрямую
                        direct_search_markers = ["загугли", "найди", "поищи", "покажи", "посмотри"]
                        is_direct_search = any(query.lower().startswith(marker) for marker in direct_search_markers)
                        
                        # Вызываем инструмент web_search
                        tool_result = await self.execute_tool("web_search", {"query": query})
                        
                        if is_direct_search:
                            # Возвращаем результаты поиска напрямую
                            return tool_result
                        else:
                            # Добавляем ответ инструмента и создаем новое сообщение от ассистента
                            tool_message = {"role": "tool", "tool_call_id": "1", "content": tool_result}
                            self.messages.append(tool_message)
                            
                            # Вызываем LLM еще раз с результатом инструмента
                            response = await self.call_llm()
                            
                            # Возвращаем финальный ответ пользователю
                            return response.content
                    
                    # Если запрос не требует инструментов, просто возвращаем ответ модели
                    return response.content
                
                # Если есть текстовое содержимое и нет вызовов инструментов
                if response.content and not response.tool_calls:
                    assistant_message = {
                        "role": "assistant",
                        "content": response.content,
                    }
                    self.messages.append(assistant_message)
                    await self.log_conversation()
                    break

                # Если есть вызовы инструментов
                if response.tool_calls:
                    assistant_message = {
                        "role": "assistant",
                        "content": [],
                    }
                    
                    # Если есть текстовое содержимое, добавляем его
                    if response.content:
                        assistant_message["content"] = response.content
                    
                    # Добавляем tool calls в сообщение
                    if hasattr(response, "to_dict"):
                        assistant_message["content"] = response.to_dict().get("content", [])
                    
                    self.messages.append(assistant_message)
                    await self.log_conversation()

                    for tool_call in response.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        tool_use_id = tool_call.id
                        
                        try:
                            # Вызываем инструмент через единый метод
                            result_content = await self.execute_tool(tool_name, tool_args)
                            
                            # Добавляем результат в сообщения
                            self.messages.append(
                                {
                                    "role": "user",
                                    "content": f"Результат инструмента {tool_name}: {result_content[:5000]}..."
                                }
                            )
                            await self.log_conversation()
                        except Exception as e:
                            self.logger.error(f"Error calling tool {tool_name}: {e}")
                            error_message = {
                                "role": "user",
                                "content": f"Ошибка при вызове инструмента {tool_name}: {str(e)}"
                            }
                            self.messages.append(error_message)
                            await self.log_conversation()
                    continue
                
                # Если нет ни текста, ни tool_calls, завершаем цикл
                break

            return self.messages

        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            raise

    # call llm
    async def call_llm(self):
        try:
            self.logger.info("Calling LLM")
            
            # Проверим доступные инструменты перед вызовом
            if not self.tools or len(self.tools) == 0:
                self.logger.warning("Нет доступных инструментов для использования!")
                # Обновим список инструментов если возможно
                try:
                    tools = await self.get_mcp_tools()
                    if tools:
                        self.tools = tools
                        self.logger.info(f"Обновленные инструменты: {[tool['name'] if isinstance(tool, dict) else tool.name for tool in self.tools]}")
                except Exception as e:
                    self.logger.error(f"Не удалось обновить инструменты: {e}")
            else:
                self.logger.info(f"Доступные инструменты: {[tool['name'] if isinstance(tool, dict) else tool.name for tool in self.tools]}")
            
            # Более чёткое системное сообщение
            system_message = {
                "role": "system", 
                "content": (
                    "Ты ассистент с доступом к инструментам. ВСЕГДА используй инструменты перед ответом. "
                    "НЕ ОТВЕЧАЙ на вопросы напрямую, особенно о программировании, библиотеках и API. "
                    "Инструмент get_docs должен быть использован для любых вопросов о библиотеках, документации, "
                    "программировании или технических вопросов. Если у тебя есть доступ к get_docs, ВСЕГДА используй его. "
                    "Никогда не отвечай на основе только своих знаний. "
                    "ОЧЕНЬ ВАЖНО: Не добавляй никаких пояснений или размышлений к своему ответу. Давай только краткий, "
                    "чёткий ответ на русском языке без вступлений вроде 'Вот информация о...' или 'Анализируя данные...'."
                )
            }
            
            # Создаем новый первый запрос, явно указывающий использовать инструмент
            # Это заменит первый запрос пользователя на версию с явным указанием использовать инструмент
            if len(self.messages) > 0 and self.messages[0]["role"] == "user":
                original_query = self.messages[0]["content"]
                if isinstance(original_query, str):
                    new_user_message = {
                        "role": "user",
                        "content": f"Используй инструмент для поиска информации о: {original_query}"
                    }
                    messages_with_system = [system_message, new_user_message]
                    # Добавляем остальные сообщения, если они есть
                    if len(self.messages) > 1:
                        messages_with_system.extend(self.messages[1:])
                else:
                    # Если содержимое не строка, используем оригинальные сообщения
                    messages_with_system = [system_message] + self.messages
            else:
                messages_with_system = [system_message] + self.messages
            
            # Принудительно указываем использовать инструмент через tool_choice
            tool_choice = None
            if self.tools and len(self.tools) > 0:
                # Проверяем тип запроса пользователя, чтобы выбрать подходящий инструмент
                first_user_message = ""
                for msg in self.messages:
                    if msg["role"] == "user":
                        first_user_message = msg["content"] if isinstance(msg["content"], str) else ""
                        break
                
                first_user_message = first_user_message.lower()
                
                # Определяем ключевые слова для разных типов запросов
                weather_keywords = ["погода", "температура", "градус", "осадки", "дождь", "снег", "ветер"]
                search_keywords = ["найти", "поиск", "погугли", "загугли", "информация", "данные", "статистика"]
                doc_keywords = ["документ", "библиотек", "api", "код", "пример", "функция", "метод"]
                
                # Добавляем ключевые слова для спортивной тематики
                sports_keywords = ["спорт", "клуб", "команда", "игра", "матч", "хоккей", "футбол", "баскетбол", 
                                  "чемпионат", "турнир", "лига", "игрок", "гол", "очко", "счет", "счёт", "статистика"]
                
                # Приоритет выбора инструмента
                selected_tool = None
                
                # Проверяем запрос о погоде
                if any(keyword in first_user_message for keyword in weather_keywords):
                    for tool in self.tools:
                        tool_name = tool.get('name', None) if isinstance(tool, dict) else getattr(tool, 'name', None)
                        if tool_name == 'get_weather':
                            selected_tool = 'get_weather'
                            self.logger.info("Выбран инструмент get_weather на основе контекста запроса")
                            break
                
                # Если это запрос о спорте, используем web_search
                if not selected_tool and any(keyword in first_user_message for keyword in sports_keywords):
                    for tool in self.tools:
                        tool_name = tool.get('name', None) if isinstance(tool, dict) else getattr(tool, 'name', None)
                        if tool_name == 'web_search':
                            selected_tool = 'web_search'
                            self.logger.info("Выбран инструмент web_search для спортивной тематики")
                            break
                
                # Если это не запрос о погоде, проверяем запрос на поиск в интернете
                if not selected_tool and any(keyword in first_user_message for keyword in search_keywords):
                    for tool in self.tools:
                        tool_name = tool.get('name', None) if isinstance(tool, dict) else getattr(tool, 'name', None)
                        if tool_name == 'web_search':
                            selected_tool = 'web_search'
                            self.logger.info("Выбран инструмент web_search на основе контекста запроса")
                            break
                
                # По умолчанию используем get_docs, если это запрос о документации или если другие инструменты не выбраны
                if not selected_tool:
                    for tool in self.tools:
                        tool_name = tool.get('name', None) if isinstance(tool, dict) else getattr(tool, 'name', None)
                        if tool_name == 'get_docs':
                            selected_tool = 'get_docs'
                            self.logger.info("Выбран инструмент get_docs на основе контекста запроса")
                            break
                
                # Устанавливаем выбранный инструмент в tool_choice
                if selected_tool:
                    tool_choice = {
                        "type": "function",
                        "function": {"name": selected_tool}
                    }
            
            # Вызываем LLM с указанием использовать инструменты
            response = self.llm.chat.completions.create(
                model="google/gemini-2.5-flash-preview:thinking",
                max_tokens=1000,
                messages=messages_with_system,
                tools=self.tools,
                tool_choice=tool_choice if tool_choice else "auto",
                temperature=0.0,  # Минимальная температура для детерминированных ответов
            )
            self.logger.info(f"LLM response: {response}")
            
            message = response.choices[0].message
            
            # Очищаем текстовый ответ от служебных пояснений и размышлений
            if message.content:
                message.content = self._clean_llm_response(message.content)
                
            return message
        except Exception as e:
            self.logger.error(f"Error calling LLM: {e}")
            raise

    def _clean_llm_response(self, text: str) -> str:
        """
        Очищает ответ LLM от служебных данных и размышлений моделей Gemini.
        """
        if not text:
            return text
            
        # Проверяем на незавершенные размышления без полезного содержимого
        if "thinking" in text.lower() or "expected outcome" in text.lower() or "tool call" in text.lower():
            # Возвращаем сообщение о необходимости искать в интернете
            return "Для получения этой информации требуется поиск в интернете. Пожалуйста, уточните запрос."
            
        # Удаляем блоки с размышлениями и служебными данными в маркдауне
        lines = text.split('\n')
        cleaned_lines = []
        skip_block = False
        
        for line in lines:
            lower_line = line.lower().strip()
            
            # Пропускаем блоки, начинающиеся с "**" (заголовки размышлений)
            if line.strip().startswith('**') and '**' in line.strip()[2:]:
                skip_block = True
                continue
                
            # Пропускаем строки с явными маркерами размышлений
            if any(marker in lower_line for marker in [
                "thinking", "assessing", "investigating", "analyzing", "reviewing",
                "i'm", "i'll", "i will", "i need", "i should", "i must", "considering",
                "using tool", "the user wants", "expected outcome", "query:", "tool:"
            ]):
                skip_block = True
                continue
                
            # Пропускаем пустые строки после заголовка размышлений
            if skip_block and not line.strip():
                continue
                
            # Прекращаем пропуск блока, когда встречаем содержательный текст (не маркер размышлений)
            if skip_block and line.strip() and not any(marker in lower_line for marker in [
                "thinking", "assessing", "query:", "tool:", "i'm", "i'll", "i will", "i need"
            ]):
                skip_block = False
                
            # Добавляем строку, если она не в пропускаемом блоке
            if not skip_block:
                cleaned_lines.append(line)
        
        # Соединяем строки обратно в текст
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Удаляем упоминания инструментов и метаданные
        phrases_to_remove = [
            "Результат инструмента:",
            "Анализирую данные",
            "Получена информация",
            "Исходя из данных",
            "Согласно полученной информации",
            "Вот информация",
            "Вот результат",
            "Согласно ответу инструмента",
            "На основе полученных данных",
        ]
        
        for phrase in phrases_to_remove:
            if cleaned_text.lower().startswith(phrase.lower()):
                # Удаляем фразу и следующий за ней текст до первого содержательного блока
                parts = cleaned_text.split('\n\n', 1)
                if len(parts) > 1:
                    cleaned_text = parts[1]
                else:
                    # Если разделителей нет, просто убираем первую строку
                    lines = cleaned_text.split('\n')
                    if len(lines) > 1:
                        cleaned_text = '\n'.join(lines[1:])
        
        # Удаляем начальные пустые строки
        cleaned_text = cleaned_text.lstrip()
        
        # Если после очистки текст стал пустым или слишком коротким, вернем сообщение по умолчанию
        if not cleaned_text or len(cleaned_text) < 20:
            return "Для получения этой информации требуется дополнительный поиск. Пожалуйста, уточните запрос."
        
        return cleaned_text

    # cleanup
    async def cleanup(self):
        try:
            if self.connect_mode == "http" and self.http_client:
                await self.http_client.aclose()
                self.logger.info("Closed HTTP connection to MCP server")
            else:
                await self.exit_stack.aclose()
                self.logger.info("Disconnected from MCP server")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            traceback.print_exc()
            raise

    async def log_conversation(self):
        os.makedirs("conversations", exist_ok=True)

        serializable_conversation = []

        for message in self.messages:
            try:
                serializable_message = {"role": message["role"], "content": []}

                # Handle both string and list content
                if isinstance(message["content"], str):
                    serializable_message["content"] = message["content"]
                elif isinstance(message["content"], list):
                    for content_item in message["content"]:
                        if hasattr(content_item, "to_dict"):
                            serializable_message["content"].append(
                                content_item.to_dict()
                            )
                        elif hasattr(content_item, "dict"):
                            serializable_message["content"].append(content_item.dict())
                        elif hasattr(content_item, "model_dump"):
                            serializable_message["content"].append(
                                content_item.model_dump()
                            )
                        else:
                            serializable_message["content"].append(content_item)

                serializable_conversation.append(serializable_message)
            except Exception as e:
                self.logger.error(f"Error processing message: {str(e)}")
                self.logger.debug(f"Message content: {message}")
                raise

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = os.path.join("conversations", f"conversation_{timestamp}.json")

        try:
            with open(filepath, "w") as f:
                json.dump(serializable_conversation, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error writing conversation to file: {str(e)}")
            self.logger.debug(f"Serializable conversation: {serializable_conversation}")
            raise
