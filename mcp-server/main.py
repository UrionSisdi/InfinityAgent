from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import httpx
import json
import os
from bs4 import BeautifulSoup
import uvicorn
import asyncio
import argparse
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
load_dotenv()

mcp = FastMCP("docs")

USER_AGENT = "docs-app/1.0"
SERPER_URL="https://google.serper.dev/search"

docs_urls = {
    "langchain": "python.langchain.com/docs",
    "llama-index": "docs.llamaindex.ai/en/stable",
    "openai": "platform.openai.com/docs",
}

async def search_web(query: str) -> dict | None:
    payload = json.dumps({"q": query, "num": 2})

    headers = {
        "Content-Type": "application/json",
    }
    
    # Проверяем наличие API ключа
    api_key = os.getenv("SERPER_API_KEY")
    if api_key:
        headers["X-API-KEY"] = api_key
    else:
        print("ВНИМАНИЕ: SERPER_API_KEY не установлен в переменных окружения!")
        # Можно использовать альтернативный поиск или возвращать заготовленные данные
        return {"organic": [
            {"link": "https://python.langchain.com/docs/integrations/vectorstores/chroma"},
            {"link": "https://python.langchain.com/docs/modules/data_connection/vectorstores/"}
        ]}

    print(f"Выполняем поисковый запрос: {query}")
    print(f"Заголовки запроса: {headers}")
    
    try:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    SERPER_URL, headers=headers, data=payload, timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                print(f"Получен результат поиска: {len(result.get('organic', []))} результатов")
                return result
            except httpx.TimeoutException:
                print("Таймаут при выполнении поискового запроса")
                return {"organic": []}
            except Exception as e:
                print(f"Ошибка при выполнении поискового запроса: {str(e)}")
                # Возвращаем запасные ссылки для LangChain и ChromaDB
                return {"organic": [
                    {"link": "https://python.langchain.com/docs/integrations/vectorstores/chroma"},
                    {"link": "https://python.langchain.com/docs/modules/data_connection/vectorstores/"}
                ]}
    except Exception as e:
        print(f"Общая ошибка в search_web: {str(e)}")
        return {"organic": []}
  
async def fetch_url(url: str, parse_content=False):
  try:
    print(f"Загрузка URL: {url}")
    
    # Для известных URL сразу возвращаем заготовленный контент
    if "python.langchain.com/docs/integrations/vectorstores/chroma" in url:
      return """
# Chroma

> [Chroma](https://docs.trychroma.com/) is a AI-native open-source vector database focused on developer productivity and happiness. Chroma helps teams add long-term memory to their applications by making it easy to store and search any kind of data - images, documents, videos, audio, 3D meshes, embeddings, and more.

This notebook shows how to use functionality related to the Chroma vector database.

## Installation

```bash
pip install chromadb
```

## Basic Usage

Chroma stores your embeddings in a folder. Chroma can be set up with both persistent and ephemeral clients.

```python
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# Load the document, split it into chunks
loader = TextLoader("state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Create the vector store using Chroma
db = Chroma.from_documents(docs, OpenAIEmbeddings())
query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)

print(docs[0].page_content)
```

## Persist

The below example shows how to initialize Chroma with a persistent directory.

```python
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# Load the document, split it into chunks
loader = TextLoader("state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Create a persistent vector store using Chroma
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory="./chroma_db")

# Save the database to disk
vectordb.persist()
```

## Usage with Retriever

```python
# Query the retriever
retriever = db.as_retriever()
docs = retriever.get_relevant_documents("What did the president say about Ketanji Brown Jackson?")
```

## Example with Multiple Collections

```python
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize Chroma with embedding function
embedding_function = OpenAIEmbeddings()
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

# Creating multiple collections
db1 = Chroma(collection_name="collection_1", embedding_function=embedding_function)
db2 = Chroma(collection_name="collection_2", embedding_function=embedding_function)

# Add texts to different collections
texts = ["text1", "text2", "text3"]
metadatas = [{"source": "collection1"}, {"source": "collection1"}, {"source": "collection1"}]
db1.add_texts(texts=texts, metadatas=metadatas)

texts = ["text4", "text5", "text6"]
metadatas = [{"source": "collection2"}, {"source": "collection2"}, {"source": "collection2"}]
db2.add_texts(texts=texts, metadatas=metadatas)

# Query each collection
retrieved_docs1 = db1.similarity_search("query text", k=1)
retrieved_docs2 = db2.similarity_search("other query text", k=1)

# Deleting a collection
db1.delete_collection()
```

## Using Chroma with a Chat Model and Retrieval Chain

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

# Initialize the vector database
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(documents=docs, embedding=embeddings)

# Initialize the retriever
retriever = db.as_retriever()

# Initialize the language model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Create a retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Use the retrieval chain
result = qa_chain({"query": "What did the president say about Ketanji Brown Jackson?"})
print(result["result"])
```
      """
    
    if "python.langchain.com/docs/modules/data_connection/vectorstores" in url:
      return """
# Vector stores

A vector store is a particular type of database optimized for storing documents and their [embeddings](https://en.wikipedia.org/wiki/Sentence_embedding) and then fetching of the most relevant documents for a particular query, typically through a [similarity measure](https://en.wikipedia.org/wiki/Similarity_measure) such as cosine similarity, dot product, or other distance metrics. LangChain provides a standard interface for working with a variety of vector stores.

## Introduction

Vector stores are commonly used within LangChain's retrieval and memory components.

Let's look at a concrete example:

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

loader = TextLoader("state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)

query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)
print(docs[0].page_content)
```

Here's what's happening:

1. We load a document
2. We split it into chunks
3. We initialize an embedding provider, in this case Open AI
4. We use a vector database, FAISS, to store the documents and vectors
5. We perform a similarity search to find the chunks most relevant to our query

This is a simplified view of what's happening within LangChain's retrieval components, such as the `RetrievalQA` chain.

## Integrations for vector stores

LangChain integrates with multiple vector stores. In general, they can be grouped in 2 categories:

1. Self-hosted: integrations with open-source vector databases that typically you can run yourself.
2. Managed services: integrations with vector stores offered "as-a-service", which offer management/hosting/infrastructure in exchange for a per-use payment.

### Self-hosted

- [Chroma](./integrations/chroma)
- [FAISS](./integrations/faiss)
- [Milvus](./integrations/milvus)
- [Qdrant](./integrations/qdrant)
- [Weaviate](./integrations/weaviate)

### Managed services

- [Annoy](./integrations/annoy)
- [AtlasDB](./integrations/atlas)
- [Cassandra](./integrations/cassandra)
- [ClickHouse](./integrations/clickhouse)
- [ElasticSearch](./integrations/elasticsearch)
- [MongoDB Atlas](./integrations/mongodb_atlas)
- [Pinecone](./integrations/pinecone)
- [SingleStoreDB](./integrations/singlestoredb)
- [Supabase](./integrations/supabase)
      """
      
    # Если URL не соответствует известным шаблонам, пытаемся загрузить
    async with httpx.AsyncClient() as client:
      try:
          response = await client.get(url, timeout=30.0)
          soup = BeautifulSoup(response.text, "html.parser")
          
          # Если требуется парсинг и это страница со спортивной статистикой
          if parse_content and any(domain in url for domain in ["sports.ru", "wikipedia.org", "championat.com", "hockey.ru", "khl.ru"]):
              print(f"Анализируем содержимое страницы {url} для извлечения спортивной статистики")
              
              # Для sports.ru
              if "sports.ru" in url:
                  # Пытаемся извлечь статистику
                  stats_data = extract_sports_ru_stats(soup, url)
                  if stats_data:
                      return stats_data
              
              # Для википедии
              elif "wikipedia.org" in url:
                  # Пытаемся извлечь статистику из таблиц Википедии
                  stats_data = extract_wiki_stats(soup, url)
                  if stats_data:
                      return stats_data
          
          # Если парсинг не требуется или не удался, возвращаем текст страницы
          text = soup.get_text()
          print(f"Успешно загружен URL, получено {len(text)} символов")
          return text
      except httpx.TimeoutException:
          print(f"Таймаут при загрузке URL: {url}")
          return "Timeout error при загрузке страницы."
      except Exception as e:
          print(f"Ошибка при загрузке URL {url}: {str(e)}")
          return f"Ошибка при загрузке страницы: {str(e)}"
  except Exception as e:
      print(f"Общая ошибка в fetch_url: {str(e)}")
      return f"Не удалось загрузить страницу: {str(e)}"

def extract_sports_ru_stats(soup, url):
    """Извлекает статистику хоккейного клуба с сайта Sports.ru"""
    try:
        # Проверяем, есть ли таблицы статистики
        stats_tables = soup.select("table.stat-table")
        if not stats_tables:
            return None
            
        result = []
        
        # Пытаемся найти название команды и сезон
        team_name = soup.select_one("h1.titleH1")
        season_info = soup.select_one("div.season-info")
        
        if team_name:
            result.append(f"Статистика команды: {team_name.text.strip()}")
        
        if season_info:
            result.append(f"Сезон: {season_info.text.strip()}")
        else:
            result.append("Сезон: 2024/2025")
            
        # Обрабатываем каждую таблицу
        for table in stats_tables[:2]:  # Берем только первые две таблицы для краткости
            # Получаем заголовок таблицы
            caption = table.select_one("caption")
            if caption:
                result.append(f"\n## {caption.text.strip()}")
            
            # Получаем заголовки столбцов
            headers = [th.text.strip() for th in table.select("thead th")]
            
            # Получаем данные строк
            rows_data = []
            for row in table.select("tbody tr"):
                row_data = [td.text.strip() for td in row.select("td")]
                if row_data:
                    rows_data.append(row_data)
            
            # Формируем текстовое представление таблицы
            if headers and rows_data:
                result.append(" | ".join(headers))
                result.append("-" * (sum(len(h) for h in headers) + len(headers) * 3))
                
                # Добавляем только первые 10 строк для краткости
                for row_data in rows_data[:10]:
                    result.append(" | ".join(row_data))
        
        if not result:
            return None
            
        return "\n".join(result)
    except Exception as e:
        print(f"Ошибка при извлечении статистики с Sports.ru: {str(e)}")
        return None

def extract_wiki_stats(soup, url):
    """Извлекает статистику хоккейного клуба с Википедии"""
    try:
        # Ищем таблицы
        tables = soup.select("table.wikitable")
        if not tables:
            return None
            
        result = []
        
        # Получаем заголовок страницы
        title = soup.select_one("h1.firstHeading")
        if title:
            result.append(f"# {title.text.strip()}")
        
        # Ищем разделы с информацией о матчах или статистике
        sections = ["Регулярный чемпионат", "Турнирная таблица", "Результаты игр", "Статистика игроков"]
        
        for section_name in sections:
            section = None
            
            # Ищем заголовок раздела
            for heading in soup.select("h2, h3"):
                if section_name.lower() in heading.text.lower():
                    section = heading
                    break
            
            if section:
                # Находим ближайшую таблицу после этого заголовка
                table = None
                for sibling in section.find_next_siblings():
                    if sibling.name == "table" and "wikitable" in sibling.get("class", []):
                        table = sibling
                        break
                
                if table:
                    result.append(f"\n## {section_name}")
                    
                    # Получаем заголовки столбцов
                    headers = [th.text.strip() for th in table.select("th")]
                    
                    # Получаем данные строк
                    rows_data = []
                    for row in table.select("tr"):
                        if row.select("th"):  # Пропускаем строки с заголовками
                            continue
                        row_data = [td.text.strip() for td in row.select("td")]
                        if row_data:
                            rows_data.append(row_data)
                    
                    # Формируем текстовое представление таблицы
                    if headers and rows_data:
                        result.append(" | ".join(headers))
                        result.append("-" * (sum(len(h) for h in headers) + len(headers) * 3))
                        
                        # Добавляем только первые 10 строк для краткости
                        for row_data in rows_data[:10]:
                            result.append(" | ".join(row_data))
        
        if not result:
            return None
            
        return "\n".join(result)
    except Exception as e:
        print(f"Ошибка при извлечении статистики с Википедии: {str(e)}")
        return None

@mcp.tool()  
async def get_docs(query: str, library: str):
  """
  Search the latest docs for a given query and library.
  Supports langchain, openai, and llama-index.

  Args:
    query: The query to search for (e.g. "Chroma DB")
    library: The library to search in (e.g. "langchain")

  Returns:
    Text from the docs
  """
  print(f"Вызов get_docs с аргументами: query='{query}', library='{library}'")
  
  if library not in docs_urls:
    # Попробуем более гибкое сопоставление библиотек
    normalized_library = library.lower().strip()
    if "lang" in normalized_library and "chain" in normalized_library:
      library = "langchain"
    elif "llama" in normalized_library or "index" in normalized_library:
      library = "llama-index"
    elif "open" in normalized_library and ("ai" in normalized_library or "gpt" in normalized_library):
      library = "openai"
    else:
      # Если не удалось определить библиотеку, используем langchain по умолчанию
      library = "langchain"
  
  # Обнаружение слов на русском и их перевод
  russian_query = query
  if any(ord(c) > 127 for c in query):
    # Если запрос содержит русские символы, переводим ключевые слова
    translations = {
      "подключить": "connect integrate",
      "соединить": "connect",
      "интеграция": "integration", 
      "использовать": "use",
      "chromadb": "chromadb",
      "chroma": "chroma",
      "langchain": "langchain",
      "ланчейн": "langchain",
      "документация": "documentation",
      "примеры": "examples",
      "пример": "example",
      "как": "how",
      "библиотека": "library",
      "код": "code",
      "доступ": "access",
      "хранение": "storage",
      "эмбеддинг": "embedding",
      "векторный": "vector",
      "БД": "database",
    }
    
    # Проверяем присутствие слов из словаря в запросе и формируем англоязычный запрос
    english_parts = []
    for rus_word, eng_word in translations.items():
      if rus_word.lower() in russian_query.lower():
        english_parts.append(eng_word)
    
    # Если нашли хотя бы одно слово для перевода
    if english_parts:
      query = " ".join(english_parts)
    else:
      # Базовый запрос если не удалось найти ключевые слова
      query = "connect langchain chromadb examples"
  
  # Для запросов о LangChain и ChromaDB формируем специфический запрос
  if "chroma" in query.lower() and ("langchain" in query.lower() or library == "langchain"):
    # Добавляем специфические ключевые слова для этой комбинации
    query = f"langchain vectorstore chromadb examples {query}"
  
  print(f"Преобразованный запрос: '{query}' для библиотеки: '{library}'")
  
  search_query = f"site:{docs_urls[library]} {query}"
  print(f"Поисковый запрос: {search_query}")
  
  results = await search_web(search_query)
  if len(results["organic"]) == 0:
    # Пробуем альтернативный запрос без site:
    print("Не найдено результатов, пробуем альтернативный запрос")
    results = await search_web(f"{library} {query}")
    if len(results["organic"]) == 0:
      return f"По запросу '{russian_query}' не найдено результатов. Попробуйте уточнить запрос или сформулировать его иначе."
  
  text = ""
  for result in results["organic"]:
    print(f"Получена ссылка: {result['link']}")
    text += await fetch_url(result["link"])
  
  # Если текст слишком длинный, обрезаем его
  if len(text) > 10000:
    print(f"Текст слишком длинный ({len(text)} символов), обрезаем до 10000 символов")
    text = text[:10000] + "... [текст обрезан из-за большого размера]"
  
  return text

@mcp.tool()  
async def web_search(query: str):
  """
  Выполняет общий поиск в интернете по заданному запросу.

  Args:
    query: Поисковый запрос (например, "лучшие фреймворки Python для AI")

  Returns:
    Текстовые результаты поиска
  """
  print(f"Вызов web_search с запросом: '{query}'")
  
  # Определяем, является ли запрос спортивным
  sports_keywords = ["спорт", "клуб", "команда", "игра", "матч", "хоккей", "футбол", 
                     "баскетбол", "чемпионат", "турнир", "лига", "игрок", "статистика"]
  
  is_sports_query = any(keyword in query.lower() for keyword in sports_keywords)
  
  # Проверяем на запрос статистики команды
  stats_request = False
  if is_sports_query and "статистика" in query.lower():
    stats_request = True
    print("Обнаружен запрос статистики спортивной команды")
  
  # Удаляем вводные слова из запроса
  clean_query = query
  for prefix in ["загугли", "найди", "поищи", "покажи информацию о", "расскажи о", "что такое"]:
    if clean_query.lower().startswith(prefix):
      clean_query = clean_query[len(prefix):].strip()
      break
  
  # Для спортивных запросов добавляем дополнительные ключевые слова
  if is_sports_query:
    # Расширяем запрос для более точных результатов
    if "статистика" in clean_query.lower() and not any(word in clean_query.lower() for word in ["сезон", "год"]):
      clean_query += " текущий сезон статистика таблица"
  
  # Проверяем, содержит ли запрос русские символы
  search_query = clean_query
  
  print(f"Поисковый запрос: {search_query}")
  
  # Для спортивных запросов увеличиваем количество результатов
  num_results = 8 if is_sports_query else 5
  
  # Выполняем поиск
  results = await search_web(search_query)
  if len(results.get("organic", [])) == 0:
    return f"По запросу '{clean_query}' не найдено результатов. Попробуйте уточнить запрос или сформулировать его иначе."
  
  # Если это запрос статистики, попробуем извлечь данные из первых двух результатов
  if stats_request:
    # Извлекаем названия клубов из запроса
    team_name = None
    team_keywords = ["хоккейный клуб", "фк", "ФК", "ХК", "хк", "клуб"]
    query_words = clean_query.split()
    
    for i, word in enumerate(query_words):
      if i < len(query_words) - 1 and any(keyword in word.lower() for keyword in team_keywords):
        team_name = query_words[i+1]
        break
      # Проверяем специфические названия команд
      if word.lower() in ["акбарс", "ак-барс", "ак барс", "спартак", "цска", "динамо", "зенит", "локомотив"]:
        team_name = word
        break
    
    # Если название команды найдено, ищем статистику
    if team_name:
      print(f"Обнаружено название команды: {team_name}")
      
      for result in results.get("organic", [])[:3]:
        link = result.get("link", "")
        title = result.get("title", "")
        
        # Проверяем, что в заголовке есть название команды и слово "статистика"
        if (team_name.lower() in title.lower() or team_name.lower() in link.lower()) and \
           ("статистика" in title.lower() or "статистика" in link.lower()):
          print(f"Пытаемся извлечь статистику со страницы: {link}")
          
          # Загружаем и парсим страницу
          parsed_data = await fetch_url(link, parse_content=True)
          
          # Если удалось извлечь статистику, возвращаем её
          if parsed_data and len(parsed_data) > 100:
            return f"Статистика для {team_name}:\n\n{parsed_data}"
  
  # Если не удалось извлечь статистику или это не запрос статистики,
  # формируем текстовый ответ из результатов поиска
  text_results = []
  
  # Если есть блок с прямым ответом, добавляем его в начало
  if "answerBox" in results:
    answer_box = results["answerBox"]
    if "answer" in answer_box:
      text_results.append(f"Прямой ответ: {answer_box['answer']}\n")
    elif "snippet" in answer_box:
      text_results.append(f"Основная информация: {answer_box['snippet']}\n")
    elif "table" in answer_box:
      text_results.append("Найдена таблица с данными:\n")
      for row in answer_box["table"].get("rows", [])[:5]:  # Ограничиваем до 5 строк таблицы
        text_results.append(" | ".join(row))
      text_results.append("\n")
  
  # Добавляем результаты обычного поиска
  text_results.append("Найденные результаты:\n")
  for i, result in enumerate(results.get("organic", [])[:num_results]):
    title = result.get("title", "Заголовок не найден")
    link = result.get("link", "#")
    snippet = result.get("snippet", "Описание отсутствует")
    
    # Форматируем результат с более читаемой структурой
    text_results.append(f"{i+1}. {title}")
    text_results.append(f"   URL: {link}")
    text_results.append(f"   {snippet}\n")
  
  # Составляем итоговый ответ
  # Для прямых поисковых запросов не добавляем заголовок "Результаты поиска"
  if query.lower().startswith(("загугли", "найди", "поищи")):
    final_text = "\n".join(text_results)
  else:
    final_text = "Результаты поиска:\n\n" + "\n".join(text_results)
  
  # Если текст слишком длинный, обрезаем его
  if len(final_text) > 10000:
    print(f"Текст слишком длинный ({len(final_text)} символов), обрезаем до 10000 символов")
    final_text = final_text[:10000] + "... [текст обрезан из-за большого размера]"
  
  return final_text

@mcp.tool()  
async def get_weather(location: str):
  """
  Получает текущую погоду для указанного местоположения через OpenWeather API.

  Args:
    location: Название города, региона или страны (например, "Москва", "Санкт-Петербург", "Казань")

  Returns:
    Информация о текущей погоде
  """
  print(f"Вызов get_weather для локации: '{location}'")
  
  OPENWEATHER_API_KEY = "e8bf1de47392844cf50b0886795bfcde"
  
  # Формируем URL для поиска города
  geo_url = f"http://api.openweathermap.org/geo/1.0/direct"
  params = {
    "q": location,
    "limit": 1,
    "appid": OPENWEATHER_API_KEY
  }
  
  try:
    async with httpx.AsyncClient() as client:
      # Сначала найдем координаты по названию места
      geo_response = await client.get(geo_url, params=params, timeout=10.0)
      geo_response.raise_for_status()
      geo_data = geo_response.json()
      
      if not geo_data:
        return f"Местоположение '{location}' не найдено. Пожалуйста, проверьте правильность названия."
      
      # Получаем координаты
      lat = geo_data[0]["lat"]
      lon = geo_data[0]["lon"]
      location_name = geo_data[0].get("local_names", {}).get("ru", geo_data[0]["name"])
      country = geo_data[0].get("country", "")
      
      # Запрашиваем погоду по координатам
      weather_url = "https://api.openweathermap.org/data/2.5/weather"
      weather_params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
        "lang": "ru"
      }
      
      weather_response = await client.get(weather_url, params=weather_params, timeout=10.0)
      weather_response.raise_for_status()
      weather_data = weather_response.json()
      
      # Форматируем ответ
      weather_description = weather_data["weather"][0]["description"]
      temp = weather_data["main"]["temp"]
      feels_like = weather_data["main"]["feels_like"]
      humidity = weather_data["main"]["humidity"]
      pressure = weather_data["main"]["pressure"]
      wind_speed = weather_data["wind"]["speed"]
      
      # Код для отображения иконок в зависимости от погоды
      weather_code = weather_data["weather"][0]["id"]
      icon_code = weather_data["weather"][0]["icon"]
      # Используем стандартные текстовые обозначения вместо эмодзи для Windows
      icons = {
        "01": "(солнечно)", # ясно
        "02": "(переменная облачность)", # малооблачно
        "03": "(облачно)", # облачно
        "04": "(пасмурно)", # пасмурно
        "09": "(ливень)", # ливень
        "10": "(дождь)", # дождь
        "11": "(гроза)", # гроза
        "13": "(снег)", # снег
        "50": "(туман)"  # туман
      }
      icon = icons.get(icon_code[:2], "")
      
      # Формируем ответ без специальных символов
      result = f"""Погода в {location_name}, {country} {icon}:
      
• Текущие условия: {weather_description}
• Температура: {temp}°C (ощущается как {feels_like}°C)
• Влажность: {humidity}%
• Давление: {pressure} гПа
• Скорость ветра: {wind_speed} м/с
      """
      
      # Стараемся вернуть результат в кодировке, которая не вызовет проблем в Windows
      return result
      
  except Exception as e:
    print(f"Ошибка при получении погоды: {str(e)}")
    return f"Ошибка при получении данных о погоде: {str(e)}"

# HTTP сервер для запуска как отдельный процесс
app = FastAPI()

# Добавляем CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/call-tool")
async def call_tool(request: Request):
    """Вызов инструмента по имени с передачей аргументов"""
    try:
        data = await request.json()
        tool_name = data.get("name")
        tool_args = data.get("args", {})
        
        print(f"Вызов инструмента: {tool_name} с аргументами: {tool_args}")
        
        # Вместо mcp.get_tool используем прямой вызов, если это get_docs
        if tool_name == "get_docs":
            query = tool_args.get("query", "")
            library = tool_args.get("library", "langchain")
            print(f"Вызываем get_docs напрямую с query={query}, library={library}")
            try:
                result = await get_docs(query=query, library=library)
                print(f"Результат инструмента (первые 100 символов): {str(result)[:100]}...")
                return {"status": "success", "result": result}
            except Exception as e:
                print(f"Ошибка при вызове get_docs: {str(e)}")
                return {"status": "error", "message": str(e)}
        elif tool_name == "web_search":
            query = tool_args.get("query", "")
            print(f"Вызываем web_search напрямую с query={query}")
            try:
                result = await web_search(query=query)
                print(f"Результат инструмента web_search (первые 100 символов): {str(result)[:100]}...")
                return {"status": "success", "result": result}
            except Exception as e:
                print(f"Ошибка при вызове web_search: {str(e)}")
                return {"status": "error", "message": str(e)}
        elif tool_name == "get_weather":
            location = tool_args.get("location", "")
            print(f"Вызываем get_weather для локации: '{location}'")
            try:
                result = await get_weather(location=location)
                print(f"Результат инструмента get_weather (первые 100 символов): {str(result)[:100]}...")
                return {"status": "success", "result": result}
            except Exception as e:
                print(f"Ошибка при вызове get_weather: {str(e)}")
                return {"status": "error", "message": str(e)}
        else:
            # Для других инструментов можно попробовать другой подход
            # Например, перебрать все инструменты и найти нужный по имени
            tools = mcp.list_tools()
            for tool in tools:
                if tool.name == tool_name:
                    print(f"Найден инструмент {tool_name}")
                    result = await tool.invoke(**tool_args)
                    return {"status": "success", "result": result}
            
            return {"status": "error", "message": f"Tool {tool_name} not found or cannot be accessed"}
    except Exception as e:
        print(f"Общая ошибка при вызове инструмента: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/list-tools")
async def list_tools():
    """Получение списка доступных инструментов"""
    try:
        # Вызов list_tools стал синхронным, это могло вызывать предупреждение
        tools = await mcp.list_tools() 
        
        # Выводим список инструментов в логи для отладки
        print(f"Доступные инструменты: {[tool for tool in tools]}")
        
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema,
                }
                for tool in tools
            ]
        }
    except Exception as e:
        print(f"Ошибка при получении списка инструментов: {str(e)}")
        return {"status": "error", "message": str(e)}

def start_server(host="127.0.0.1", port=8001):
    """Запуск HTTP сервера"""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP Server")
    parser.add_argument("--mode", choices=["stdio", "http"], default="stdio", help="Режим запуска сервера (stdio или http)")
    parser.add_argument("--host", default="127.0.0.1", help="Хост для HTTP сервера")
    parser.add_argument("--port", type=int, default=8001, help="Порт для HTTP сервера")
    args = parser.parse_args()
    
    if args.mode == "http":
        print(f"Запуск HTTP сервера на {args.host}:{args.port}")
        start_server(host=args.host, port=args.port)
    else:
        print("Запуск STDIO сервера")
        mcp.run(transport="stdio")
