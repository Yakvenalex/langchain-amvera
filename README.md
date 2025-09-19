# LangChain Amvera

Интеграция LangChain для работы с Amvera LLM API.

## Установка

```bash
pip install langchain-amvera
```

## Быстрый старт

```python
from langchain_amvera import AmveraLLM
from langchain_core.messages import HumanMessage

# Инициализация модели
llm = AmveraLLM(
    model="llama70b",  # или "gpt-4.1", "gpt-5"
    temperature=0.7,
    api_token="your-token"  # или через переменную AMVERA_API_TOKEN
)

# Синхронное использование
response = llm.invoke([HumanMessage(content="Привет!")])
print(response.content)

# Асинхронное использование  
import asyncio

async def main():
    response = await llm.ainvoke([HumanMessage(content="Привет!")])
    print(response.content)
    await llm.aclose()  # Закрытие соединений

asyncio.run(main())
```

## Поддерживаемые модели

### LLaMA модели
- `llama8b` - Llama 3.1 8B модель
- `llama70b` - Llama 3.1 70B модель (по умолчанию)

### GPT модели
- `gpt-4.1` - модель GPT-4.1 от OpenAI
- `gpt-5` - модель GPT-5 от OpenAI

## Конфигурация

### Переменные окружения

Создайте файл `.env`:

```env
AMVERA_API_TOKEN=your-amvera-api-token
```

### Параметры инициализации

```python
llm = AmveraLLM(
    model="llama70b",           # Модель: llama8b, llama70b, gpt-4.1, gpt-5
    temperature=0.7,            # Параметр творчества (0.0-2.0)
    max_tokens=1000,            # Максимальное количество токенов
    timeout=60,                 # Таймаут запроса в секундах
    verbose=True,               # Детальное логирование
    json_mode=False,            # JSON режим ответа
    json_schema={},             # JSON схема для валидации
    tools=None,                 # Инструменты для function calling
    api_token="your-token",     # API токен
    base_url="https://..."      # Базовый URL API
)
```

## Возможности

### Function Calling

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Определяем схему аргументов
class WeatherArgs(BaseModel):
    location: str = Field(description="Название города")
    unit: str = Field(default="celsius", description="Единицы температуры")

# Создаем инструмент с декоратором @tool
@tool("get_weather", args_schema=WeatherArgs)
def get_weather(location: str, unit: str = "celsius") -> str:
    """Получить прогноз погоды для указанного города."""
    # Ваша логика здесь
    return f"Погода в {location}: солнечно, {unit}"

# Привязываем инструменты к модели  
llm = AmveraLLM(model="llama70b")  # или "gpt-4.1", "gpt-5"
llm_with_tools = llm.bind_tools([get_weather])

# Использование
response = llm_with_tools.invoke([HumanMessage(content="Какая погода в Москве?")])
```

### JSON режим

```python
llm = AmveraLLM(
    model="llama70b",
    json_mode=True,
    json_schema={
        "type": "object",
        "properties": {
            "answer": {"type": "string"}
        }
    }
)
```

### Получение информации об использовании

```python
response = llm.invoke([HumanMessage(content="Привет!")])

# Информация о токенах
usage = llm.get_token_usage(response)
print(f"Использовано токенов: {usage}")

# Версия модели
version = llm.get_model_version(response)
print(f"Версия модели: {version}")
```

## Примеры использования

Смотрите папку [examples/](examples/) для подробных примеров:

- [Базовый пример](examples/basic_usage.py)
- [Асинхронное использование](examples/async_usage.py) 
- [Function calling](examples/function_calling.py)
- [JSON режим](examples/json_mode.py)
- [Интеграция с LangChain](examples/langchain_integration.py)

## Требования

- Python 3.8+
- langchain-core >= 0.1.0
- langchain >= 0.1.0
- httpx >= 0.24.0
- pydantic >= 2.0.0
- python-dotenv >= 0.19.0

## Разработка

### Установка для разработки

```bash
git clone https://github.com/amvera-ru/langchain-amvera.git
cd langchain-amvera
pip install -e ".[dev]"
```

### Запуск тестов

```bash
pytest
```

### Форматирование кода

```bash
black langchain_amvera/
ruff langchain_amvera/
```

## Лицензия

MIT License. Смотрите [LICENSE](LICENSE) для подробностей.

## Поддержка

- [Документация Amvera](https://docs.amvera.ru)
- [GitHub Issues](https://github.com/amvera-ru/langchain-amvera/issues)
- Email: support@amvera.ru