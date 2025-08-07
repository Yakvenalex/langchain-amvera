#!/usr/bin/env python3
"""Отладочный пример Function Calling с подробным выводом."""

import json
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_amvera import AmveraLLM


def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Получить текущую погоду для указанного местоположения."""
    # Имитация API погоды
    weather_data = {
        "moscow": {"temperature": -2, "condition": "snow", "humidity": 80},
        "london": {"temperature": 8, "condition": "rainy", "humidity": 75},
        "tokyo": {"temperature": 15, "condition": "cloudy", "humidity": 60},
        "new york": {"temperature": 12, "condition": "sunny", "humidity": 45},
    }
    
    location_lower = location.lower()
    if location_lower in weather_data:
        data = weather_data[location_lower]
        temp = data["temperature"]
        
        if unit == "fahrenheit":
            temp = temp * 9/5 + 32
            unit_symbol = "°F"
        else:
            unit_symbol = "°C"
        
        return json.dumps({
            "location": location,
            "temperature": f"{temp}{unit_symbol}",
            "condition": data["condition"],
            "humidity": f"{data['humidity']}%",
            "timestamp": datetime.now().isoformat()
        })
    else:
        return json.dumps({
            "error": f"Данные о погоде для {location} не найдены"
        })


def main():
    """Отладочная демонстрация Function Calling."""
    print("🔧 Отладочный пример Function Calling с Amvera LLM\n")
    
    # Упрощенные инструменты для тестирования
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Получить текущую погоду для указанного города",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Название города"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "default": "celsius"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    # Создаем модель с отладкой
    llm = AmveraLLM(
        model="llama70b",
        temperature=0.3,
        tools=tools,
        verbose=True  # Включаем детальное логирование
    )
    
    # Простой запрос для тестирования
    query = "Какая погода в Москве?"
    
    print(f"📝 Тестовый запрос: {query}")
    print("-" * 50)
    
    messages = [
        SystemMessage(content="Ты помощник с доступом к функции погоды. Используй функцию get_current_weather для получения данных о погоде."),
        HumanMessage(content=query)
    ]
    
    try:
        print("🚀 Отправка запроса...")
        result = llm.invoke(messages)
        
        print(f"\n✅ Результат получен")
        print(f"📄 Тип результата: {type(result)}")
        print(f"📄 Содержимое: {result.content}")
        
        # Проверяем атрибуты результата
        print(f"\n🔍 Атрибуты результата:")
        for attr in dir(result):
            if not attr.startswith('_'):
                try:
                    value = getattr(result, attr)
                    if not callable(value):
                        print(f"  - {attr}: {value}")
                except:
                    print(f"  - {attr}: <не удалось получить>")
        
        # Проверяем метаданные ответа
        if hasattr(result, 'response_metadata'):
            print(f"\n📊 Метаданные ответа: {result.response_metadata}")
        
        # Проверяем tool calls
        if hasattr(result, 'tool_calls'):
            print(f"\n🔧 Tool calls: {result.tool_calls}")
        else:
            print(f"\n❌ Атрибут tool_calls отсутствует")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print(f"🔍 Тип ошибки: {type(e)}")
        import traceback
        print("📋 Полная трассировка:")
        traceback.print_exc()


if __name__ == "__main__":
    main()