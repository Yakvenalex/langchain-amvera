#!/usr/bin/env python3
"""Пример Function Calling с использованием @tool декоратора и bind_tools()."""

import json
from datetime import datetime
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_amvera import AmveraLLM


# Определяем схемы аргументов для инструментов
class WeatherArgs(BaseModel):
    """Схема аргументов для получения погоды."""
    location: str = Field(
        description="Название города, например 'Moscow' или 'London'")
    unit: str = Field(
        default="celsius", description="Единицы измерения температуры: celsius или fahrenheit")


class DistanceArgs(BaseModel):
    """Схема аргументов для расчета расстояния."""
    city1: str = Field(description="Название первого города")
    city2: str = Field(description="Название второго города")


# Современные инструменты с декоратором @tool
@tool("get_current_weather", args_schema=WeatherArgs)
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


@tool("calculate_distance", args_schema=DistanceArgs)
def calculate_distance(city1: str, city2: str) -> str:
    """Вычислить приблизительное расстояние между городами."""
    # Имитация расчета расстояний
    distances = {
        ("moscow", "london"): 2500,
        ("москва", "лондон"): 2500,
        ("moscow", "tokyo"): 7400,
        ("москва", "токио"): 7400,
        ("london", "tokyo"): 9600,
        ("лондон", "токио"): 9600,
        ("moscow", "new york"): 7500,
        ("москва", "нью-йорк"): 7500,
        ("london", "new york"): 5500,
        ("лондон", "нью-йорк"): 5500,
        ("tokyo", "new york"): 10800,
        ("токио", "нью-йорк"): 10800,
    }

    key1 = (city1.lower(), city2.lower())
    key2 = (city2.lower(), city1.lower())

    distance = distances.get(key1) or distances.get(key2)

    if distance:
        return json.dumps({
            "from": city1,
            "to": city2,
            "distance_km": distance,
            "distance_miles": round(distance * 0.621371, 1)
        })
    else:
        return json.dumps({
            "error": f"Расстояние между {city1} и {city2} не найдено"
        })


def handle_tool_calls(message: AIMessage, tools_dict: Dict[str, Any]) -> List[ToolMessage]:
    """Обрабатывает вызовы инструментов из ответа модели."""
    tool_messages = []

    if hasattr(message, 'tool_calls') and message.tool_calls:
        for tool_call in message.tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})

            print(f"🔧 Вызов инструмента: {tool_name}({tool_args})")

            if tool_name in tools_dict:
                try:
                    # Вызываем функцию
                    result = tools_dict[tool_name].invoke(tool_args)
                    print(f"📊 Результат: {result}")

                    # Создаем ToolMessage для цепочки
                    tool_message = ToolMessage(
                        content=result,
                        tool_call_id=tool_call.get("id", "unknown")
                    )
                    tool_messages.append(tool_message)

                except Exception as e:
                    print(f"❌ Ошибка при вызове {tool_name}: {e}")
                    tool_message = ToolMessage(
                        content=f"Ошибка: {str(e)}",
                        tool_call_id=tool_call.get("id", "unknown")
                    )
                    tool_messages.append(tool_message)
            else:
                print(f"❌ Инструмент {tool_name} не найден")

    return tool_messages


def main():
    """Демонстрация Function Calling."""
    print("🛠️  Пример Function Calling с bind_tools()\n")

    # Список инструментов
    tools = [get_current_weather, calculate_distance]

    # Создаем словарь для быстрого доступа к инструментам
    tools_dict = {tool.name: tool for tool in tools}

    # Базовая модель
    base_llm = AmveraLLM(
        model="llama70b",
        temperature=0.3,
        verbose=True
    )

    # Привязываем инструменты к модели
    llm_with_tools = base_llm.bind_tools(tools)

    print("✅ Инструменты привязаны к модели")

    # Примеры запросов
    test_queries = [
        "Какая сейчас погода в Москве?",
        "Сколько километров от Лондона до Токио?",
        "Расскажи о погоде в Нью-Йорке и какое расстояние от Москвы до Нью-Йорка"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Запрос {i}: {query}")
        print("-" * 50)

        # Создаем сообщения
        messages = [
            SystemMessage(content="""
Ты полезный ИИ-помощник с доступом к инструментам для получения информации о погоде и расчета расстояний.

Когда пользователь спрашивает о погоде или расстояниях, используй доступные инструменты.
После получения данных от инструментов, представь информацию в удобном для пользователя формате.
            """.strip()),
            HumanMessage(content=query)
        ]

        try:
            # Первый вызов модели
            print("🚀 Отправка запроса к модели с инструментами...")
            ai_message = llm_with_tools.invoke(messages)

            print(f"🤖 Ответ модели: {ai_message.content}")

            # Обрабатываем вызовы инструментов
            tool_messages = handle_tool_calls(ai_message, tools_dict)

            if tool_messages:
                # Добавляем результаты инструментов в контекст
                messages.extend([ai_message] + tool_messages)

                # Второй вызов модели для формирования финального ответа
                print("\n🔄 Отправка результатов инструментов для финального ответа...")
                try:
                    # Используем базовую модель без tools
                    final_response = base_llm.invoke(messages)
                    print(f"✨ Финальный ответ: {final_response.content}")
                except Exception as final_error:
                    print(f"⚠️ Ошибка при генерации финального ответа: {final_error}")
                    print("📋 Результаты инструментов:")
                    for i, msg in enumerate(tool_messages, 1):
                        print(f"   {i}. {msg.content}")

            # Информация об использовании
            usage = llm_with_tools.get_token_usage(ai_message)
            if usage:
                print(
                    f"\n📊 Использовано токенов: {usage.get('totalTokens', 'N/A')}")

        except Exception as e:
            print(f"❌ Ошибка при обработке запроса: {e}")
            import traceback
            traceback.print_exc()

        print("\n" + "="*60)


if __name__ == "__main__":
    main()
