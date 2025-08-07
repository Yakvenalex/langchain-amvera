#!/usr/bin/env python3
"""Пример использования JSON режима с langchain-amvera."""

import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_amvera import AmveraLLM


def basic_json_example():
    """Базовый пример JSON режима."""
    print("📝 Пример 1: Базовый JSON режим\n")
    
    llm = AmveraLLM(
        model="llama70b",
        temperature=0.3,
        json_mode=True,  # Включаем JSON режим
        verbose=True
    )
    
    messages = [
        SystemMessage(content="""
Ты помощник, который отвечает только в формате JSON.
Всегда возвращай ответ в виде валидного JSON объекта.
        """),
        HumanMessage(content="Расскажи о Python в 3-х пунктах")
    ]
    
    try:
        result = llm.invoke(messages)
        print(f"🤖 Ответ модели (строка):\n{result.content}")
        
        # Попробуем распарсить JSON
        try:
            json_data = json.loads(result.content)
            print(f"\n✅ Валидный JSON получен:")
            print(json.dumps(json_data, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print("❌ Ответ не является валидным JSON")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")


def structured_json_example():
    """Пример со структурированным JSON ответом."""
    print("\n📝 Пример 2: Структурированный JSON с определенной схемой\n")
    
    # Определяем схему для ответа
    json_schema = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Краткое резюме"
            },
            "details": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "importance": {"type": "integer", "minimum": 1, "maximum": 5}
                    },
                    "required": ["title", "description", "importance"]
                }
            },
            "recommendation": {
                "type": "string"
            }
        },
        "required": ["summary", "details", "recommendation"]
    }
    
    llm = AmveraLLM(
        model="llama70b",
        temperature=0.2,
        json_mode=True,
        json_schema=json_schema,  # Передаем схему
        verbose=True
    )
    
    messages = [
        SystemMessage(content=f"""
Ты эксперт по веб-разработке. Отвечай строго в формате JSON согласно схеме:

{json.dumps(json_schema, indent=2)}

Каждый элемент массива details должен содержать title, description и importance (от 1 до 5).
        """),
        HumanMessage(content="Проанализируй преимущества использования React для фронтенд разработки")
    ]
    
    try:
        result = llm.invoke(messages)
        print(f"🤖 Ответ модели:\n{result.content}")
        
        # Парсим и красиво выводим JSON
        try:
            data = json.loads(result.content)
            print(f"\n✅ Структурированный ответ:")
            print(f"📋 Резюме: {data.get('summary', 'N/A')}")
            
            details = data.get('details', [])
            print(f"\n📊 Детали ({len(details)} пунктов):")
            for i, detail in enumerate(details, 1):
                title = detail.get('title', 'N/A')
                desc = detail.get('description', 'N/A')
                imp = detail.get('importance', 0)
                print(f"  {i}. {title} (важность: {imp}/5)")
                print(f"     {desc}")
            
            rec = data.get('recommendation', 'N/A')
            print(f"\n💡 Рекомендация: {rec}")
            
        except json.JSONDecodeError as e:
            print(f"❌ Ошибка парсинга JSON: {e}")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")


def data_extraction_example():
    """Пример извлечения данных в JSON формате."""
    print("\n📝 Пример 3: Извлечение данных из текста в JSON\n")
    
    llm = AmveraLLM(
        model="llama8b",  # Используем более быструю модель
        temperature=0.1,
        json_mode=True
    )
    
    # Текст для анализа
    sample_text = """
    Компания TechCorp была основана в 2020 году в Москве. 
    CEO компании - Иван Петров, CTO - Мария Сидорова. 
    Компания специализируется на разработке мобильных приложений и веб-сервисов.
    В штате работает 150 сотрудников. Годовой оборот составляет 50 миллионов рублей.
    Офисы находятся в Москве, Санкт-Петербурге и Казани.
    """
    
    messages = [
        SystemMessage(content="""
Извлеки информацию из предоставленного текста и верни в формате JSON.
Структура должна включать:
- company_name: название компании
- founded_year: год основания  
- location: основное местоположение
- leadership: массив с руководителями (name, position)
- business_focus: сфера деятельности
- employees_count: количество сотрудников
- annual_revenue: годовой оборот
- offices: массив с офисами

Если какой-то информации нет - используй null.
        """),
        HumanMessage(content=f"Извлеки данные из текста:\n\n{sample_text}")
    ]
    
    try:
        result = llm.invoke(messages)
        print(f"🤖 Извлеченные данные:\n{result.content}")
        
        # Парсим результат
        try:
            data = json.loads(result.content)
            print(f"\n✅ Структурированные данные о компании:")
            print(f"🏢 Название: {data.get('company_name', 'N/A')}")
            print(f"📅 Основана: {data.get('founded_year', 'N/A')}")
            print(f"📍 Местоположение: {data.get('location', 'N/A')}")
            
            leadership = data.get('leadership', [])
            if leadership:
                print(f"👥 Руководство:")
                for person in leadership:
                    name = person.get('name', 'N/A')
                    pos = person.get('position', 'N/A')
                    print(f"   - {name}: {pos}")
            
            print(f"🎯 Деятельность: {data.get('business_focus', 'N/A')}")
            print(f"👨‍💼 Сотрудников: {data.get('employees_count', 'N/A')}")
            print(f"💰 Оборот: {data.get('annual_revenue', 'N/A')}")
            
            offices = data.get('offices', [])
            if offices:
                print(f"🏢 Офисы: {', '.join(offices)}")
                
        except json.JSONDecodeError as e:
            print(f"❌ Ошибка парсинга: {e}")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")


def main():
    """Главная функция с примерами JSON режима."""
    print("🔧 Примеры использования JSON режима с Amvera LLM")
    print("="*60)
    
    # Запускаем примеры
    basic_json_example()
    structured_json_example()
    data_extraction_example()
    
    print("\n✅ Все примеры JSON режима завершены!")
    print("\n💡 JSON режим полезен для:")
    print("   - Структурированных ответов")
    print("   - Извлечения данных")
    print("   - Интеграции с API")
    print("   - Автоматической обработки ответов")


if __name__ == "__main__":
    main()