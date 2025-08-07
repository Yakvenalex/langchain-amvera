#!/usr/bin/env python3
"""Пример асинхронного использования langchain-amvera."""

import asyncio
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_amvera import AmveraLLM


async def single_request_example():
    """Пример одиночного асинхронного запроса."""
    print("📝 Пример 1: Одиночный асинхронный запрос")
    
    llm = AmveraLLM(
        model="llama70b",
        temperature=0.8,
        max_tokens=150
    )
    
    messages = [
        SystemMessage(content="Ты креативный писатель."),
        HumanMessage(content="Напиши короткую историю про робота-садовника")
    ]
    
    try:
        print("🚀 Отправка асинхронного запроса...")
        result = await llm.ainvoke(messages)
        
        print(f"✅ Ответ получен:\n{result.content}")
        
        usage = llm.get_token_usage(result)
        if usage:
            print(f"📊 Токенов использовано: {usage.get('totalTokens', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    finally:
        # Важно закрыть асинхронный клиент
        await llm.aclose()


async def multiple_requests_example():
    """Пример параллельных асинхронных запросов."""
    print("\n📝 Пример 2: Параллельные асинхронные запросы")
    
    llm = AmveraLLM(
        model="llama8b",  # Используем более быструю модель
        temperature=0.5,
        max_tokens=100
    )
    
    # Подготавливаем несколько запросов
    questions = [
        "Что такое машинное обучение?",
        "Объясни принцип работы нейронных сетей",
        "Расскажи о различиях между ИИ и ML"
    ]
    
    # Создаем задачи для параллельного выполнения
    tasks = []
    for i, question in enumerate(questions):
        messages = [
            SystemMessage(content="Ты эксперт по искусственному интеллекту. Отвечай кратко и понятно."),
            HumanMessage(content=question)
        ]
        task = llm.ainvoke(messages)
        tasks.append((i + 1, question, task))
    
    try:
        print("🚀 Отправка параллельных запросов...")
        
        # Выполняем все запросы параллельно
        results = await asyncio.gather(*[task for _, _, task in tasks])
        
        # Выводим результаты
        for (num, question, _), result in zip(tasks, results):
            print(f"\n❓ Вопрос {num}: {question}")
            print(f"🤖 Ответ: {result.content}")
            
            usage = llm.get_token_usage(result)
            if usage:
                print(f"📊 Токенов: {usage.get('totalTokens', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Ошибка в параллельных запросах: {e}")
    
    finally:
        await llm.aclose()


async def streaming_like_example():
    """Пример имитации стриминга с помощью нескольких быстрых запросов."""
    print("\n📝 Пример 3: Имитация стриминга")
    
    llm = AmveraLLM(
        model="llama8b",
        temperature=0.3,
        max_tokens=50  # Короткие ответы для имитации частей
    )
    
    # Разбиваем большой запрос на части
    story_parts = [
        "Начни историю про космонавта",
        "Продолжи эту историю - что происходит дальше?",
        "Заверши историю интересной концовкой"
    ]
    
    try:
        print("🚀 Создание истории по частям...")
        story = ""
        
        for i, part in enumerate(story_parts, 1):
            messages = [
                SystemMessage(content="Ты рассказчик. Пиши связные части истории."),
                HumanMessage(content=f"{part}. Предыдущая часть: {story}")
            ]
            
            print(f"📝 Генерация части {i}/3...")
            result = await llm.ainvoke(messages)
            
            new_part = result.content
            story += f" {new_part}"
            
            print(f"✨ Часть {i}: {new_part}")
            
            # Небольшая пауза для реалистичности
            await asyncio.sleep(0.5)
        
        print(f"\n📖 Полная история:\n{story.strip()}")
        
    except Exception as e:
        print(f"❌ Ошибка при создании истории: {e}")
    
    finally:
        await llm.aclose()


async def main():
    """Главная функция с примерами."""
    print("🔄 Примеры асинхронного использования Amvera LLM\n")
    
    # Запускаем примеры последовательно
    await single_request_example()
    await multiple_requests_example() 
    await streaming_like_example()
    
    print("\n✅ Все примеры завершены!")


if __name__ == "__main__":
    asyncio.run(main())