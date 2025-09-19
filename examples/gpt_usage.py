"""
Пример использования GPT моделей через Amvera API.
"""

import asyncio
from langchain_amvera import AmveraLLM
from langchain_core.messages import HumanMessage, SystemMessage


async def main():
    """Демонстрация работы с GPT моделями."""
    
    print("🤖 Тестирование GPT моделей Amvera API\n")
    
    # Список GPT моделей для тестирования
    gpt_models = ["gpt-4.1", "gpt-5"]
    
    for model_name in gpt_models:
        print(f"📋 Тестирование модели: {model_name}")
        print("-" * 50)
        
        try:
            # Создание экземпляра модели
            llm = AmveraLLM(
                model=model_name,
                temperature=0.7,
                max_tokens=200,
                verbose=True
            )
            
            # Сообщения для тестирования
            messages = [
                SystemMessage(content="Ты полезный AI-помощник."),
                HumanMessage(content="Расскажи краткий факт о космосе")
            ]
            
            print(f"\n🚀 Отправка запроса к {model_name}...")
            
            # Асинхронный вызов
            result = await llm.ainvoke(messages)
            
            print(f"\n✅ Ответ от {model_name}:")
            print(f"{result.content}\n")
            
            # Информация об использовании токенов
            usage = llm.get_token_usage(result)
            if usage:
                print(f"📊 Использование токенов: {usage}")
            
            # Версия модели
            version = llm.get_model_version(result)
            if version:
                print(f"📦 Версия модели: {version}")
                
            # Закрытие соединений
            await llm.aclose()
            
        except Exception as e:
            print(f"❌ Ошибка при работе с {model_name}: {e}")
        
        print("\n" + "="*60 + "\n")


def sync_example():
    """Синхронный пример использования GPT модели."""
    print("🔄 Синхронный пример с GPT-4.1")
    
    try:
        llm = AmveraLLM(
            model="gpt-4.1",
            temperature=0.5,
            max_tokens=150
        )
        
        messages = [
            HumanMessage(content="Что такое искусственный интеллект? Ответь кратко.")
        ]
        
        result = llm.invoke(messages)
        print(f"Ответ: {result.content}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    print("🌟 Запуск примеров использования GPT моделей\n")
    
    # Синхронный пример
    sync_example()
    print("\n" + "-"*60 + "\n")
    
    # Асинхронные примеры
    asyncio.run(main())
    
    print("✨ Тестирование завершено!")