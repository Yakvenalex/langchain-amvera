#!/usr/bin/env python3
"""Базовый пример использования langchain-amvera."""

import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_amvera import AmveraLLM


def main():
    """Демонстрация базового использования AmveraLLM."""
    
    # Убедитесь, что установлена переменная окружения AMVERA_API_TOKEN
    # или передайте токен явно при создании модели
    
    print("🚀 Создание модели Amvera LLM...")
    
    # Способ 1: Использование переменной окружения AMVERA_API_TOKEN
    llm = AmveraLLM(
        model="llama70b",
        temperature=0.7,
        max_tokens=200,
        verbose=True  # Включаем детальное логирование
    )
    
    # Способ 2: Передача токена явно (раскомментируйте при необходимости)
    # llm = AmveraLLM(
    #     model="llama70b",
    #     temperature=0.7,
    #     max_tokens=200,
    #     api_token="your-amvera-api-token-here"
    # )
    
    print(f"✅ Модель создана: {llm.model} с температурой {llm.temperature}")
    
    # Создаем сообщения
    messages = [
        SystemMessage(content="Ты полезный ИИ-помощник, который отвечает кратко и по делу."),
        HumanMessage(content="Расскажи краткий анекдот про программистов")
    ]
    
    print("\n📤 Отправка запроса к Amvera API...")
    
    try:
        # Синхронный вызов
        result = llm.invoke(messages)
        
        print(f"\n🤖 Ответ модели:\n{result.content}")
        
        # Получаем информацию об использовании токенов
        usage = llm.get_token_usage(result)
        if usage:
            print(f"\n📊 Использование токенов:")
            print(f"  - Входные токены: {usage.get('inputTextTokens', 'N/A')}")
            print(f"  - Токены ответа: {usage.get('completionTokens', 'N/A')}")
            print(f"  - Всего токенов: {usage.get('totalTokens', 'N/A')}")
        
        # Получаем версию модели
        model_version = llm.get_model_version(result)
        if model_version:
            print(f"📦 Версия модели: {model_version}")
            
    except ValueError as e:
        print(f"❌ Ошибка при вызове API: {e}")
        print("\nПроверьте:")
        print("1. Установлена ли переменная окружения AMVERA_API_TOKEN")
        print("2. Правильность API токена")
        print("3. Доступность интернет-соединения")
        
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")


if __name__ == "__main__":
    main()