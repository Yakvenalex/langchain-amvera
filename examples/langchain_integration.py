#!/usr/bin/env python3
"""Пример интеграции langchain-amvera с экосистемой LangChain."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_amvera import AmveraLLM


def basic_chain_example():
    """Базовый пример создания цепочки с AmveraLLM."""
    print("🔗 Пример 1: Базовая цепочка LangChain\n")
    
    # Создаем модель
    llm = AmveraLLM(
        model="llama70b",
        temperature=0.7,
        max_tokens=200
    )
    
    # Создаем промпт-шаблон
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "Ты эксперт по {domain}. Отвечай кратко и информативно."
        ),
        HumanMessagePromptTemplate.from_template("{question}")
    ])
    
    # Создаем парсер выходных данных
    output_parser = StrOutputParser()
    
    # Создаем цепочку
    chain = prompt | llm | output_parser
    
    try:
        # Используем цепочку
        result = chain.invoke({
            "domain": "искусственный интеллект",
            "question": "Что такое нейронные сети?"
        })
        
        print(f"✅ Результат цепочки:\n{result}")
        
    except Exception as e:
        print(f"❌ Ошибка в цепочке: {e}")


def conversation_chain_example():
    """Пример цепочки для ведения диалога."""
    print("\n🔗 Пример 2: Цепочка для диалога\n")
    
    llm = AmveraLLM(
        model="llama70b",
        temperature=0.8,
        max_tokens=150
    )
    
    # Шаблон для диалога
    conversation_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
Ты дружелюбный помощник-консультант. Веди естественный диалог.
Запоминай контекст разговора и отвечай соответственно.
        """),
        HumanMessagePromptTemplate.from_template("{user_input}")
    ])
    
    chain = conversation_prompt | llm | StrOutputParser()
    
    # Симулируем диалог
    conversation_steps = [
        "Привет! Как дела?",
        "Можешь посоветовать хорошую книгу по программированию?",
        "А что думаешь о Python для начинающих?",
        "Спасибо за совет! До свидания."
    ]
    
    try:
        for i, user_message in enumerate(conversation_steps, 1):
            print(f"👤 Пользователь: {user_message}")
            
            response = chain.invoke({"user_input": user_message})
            print(f"🤖 Помощник: {response}")
            
            if i < len(conversation_steps):
                print("-" * 40)
                
    except Exception as e:
        print(f"❌ Ошибка в диалоге: {e}")


def multi_step_chain_example():
    """Пример многошаговой цепочки."""
    print("\n🔗 Пример 3: Многошаговая цепочка обработки\n")
    
    llm = AmveraLLM(
        model="llama8b",
        temperature=0.5,
        max_tokens=100
    )
    
    # Шаг 1: Анализ темы
    analysis_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="Проанализируй тему и выдели ключевые аспекты в одном предложении."),
        HumanMessagePromptTemplate.from_template("Тема: {topic}")
    ])
    
    # Шаг 2: Генерация идей
    ideas_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="На основе анализа предложи 3 практические идеи."),
        HumanMessagePromptTemplate.from_template("Анализ: {analysis}")
    ])
    
    # Шаг 3: Финальные рекомендации
    recommendations_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="Дай конкретные рекомендации на основе идей."),
        HumanMessagePromptTemplate.from_template("Идеи: {ideas}")
    ])
    
    # Создаем отдельные цепочки
    analysis_chain = analysis_prompt | llm | StrOutputParser()
    ideas_chain = ideas_prompt | llm | StrOutputParser()
    recommendations_chain = recommendations_prompt | llm | StrOutputParser()
    
    topic = "Улучшение продуктивности в удаленной работе"
    
    try:
        print(f"📋 Исходная тема: {topic}")
        print("\n🔍 Шаг 1: Анализ темы...")
        
        # Шаг 1
        analysis = analysis_chain.invoke({"topic": topic})
        print(f"📊 Анализ: {analysis}")
        
        print("\n💡 Шаг 2: Генерация идей...")
        
        # Шаг 2  
        ideas = ideas_chain.invoke({"analysis": analysis})
        print(f"✨ Идеи: {ideas}")
        
        print("\n🎯 Шаг 3: Формирование рекомендаций...")
        
        # Шаг 3
        recommendations = recommendations_chain.invoke({"ideas": ideas})
        print(f"📝 Рекомендации: {recommendations}")
        
        print(f"\n✅ Многошаговая обработка темы '{topic}' завершена!")
        
    except Exception as e:
        print(f"❌ Ошибка в многошаговой цепочке: {e}")


def custom_output_parser_example():
    """Пример с кастомным парсером выходных данных."""
    print("\n🔗 Пример 4: Кастомный парсер выходных данных\n")
    
    from langchain_core.output_parsers import BaseOutputParser
    from typing import List
    import re
    
    class ListOutputParser(BaseOutputParser[List[str]]):
        """Парсер для извлечения списков из ответа."""
        
        def parse(self, text: str) -> List[str]:
            # Ищем пронумерованные или маркированные списки
            patterns = [
                r'^\d+\.\s*(.+)$',  # 1. item
                r'^-\s*(.+)$',      # - item  
                r'^\*\s*(.+)$',     # * item
                r'^\•\s*(.+)$',     # • item
            ]
            
            items = []
            lines = text.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                for pattern in patterns:
                    match = re.match(pattern, line, re.MULTILINE)
                    if match:
                        items.append(match.group(1).strip())
                        break
            
            # Если не нашли структурированный список, разбиваем по предложениям
            if not items:
                sentences = [s.strip() for s in text.split('.') if s.strip()]
                items = sentences[:5]  # Берем первые 5 предложений
            
            return items
    
    llm = AmveraLLM(
        model="llama70b", 
        temperature=0.6,
        max_tokens=200
    )
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="Создай список рекомендаций. Используй нумерацию или маркеры."),
        HumanMessagePromptTemplate.from_template("{request}")
    ])
    
    list_parser = ListOutputParser()
    
    chain = prompt | llm | list_parser
    
    try:
        result = chain.invoke({
            "request": "Дай 5 советов по изучению нового языка программирования"
        })
        
        print("📋 Извлеченный список советов:")
        for i, item in enumerate(result, 1):
            print(f"  {i}. {item}")
            
    except Exception as e:
        print(f"❌ Ошибка с кастомным парсером: {e}")


def main():
    """Главная функция с примерами интеграции LangChain."""
    print("🔗 Примеры интеграции Amvera LLM с экосистемой LangChain")
    print("="*70)
    
    # Запускаем примеры
    basic_chain_example()
    conversation_chain_example() 
    multi_step_chain_example()
    custom_output_parser_example()
    
    print(f"\n✅ Все примеры интеграции LangChain завершены!")
    print(f"\n💡 AmveraLLM полностью совместим с:")
    print("   - Промпт-шаблонами LangChain")
    print("   - Цепочками обработки")
    print("   - Парсерами выходных данных")
    print("   - Системой callbacks")


if __name__ == "__main__":
    main()