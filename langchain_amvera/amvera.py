"""Amvera LLM integration for LangChain."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence
from pydantic import Field, SecretStr, model_validator, ConfigDict

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    ChatMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from langchain_core.tools import BaseTool

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class AmveraLLM(BaseChatModel):
    """
    Amvera LLM для работы с моделями llama8b и llama70b.

    Example:
        .. code-block:: python

            from langchain_amvera import AmveraLLM

            llm = AmveraLLM(
                model="llama70b",
                temperature=0.7,
                max_tokens=1000,
                api_token="your-token"  # или через переменную AMVERA_API_TOKEN
            )

            # Синхронное использование
            response = llm.invoke("Расскажи анекдот")
            print(response.content)

            # Асинхронное использование
            response = await llm.ainvoke("Расскажи анекдот")
            print(response.content)
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )

    # Основные параметры
    api_token: SecretStr = Field(alias="amvera_api_token")
    """API токен для Amvera."""

    model: str = Field(default="llama70b")
    """ID модели для использования. Доступно: llama8b, llama70b"""

    base_url: str = Field(default="https://kong-proxy.yc.amvera.ru/api/v1")
    """Базовый URL для Amvera API"""

    # Параметры генерации
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    """Параметр творчества модели от 0 до 2"""

    max_tokens: Optional[int] = Field(default=None, ge=1)
    """Максимальное количество токенов для генерации"""

    timeout: int = Field(default=60, ge=1)
    """Таймаут запроса в секундах"""

    # Дополнительные параметры
    json_mode: bool = Field(default=False)
    """Включить JSON режим ответа"""

    json_schema: Optional[Dict[str, Any]] = Field(default=None)
    """JSON схема для валидации ответа в JSON режиме"""


    verbose: bool = Field(default=False)
    """Включить детальное логирование"""

    # Приватное поле для хранения инструментов (используется только с bind_tools)
    bound_tools: Optional[List[Dict[str, Any]]] = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, data: Any) -> Any:
        """Валидация переменных окружения."""
        if isinstance(data, dict):
            api_token = get_from_dict_or_env(
                data, "api_token", "AMVERA_API_TOKEN", default=""
            )
            if api_token:
                data["api_token"] = convert_to_secret_str(api_token)
            else:
                raise ValueError(
                    "API токен Amvera не найден. "
                    "Установите переменную окружения AMVERA_API_TOKEN "
                    "или передайте api_token при инициализации."
                )
        return data

    def __init__(self, **kwargs: Any):
        """Инициализация клиента Amvera."""
        super().__init__(**kwargs)
        # Создаем клиенты после инициализации родительского класса
        self._setup_clients()

    def _setup_clients(self) -> None:
        """Настройка HTTP клиентов."""
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "X-Auth-Token": f"Bearer {self.api_token.get_secret_value()}",
        }

        # Синхронный клиент
        self._sync_client = httpx.Client(
            headers=headers, timeout=self.timeout, base_url=self.base_url
        )

        # Асинхронный клиент создается по требованию
        self._async_client = None

    def _get_async_client(self) -> httpx.AsyncClient:
        """Получение или создание асинхронного клиента."""
        if self._async_client is None:
            headers = {
                "accept": "application/json",
                "Content-Type": "application/json",
                "X-Auth-Token": f"Bearer {self.api_token.get_secret_value()}",
            }

            self._async_client = httpx.AsyncClient(
                headers=headers, timeout=self.timeout, base_url=self.base_url
            )
        return self._async_client

    @property
    def _llm_type(self) -> str:
        """Возвращает тип языковой модели."""
        return "amvera"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Параметры для идентификации модели."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "base_url": self.base_url,
        }

    def _convert_messages_to_amvera_format(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, Any]]:
        """Конвертация сообщений LangChain в формат Amvera API."""
        amvera_messages = []

        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, ToolMessage):
                role = "user"  # Представляем результат инструмента как пользовательское сообщение
            elif isinstance(message, ChatMessage):
                role = message.role
            else:
                logger.warning(
                    f"Неизвестный тип сообщения {type(message)}, используем 'user'"
                )
                role = "user"

            # Обрабатываем content как строку
            content = message.content
            if content is None:
                content = ""
            elif not isinstance(content, str):
                content = str(content)

            # Особая обработка для ToolMessage
            if isinstance(message, ToolMessage):
                # Форматируем результат инструмента для лучшего понимания
                tool_name = getattr(message, 'tool_call_id', 'unknown_tool')
                content = f"Результат выполнения инструмента: {content}"

            amvera_message: Dict[str, Any] = {
                "role": role,
                "text": content,
            }

            # Обработка tool calls для AI сообщений
            if isinstance(message, AIMessage) and hasattr(message, "tool_calls"):
                if message.tool_calls:
                    amvera_message["toolCallList"] = {
                        "toolCalls": [
                            {
                                "functionCall": {
                                    "name": tool_call.get("name", ""),
                                    "arguments": tool_call.get("args", {}),
                                }
                            }
                            for tool_call in message.tool_calls
                        ]
                    }

            amvera_messages.append(amvera_message)

        return amvera_messages

    def _create_payload(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Создание payload для запроса к API."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": self._convert_messages_to_amvera_format(messages),
        }

        # Параметры генерации
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if stop:
            payload["stop"] = stop

        # Дополнительные возможности
        if self.bound_tools:
            payload["tools"] = self.bound_tools
        if self.json_mode:
            payload["jsonObject"] = True
            if self.json_schema:
                payload["jsonSchema"] = {"schema": self.json_schema}

        # Параметры из kwargs
        payload.update(kwargs)
        return payload

    def _parse_response(
        self, response_data: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any], Optional[List[Dict[str, Any]]]]:
        """Парсинг ответа от Amvera API."""
        content = ""
        generation_info = {}
        tool_calls = None

        # Основной формат ответа Amvera
        if "result" in response_data and "alternatives" in response_data["result"]:
            alternatives = response_data["result"]["alternatives"]

            if alternatives:
                # Ищем финальную альтернативу
                for alt in alternatives:
                    if alt.get("status") == "ALTERNATIVE_STATUS_FINAL":
                        message = alt.get("message", {})
                        content = message.get("text", "")
                        
                        # Извлекаем tool calls если есть
                        if "toolCallList" in message and "toolCalls" in message["toolCallList"]:
                            tool_calls = []
                            for tool_call in message["toolCallList"]["toolCalls"]:
                                if "functionCall" in tool_call:
                                    func_call = tool_call["functionCall"]
                                    tool_calls.append({
                                        "name": func_call.get("name", ""),
                                        "args": func_call.get("arguments", {}),
                                        "id": tool_call.get("id", f"call_{len(tool_calls)}")
                                    })
                        break

                # Если не нашли финальную, берем первую
                if not content and alternatives:
                    message = alternatives[0].get("message", {})
                    content = message.get("text", "")
                    
                    # Извлекаем tool calls если есть
                    if "toolCallList" in message and "toolCalls" in message["toolCallList"]:
                        tool_calls = []
                        for tool_call in message["toolCallList"]["toolCalls"]:
                            if "functionCall" in tool_call:
                                func_call = tool_call["functionCall"]
                                tool_calls.append({
                                    "name": func_call.get("name", ""),
                                    "args": func_call.get("arguments", {}),
                                    "id": tool_call.get("id", f"call_{len(tool_calls)}")
                                })

            # Информация об использовании
            if "usage" in response_data["result"]:
                generation_info["usage"] = response_data["result"]["usage"]
            if "modelVersion" in response_data["result"]:
                generation_info["model_version"] = response_data["result"][
                    "modelVersion"
                ]

        # Fallback форматы
        elif "choices" in response_data and response_data["choices"]:
            choice = response_data["choices"][0]
            message = choice.get("message", {})
            content = message.get("content", "")
            
            # Проверяем tool_calls в OpenAI формате
            if "tool_calls" in message:
                tool_calls = []
                for tool_call in message["tool_calls"]:
                    if tool_call.get("type") == "function" and "function" in tool_call:
                        func_call = tool_call["function"]
                        args = func_call.get("arguments", "{}")
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except:
                                args = {}
                        tool_calls.append({
                            "name": func_call.get("name", ""),
                            "args": args,
                            "id": tool_call.get("id", f"call_{len(tool_calls)}")
                        })
                        
        elif "response" in response_data:
            content = response_data["response"]
        else:
            content = str(response_data)

        return content, generation_info, tool_calls

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Синхронная генерация ответа."""
        payload = self._create_payload(messages, stop, **kwargs)

        if self.verbose:
            logger.info(
                f"Отправка запроса к Amvera API: {json.dumps(payload, ensure_ascii=False, indent=2)}"
            )

        try:
            response = self._sync_client.post("/models/llama", json=payload)
            response.raise_for_status()
            response_data = response.json()

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ValueError("Неверный API токен Amvera")
            elif e.response.status_code == 429:
                raise ValueError("Превышен лимит запросов к Amvera API")
            else:
                raise ValueError(f"HTTP ошибка {e.response.status_code}: {e}")
        except httpx.TimeoutException:
            raise ValueError("Таймаут запроса к Amvera API")
        except httpx.RequestError as e:
            raise ValueError(f"Ошибка соединения с Amvera API: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Ошибка парсинга JSON ответа: {e}")

        if self.verbose:
            logger.info(
                f"Получен ответ от Amvera API: {json.dumps(response_data, ensure_ascii=False, indent=2)}"
            )

        content, generation_info, tool_calls = self._parse_response(response_data)

        message = AIMessage(
            content=content,
            response_metadata=generation_info,
            tool_calls=tool_calls or [],
        )

        generation = ChatGeneration(
            message=message,
            generation_info=generation_info,
        )

        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Асинхронная генерация ответа."""
        payload = self._create_payload(messages, stop, **kwargs)

        if self.verbose:
            logger.info(
                f"[ASYNC] Отправка запроса к Amvera API: {json.dumps(payload, ensure_ascii=False, indent=2)}"
            )

        client = self._get_async_client()

        try:
            response = await client.post("/models/llama", json=payload)
            response.raise_for_status()
            response_data = response.json()

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ValueError("Неверный API токен Amvera")
            elif e.response.status_code == 429:
                raise ValueError("Превышен лимит запросов к Amvera API")
            else:
                raise ValueError(f"HTTP ошибка {e.response.status_code}: {e}")
        except httpx.TimeoutException:
            raise ValueError("Таймаут запроса к Amvera API")
        except httpx.RequestError as e:
            raise ValueError(f"Ошибка соединения с Amvera API: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Ошибка парсинга JSON ответа: {e}")

        if self.verbose:
            logger.info(
                f"[ASYNC] Получен ответ от Amvera API: {json.dumps(response_data, ensure_ascii=False, indent=2)}"
            )

        content, generation_info, tool_calls = self._parse_response(response_data)

        message = AIMessage(
            content=content,
            response_metadata=generation_info,
            tool_calls=tool_calls or [],
        )

        generation = ChatGeneration(
            message=message,
            generation_info=generation_info,
        )

        return ChatResult(generations=[generation])

    def get_token_usage(self, result: AIMessage) -> Optional[Dict[str, Any]]:
        """Получить информацию об использовании токенов из AIMessage."""
        if hasattr(result, "response_metadata") and result.response_metadata:
            return result.response_metadata.get("usage")
        return None

    def get_model_version(self, result: AIMessage) -> Optional[str]:
        """Получить версию модели из AIMessage."""
        if hasattr(result, "response_metadata") and result.response_metadata:
            return result.response_metadata.get("model_version")
        return None

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """Секретные поля для сериализации."""
        return {"api_token": "AMVERA_API_TOKEN"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Пространство имен для LangChain."""
        return ["langchain", "chat_models", "amvera"]

    def bind_tools(self, tools: Sequence[BaseTool], **kwargs: Any) -> "AmveraLLM":
        """
        Привязывает инструменты к модели для Function Calling.

        Args:
            tools: Список инструментов LangChain для привязки
            **kwargs: Дополнительные параметры

        Returns:
            Новый экземпляр AmveraLLM с привязанными инструментами
        """
        # Конвертируем LangChain tools в формат Amvera API
        amvera_tools = []
        for tool in tools:
            tool_def = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                }
            }
            
            # Добавляем параметры если они есть
            if hasattr(tool, 'args_schema') and tool.args_schema:
                try:
                    # Получаем JSON схему из pydantic модели
                    schema = tool.args_schema.model_json_schema()
                    tool_def["function"]["parameters"] = schema
                except Exception:
                    # Fallback: базовая схема
                    tool_def["function"]["parameters"] = {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
            
            amvera_tools.append(tool_def)
        
        # Создаем копию с новыми инструментами
        new_kwargs = self.dict()
        new_kwargs.update(kwargs)
        new_kwargs["bound_tools"] = amvera_tools
        
        return self.__class__(**new_kwargs)

    def __del__(self):
        """Закрытие клиентов при удалении объекта."""
        try:
            if hasattr(self, "_sync_client") and self._sync_client:
                self._sync_client.close()
        except Exception:
            pass  # Игнорируем ошибки при закрытии

    async def aclose(self) -> None:
        """Явное закрытие асинхронного клиента."""
        if hasattr(self, "_async_client") and self._async_client:
            await self._async_client.aclose()
            self._async_client = None


def create_amvera_chat_model(
    model: str = "llama70b",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    api_token: Optional[str] = None,
    **kwargs: Any,
) -> AmveraLLM:
    """
    Фабричная функция для создания Amvera chat model.

    Args:
        model: Модель для использования (llama8b или llama70b)
        temperature: Параметр творчества (0.0-2.0)
        max_tokens: Максимальное количество токенов
        api_token: API токен
        **kwargs: Дополнительные параметры

    Returns:
        Настроенный экземпляр AmveraLLM
    """
    params = {"model": model, "temperature": temperature, **kwargs}

    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if api_token is not None:
        params["api_token"] = api_token

    return AmveraLLM(**params)


# Алиасы для совместимости
AmveraChatModel = AmveraLLM
ChatAmvera = AmveraLLM


# Пример использования
if __name__ == "__main__":
    import asyncio
    from langchain_core.messages import HumanMessage, SystemMessage

    async def main():
        try:
            llm = create_amvera_chat_model(
                model="llama70b", temperature=0.7, verbose=True
            )

            print(f"Создан {llm._llm_type} с моделью {llm.model}")

            messages = [
                SystemMessage(content="Ты полезный помощник."),
                HumanMessage(content="Расскажи краткий анекдот про программистов"),
            ]

            print("\n🤖 Отправка запроса...")
            result = await llm.ainvoke(messages)

            print(f"\n✅ Ответ модели:\n{result.content}")

            # Информация о токенах
            usage = llm.get_token_usage(result)
            if usage:
                print(f"\n📊 Использование токенов: {usage}")

            version = llm.get_model_version(result)
            if version:
                print(f"📦 Версия модели: {version}")

            # Закрытие клиентов
            await llm.aclose()

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(main())
