"""Tests for AmveraLLM."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

from langchain_amvera import AmveraLLM, create_amvera_chat_model


# Тестовые инструменты
class ToolArgs(BaseModel):
    query: str = Field(description="Test query")


@tool("test_tool", args_schema=ToolArgs)
def test_tool(query: str) -> str:
    """Test tool for unit tests."""
    return f"Result for: {query}"


class TestAmveraLLM:
    """Test cases for AmveraLLM."""

    @pytest.fixture
    def mock_response_data(self):
        """Mock response data from Amvera API."""
        return {
            "result": {
                "alternatives": [
                    {
                        "status": "ALTERNATIVE_STATUS_FINAL", 
                        "message": {"text": "Test response"}
                    }
                ],
                "usage": {
                    "inputTextTokens": 10,
                    "completionTokens": 5,
                    "totalTokens": 15
                },
                "modelVersion": "llama70b-v1.0"
            }
        }

    @patch.dict("os.environ", {"AMVERA_API_TOKEN": "test-token"})
    def test_initialization(self):
        """Test AmveraLLM initialization."""
        llm = AmveraLLM()
        assert llm.model == "llama70b"
        assert llm.temperature == 0.7
        assert llm._llm_type == "amvera"

    @patch.dict("os.environ", {"AMVERA_API_TOKEN": "test-token"})
    def test_initialization_with_params(self):
        """Test AmveraLLM initialization with custom parameters."""
        llm = AmveraLLM(
            model="llama8b",
            temperature=0.5,
            max_tokens=100,
            timeout=30,
            verbose=True
        )
        assert llm.model == "llama8b"
        assert llm.temperature == 0.5
        assert llm.max_tokens == 100
        assert llm.timeout == 30
        assert llm.verbose is True

    @patch.dict("os.environ", {"AMVERA_API_TOKEN": "test-token"})
    def test_gpt_model_initialization(self):
        """Test AmveraLLM initialization with GPT models."""
        llm_gpt4 = AmveraLLM(model="gpt-4.1")
        llm_gpt5 = AmveraLLM(model="gpt-5")
        
        assert llm_gpt4.model == "gpt-4.1"
        assert llm_gpt5.model == "gpt-5"

    def test_initialization_without_token(self):
        """Test that initialization fails without API token."""
        with pytest.raises(ValueError, match="API токен Amvera не найден"):
            AmveraLLM()

    @patch.dict("os.environ", {"AMVERA_API_TOKEN": "test-token"})
    def test_convert_messages_to_amvera_format(self):
        """Test message format conversion."""
        llm = AmveraLLM()
        messages = [
            SystemMessage(content="You are helpful"),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there")
        ]
        
        converted = llm._convert_messages_to_amvera_format(messages)
        
        assert len(converted) == 3
        assert converted[0]["role"] == "system"
        assert converted[0]["text"] == "You are helpful"
        assert converted[1]["role"] == "user"
        assert converted[1]["text"] == "Hello"
        assert converted[2]["role"] == "assistant"
        assert converted[2]["text"] == "Hi there"

    @patch.dict("os.environ", {"AMVERA_API_TOKEN": "test-token"})
    def test_create_payload(self):
        """Test payload creation for API requests."""
        llm = AmveraLLM(temperature=0.8, max_tokens=200)
        messages = [HumanMessage(content="Test")]
        
        payload = llm._create_payload(messages, stop=["END"])
        
        assert payload["model"] == "llama70b"
        assert payload["temperature"] == 0.8
        assert payload["max_tokens"] == 200
        assert payload["stop"] == ["END"]
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"

    @patch.dict("os.environ", {"AMVERA_API_TOKEN": "test-token"})
    def test_parse_response(self, mock_response_data):
        """Test response parsing."""
        llm = AmveraLLM()
        
        content, info, tool_calls = llm._parse_response(mock_response_data)
        
        assert content == "Test response"
        assert "usage" in info
        assert "model_version" in info
        assert info["usage"]["totalTokens"] == 15
        assert info["model_version"] == "llama70b-v1.0"
        assert tool_calls is None

    @patch.dict("os.environ", {"AMVERA_API_TOKEN": "test-token"})
    @patch("httpx.Client.post")
    def test_generate_success(self, mock_post, mock_response_data):
        """Test successful generation."""
        mock_response = Mock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        llm = AmveraLLM()
        messages = [HumanMessage(content="Test")]
        
        result = llm._generate(messages)
        
        assert len(result.generations) == 1
        assert result.generations[0].message.content == "Test response"
        mock_post.assert_called_once()

    @patch.dict("os.environ", {"AMVERA_API_TOKEN": "test-token"})
    @patch("httpx.AsyncClient.post")
    async def test_agenerate_success(self, mock_post, mock_response_data):
        """Test successful async generation."""
        mock_response = Mock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        llm = AmveraLLM()
        messages = [HumanMessage(content="Test")]
        
        result = await llm._agenerate(messages)
        
        assert len(result.generations) == 1
        assert result.generations[0].message.content == "Test response"
        mock_post.assert_called_once()
        
        await llm.aclose()

    @patch.dict("os.environ", {"AMVERA_API_TOKEN": "test-token"})
    def test_get_token_usage(self):
        """Test token usage extraction."""
        llm = AmveraLLM()
        message = AIMessage(
            content="test",
            response_metadata={"usage": {"totalTokens": 15}}
        )
        
        usage = llm.get_token_usage(message)
        assert usage["totalTokens"] == 15

    @patch.dict("os.environ", {"AMVERA_API_TOKEN": "test-token"})
    def test_get_model_version(self):
        """Test model version extraction."""
        llm = AmveraLLM()
        message = AIMessage(
            content="test",
            response_metadata={"model_version": "llama70b-v1.0"}
        )
        
        version = llm.get_model_version(message)
        assert version == "llama70b-v1.0"

    @patch.dict("os.environ", {"AMVERA_API_TOKEN": "test-token"})
    def test_json_mode_payload(self):
        """Test JSON mode payload creation."""
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        llm = AmveraLLM(json_mode=True, json_schema=schema)
        messages = [HumanMessage(content="Test")]
        
        payload = llm._create_payload(messages)
        
        assert payload["jsonObject"] is True
        assert "jsonSchema" in payload
        assert payload["jsonSchema"]["schema"] == schema

    @patch.dict("os.environ", {"AMVERA_API_TOKEN": "test-token"})
    def test_tools_payload(self):
        """Test tools in payload creation."""
        tools = [{"type": "function", "function": {"name": "test_func"}}]
        llm = AmveraLLM(bound_tools=tools)
        messages = [HumanMessage(content="Test")]
        
        payload = llm._create_payload(messages)
        
        assert payload["tools"] == tools


class TestCreateAmveraChatModel:
    """Test cases for create_amvera_chat_model factory function."""

    @patch.dict("os.environ", {"AMVERA_API_TOKEN": "test-token"})
    def test_create_with_defaults(self):
        """Test creation with default parameters."""
        llm = create_amvera_chat_model()
        
        assert isinstance(llm, AmveraLLM)
        assert llm.model == "llama70b"
        assert llm.temperature == 0.7

    @patch.dict("os.environ", {"AMVERA_API_TOKEN": "test-token"})
    def test_create_with_custom_params(self):
        """Test creation with custom parameters."""
        llm = create_amvera_chat_model(
            model="llama8b",
            temperature=0.5,
            max_tokens=100,
            verbose=True
        )
        
        assert llm.model == "llama8b"
        assert llm.temperature == 0.5
        assert llm.max_tokens == 100
        assert llm.verbose is True

    def test_create_with_explicit_token(self):
        """Test creation with explicit API token."""
        llm = create_amvera_chat_model(api_token="explicit-token")
        
        assert isinstance(llm, AmveraLLM)
        assert llm.api_token.get_secret_value() == "explicit-token"


class TestAmveraLLMBindTools:
    """Test cases for bind_tools method."""

    @patch.dict("os.environ", {"AMVERA_API_TOKEN": "test-token"})
    def test_bind_tools_basic(self):
        """Test basic bind_tools functionality."""
        llm = AmveraLLM()
        tools = [test_tool]
        
        llm_with_tools = llm.bind_tools(tools)
        
        assert isinstance(llm_with_tools, AmveraLLM)
        assert llm_with_tools.bound_tools is not None
        assert len(llm_with_tools.bound_tools) == 1
        
        tool_def = llm_with_tools.bound_tools[0]
        assert tool_def["type"] == "function"
        assert tool_def["function"]["name"] == "test_tool"
        assert "description" in tool_def["function"]

    @patch.dict("os.environ", {"AMVERA_API_TOKEN": "test-token"})
    def test_bind_tools_with_schema(self):
        """Test bind_tools with parameter schema."""
        llm = AmveraLLM()
        tools = [test_tool]
        
        llm_with_tools = llm.bind_tools(tools)
        
        tool_def = llm_with_tools.bound_tools[0]
        assert "parameters" in tool_def["function"]
        
        parameters = tool_def["function"]["parameters"]
        assert parameters["type"] == "object"
        assert "properties" in parameters
        assert "query" in parameters["properties"]

    @patch.dict("os.environ", {"AMVERA_API_TOKEN": "test-token"})
    def test_bind_tools_multiple(self):
        """Test binding multiple tools."""
        
        @tool("second_tool")
        def second_test_tool(text: str) -> str:
            """Second test tool."""
            return f"Second: {text}"
        
        llm = AmveraLLM()
        tools = [test_tool, second_test_tool]
        
        llm_with_tools = llm.bind_tools(tools)
        
        assert len(llm_with_tools.bound_tools) == 2
        
        tool_names = [tool["function"]["name"] for tool in llm_with_tools.bound_tools]
        assert "test_tool" in tool_names
        assert "second_tool" in tool_names

    @patch.dict("os.environ", {"AMVERA_API_TOKEN": "test-token"})
    def test_bind_tools_returns_new_instance(self):
        """Test that bind_tools returns a new instance."""
        llm = AmveraLLM()
        tools = [test_tool]
        
        llm_with_tools = llm.bind_tools(tools)
        
        # Должны быть разные объекты
        assert llm is not llm_with_tools
        
        # Исходная модель не должна иметь инструментов
        assert llm.bound_tools is None
        
        # Новая модель должна иметь инструменты
        assert llm_with_tools.bound_tools is not None


class TestAmveraLLMGPTModels:
    """Test cases specifically for GPT models."""

    @pytest.fixture
    def mock_gpt_response_data(self):
        """Mock GPT response data from Amvera API."""
        return {
            "result": {
                "alternatives": [
                    {
                        "status": "ALTERNATIVE_STATUS_FINAL", 
                        "message": {"text": "GPT model response"}
                    }
                ],
                "usage": {
                    "inputTextTokens": 12,
                    "completionTokens": 8,
                    "totalTokens": 20
                },
                "modelVersion": "gpt-5-v1.0"
            }
        }

    @patch.dict("os.environ", {"AMVERA_API_TOKEN": "test-token"})
    def test_gpt_parse_response(self, mock_gpt_response_data):
        """Test GPT response parsing uses same format as Llama."""
        llm = AmveraLLM(model="gpt-5")
        
        content, info, tool_calls = llm._parse_response(mock_gpt_response_data)
        
        assert content == "GPT model response"
        assert "usage" in info
        assert "model_version" in info
        assert info["usage"]["totalTokens"] == 20
        assert info["model_version"] == "gpt-5-v1.0"
        assert tool_calls is None

    @patch.dict("os.environ", {"AMVERA_API_TOKEN": "test-token"})
    @patch("httpx.Client.post")
    def test_gpt_endpoint_selection(self, mock_post):
        """Test that GPT models use correct endpoint."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "result": {
                "alternatives": [{"status": "ALTERNATIVE_STATUS_FINAL", "message": {"text": "test"}}]
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        # Test GPT model
        llm_gpt = AmveraLLM(model="gpt-5")
        messages = [HumanMessage(content="Test")]
        
        llm_gpt._generate(messages)
        
        # Verify it called the /models/gpt endpoint
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "/models/gpt"

    @patch.dict("os.environ", {"AMVERA_API_TOKEN": "test-token"})
    @patch("httpx.Client.post")
    def test_llama_endpoint_selection(self, mock_post):
        """Test that Llama models use correct endpoint."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "result": {
                "alternatives": [{"status": "ALTERNATIVE_STATUS_FINAL", "message": {"text": "test"}}]
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        # Test Llama model
        llm_llama = AmveraLLM(model="llama70b")
        messages = [HumanMessage(content="Test")]
        
        llm_llama._generate(messages)
        
        # Verify it called the /models/llama endpoint
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "/models/llama"

    @patch.dict("os.environ", {"AMVERA_API_TOKEN": "test-token"})
    def test_gpt_payload_creation(self):
        """Test payload creation for GPT models."""
        llm = AmveraLLM(model="gpt-4.1", temperature=0.8, max_tokens=150)
        messages = [
            SystemMessage(content="You are helpful"),
            HumanMessage(content="Hello")
        ]
        
        payload = llm._create_payload(messages)
        
        assert payload["model"] == "gpt-4.1"
        assert payload["temperature"] == 0.8
        assert payload["max_tokens"] == 150
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["text"] == "You are helpful"
        assert payload["messages"][1]["role"] == "user"
        assert payload["messages"][1]["text"] == "Hello"