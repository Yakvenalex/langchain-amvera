"""Pytest configuration and fixtures."""

import pytest
import os
from unittest.mock import patch


@pytest.fixture(autouse=True)
def clean_env():
    """Clean environment variables before each test."""
    # Сохраняем текущие значения
    original_token = os.environ.get("AMVERA_API_TOKEN")
    
    # Очищаем переменную
    if "AMVERA_API_TOKEN" in os.environ:
        del os.environ["AMVERA_API_TOKEN"]
    
    yield
    
    # Восстанавливаем значения после теста
    if original_token is not None:
        os.environ["AMVERA_API_TOKEN"] = original_token
    elif "AMVERA_API_TOKEN" in os.environ:
        del os.environ["AMVERA_API_TOKEN"]


@pytest.fixture
def mock_amvera_response():
    """Mock response from Amvera API."""
    return {
        "result": {
            "alternatives": [
                {
                    "status": "ALTERNATIVE_STATUS_FINAL",
                    "message": {"text": "Mock response from Amvera"}
                }
            ],
            "usage": {
                "inputTextTokens": 5,
                "completionTokens": 8,
                "totalTokens": 13
            },
            "modelVersion": "llama70b-test"
        }
    }