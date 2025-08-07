.PHONY: help install install-dev test lint format clean build upload

help:  ## Показать справку
	@echo "Доступные команды:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Установить пакет
	pip install -e .

install-dev:  ## Установить пакет в режиме разработки со всеми зависимостями
	pip install -e ".[dev,test]"

test:  ## Запустить тесты
	pytest tests/ -v

test-cov:  ## Запустить тесты с покрытием
	pytest tests/ -v --cov=langchain_amvera --cov-report=html --cov-report=term

lint:  ## Проверить код линтерами
	ruff langchain_amvera/ tests/ examples/
	mypy langchain_amvera/

format:  ## Отформатировать код
	black langchain_amvera/ tests/ examples/
	ruff --fix langchain_amvera/ tests/ examples/

clean:  ## Очистить временные файлы
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Собрать пакет
	python -m build

upload-test:  ## Загрузить на TestPyPI
	python -m twine upload --repository testpypi dist/*

upload:  ## Загрузить на PyPI
	python -m twine upload dist/*

check:  ## Проверить пакет перед загрузкой
	python -m twine check dist/*