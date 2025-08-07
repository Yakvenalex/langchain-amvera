# Установка langchain-amvera

## Установка из собранного пакета (локально)

1. Перейдите в папку проекта:
   ```bash
   cd /home/yakvenalexx/PycharmProjects/LangchainAmveraLLM
   ```

2. Установите пакет из локального файла wheel:
   ```bash
   pip install dist/langchain_amvera-0.1.0-py3-none-any.whl
   ```

   Или из tar.gz архива:
   ```bash
   pip install dist/langchain_amvera-0.1.0.tar.gz
   ```

3. Проверьте установку:
   ```python
   from langchain_amvera import AmveraLLM
   print("✅ Пакет успешно установлен!")
   ```

## Установка в режиме разработки

Для разработки пакета:

```bash
pip install -e .
```

Или с дополнительными зависимостями для разработки:

```bash
pip install -e ".[dev,test]"
```

## Публикация на PyPI

### На Test PyPI (для тестирования)

1. Установите twine:
   ```bash
   pip install twine
   ```

2. Загрузите на Test PyPI:
   ```bash
   python -m twine upload --repository testpypi dist/*
   ```

3. Установите с Test PyPI:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ langchain-amvera
   ```

### На основной PyPI

1. Загрузите на PyPI:
   ```bash
   python -m twine upload dist/*
   ```

2. После публикации пользователи смогут установить:
   ```bash
   pip install langchain-amvera
   ```

## Структура проекта

```
langchain-amvera/
├── langchain_amvera/          # Основной код библиотеки
│   ├── __init__.py
│   └── amvera.py
├── tests/                     # Тесты
│   ├── __init__.py
│   ├── conftest.py
│   └── test_amvera.py
├── examples/                  # Примеры использования
│   ├── basic_usage.py
│   ├── async_usage.py
│   ├── function_calling.py
│   ├── json_mode.py
│   └── langchain_integration.py
├── dist/                      # Собранные пакеты
│   ├── langchain_amvera-0.1.0-py3-none-any.whl
│   └── langchain_amvera-0.1.0.tar.gz
├── pyproject.toml            # Конфигурация проекта
├── setup.py                  # Скрипт установки
├── requirements.txt          # Зависимости
├── README.md                 # Основная документация
├── LICENSE                   # Лицензия
├── MANIFEST.in              # Дополнительные файлы для пакета
├── Makefile                 # Команды для разработки
└── .gitignore               # Игнорируемые файлы Git
```

## Команды разработки

Используйте Makefile для удобной работы:

- `make help` - показать все доступные команды
- `make install` - установить пакет
- `make install-dev` - установить с dev зависимостями
- `make test` - запустить тесты
- `make lint` - проверить код линтерами
- `make format` - отформатировать код
- `make build` - собрать пакет
- `make clean` - очистить временные файлы

## Требования к окружению

- Python 3.8+
- pip
- Переменная окружения `AMVERA_API_TOKEN` с вашим API токеном