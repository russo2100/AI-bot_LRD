#!/bin/bash
# Скрипт быстрой установки AI-торгового бота

set -e

echo "🤖 Установка AI-торгового бота..."

# Проверка Python версии
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo "✅ Python $python_version найден"
else
    echo "❌ Требуется Python $required_version или выше (найден $python_version)"
    exit 1
fi

# Создание виртуального окружения
echo "📦 Создание виртуального окружения..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Виртуальное окружение создано"
else
    echo "ℹ️ Виртуальное окружение уже существует"
fi

# Активация виртуального окружения
echo "🔄 Активация виртуального окружения..."
source venv/bin/activate

# Обновление pip
echo "📋 Обновление pip..."
pip install --upgrade pip

# Установка зависимостей
echo "📚 Установка зависимостей..."
pip install -r requirements.txt

# Создание необходимых директорий
echo "📁 Создание директорий..."
python cli.py setup

# Проверка установки
echo "🧪 Проверка установки..."
python -c "import pandas, numpy, sklearn, yfinance; print('✅ Основные зависимости установлены')"

# Создание .env файла из примера
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo "✅ Создан файл .env из шаблона"
    echo "⚠️ Не забудьте заполнить API ключи в .env"
fi

echo ""
echo "🎉 Установка завершена!"
echo ""
echo "📋 Следующие шаги:"
echo "   1. Активируйте окружение: source venv/bin/activate"
echo "   2. Заполните API ключи в файле .env"  
echo "   3. Настройте config/config.yaml под ваши нужды"
echo "   4. Обучите модели: python cli.py train -s AAPL MSFT"
echo "   5. Запустите бэктест: python cli.py backtest -s AAPL"
echo "   6. Запустите в демо-режиме: python cli.py start --demo"
echo ""
echo "📚 Документация: README.md"
echo "🆘 Помощь: python cli.py --help"
