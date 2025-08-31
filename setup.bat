@echo off
REM Скрипт быстрой установки AI-торгового бота для Windows

echo 🤖 Установка AI-торгового бота...

REM Проверка Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python не найден. Установите Python 3.9+ с python.org
    pause
    exit /b 1
)

echo ✅ Python найден

REM Создание виртуального окружения
echo 📦 Создание виртуального окружения...
if not exist venv (
    python -m venv venv
    echo ✅ Виртуальное окружение создано
) else (
    echo ℹ️ Виртуальное окружение уже существует
)

REM Активация виртуального окружения
echo 🔄 Активация виртуального окружения...
call venv\Scripts\activate.bat

REM Обновление pip
echo 📋 Обновление pip...
python -m pip install --upgrade pip

REM Установка зависимостей
echo 📚 Установка зависимостей...
pip install -r requirements.txt

REM Создание необходимых директорий  
echo 📁 Создание директорий...
python cli.py setup

REM Создание .env файла
if not exist .env (
    if exist .env.example (
        copy .env.example .env
        echo ✅ Создан файл .env из шаблона
        echo ⚠️ Не забудьте заполнить API ключи в .env
    )
)

echo.
echo 🎉 Установка завершена!
echo.
echo 📋 Следующие шаги:
echo    1. Активируйте окружение: venv\Scripts\activate.bat
echo    2. Заполните API ключи в файле .env
echo    3. Настройте config\config.yaml под ваши нужды
echo    4. Обучите модели: python cli.py train -s SBER -s YNDX -s NVTK -s ROSN -s T -s TATN -s NRU5 -s SVU5
echo    5. Запустите бэктест: python cli.py backtest -s SBER -s YNDX -s NVTK -s ROSN -s T -s TATN -s NRU5 -s SVU5
echo    6. Запустите в демо-режиме: python cli.py start --demo
echo.
echo 📚 Документация: README.md
echo 🆘 Помощь: python cli.py --help

pause
