#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI интерфейс для управления AI-торговым ботом
"""

import click
import json
import os
import sys
from datetime import datetime, timedelta

# Добавляем корневую папку в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import AITradingBot

@click.group()
@click.version_option(version='1.0.0', prog_name='AI Trading Bot')
def cli():
    """AI-торговый бот с машинным обучением"""
    pass

@cli.command()
@click.option('--config', '-c', default='config/config.yaml', help='Путь к конфигурации')
@click.option('--symbols', '-s', multiple=True, default=['SBER', 'GAZP'], help='Торговые инструменты')
@click.option('--demo', is_flag=True, help='Демо режим (без реальной торговли)')
def start(config, symbols, demo):
    """Запустить торгового бота"""
    click.echo("[BOT] Запуск AI-торгового бота...")

    if demo:
        click.echo("[WARNING] Демо режим: реальная торговля отключена")

    try:
        bot = AITradingBot(config)
        bot.start_trading(list(symbols))
    except KeyboardInterrupt:
        click.echo("\n[STOP] Бот остановлен пользователем")
    except Exception as e:
        click.echo(f"❌ Ошибка: {e}")
        sys.exit(1)

@cli.command()
@click.option('--config', '-c', default='config/config.yaml', help='Путь к конфигурации')
@click.option('--symbols', '-s', multiple=True, default=['SBER', 'GAZP'], help='Торговые инструменты')
@click.option('--start-date', default='2023-01-01', help='Дата начала')
@click.option('--end-date', default='2023-12-31', help='Дата окончания')
def backtest(config, symbols, start_date, end_date):
    """Запустить бэктестирование стратегий"""
    click.echo(f"🔬 Бэктест за период {start_date} - {end_date}")
    click.echo(f"📈 Инструменты: {', '.join(symbols)}")

    try:
        bot = AITradingBot(config)
        bot.run_backtest(list(symbols), start_date, end_date)
        click.echo("✅ Бэктест завершен. Результаты сохранены в backtest_results/")
    except Exception as e:
        click.echo(f"❌ Ошибка бэктеста: {e}")
        sys.exit(1)

@cli.command()
@click.option('--config', '-c', default='config/config.yaml', help='Путь к конфигурации')
@click.option('--symbols', '-s', multiple=True, default=['SBER', 'GAZP'], help='Торговые инструменты')
@click.option('--force', is_flag=True, help='Принудительное переобучение')
def train(config, symbols, force):
    """Обучить или переобучить модели ИИ"""
    click.echo("[AI] Обучение моделей машинного обучения...")
    click.echo(f"📊 Инструменты: {', '.join(symbols)}")

    if force:
        click.echo("⚠️ Принудительное переобучение всех моделей")

    try:
        bot = AITradingBot(config)
        bot.load_or_train_models(list(symbols), retrain=force)
        click.echo("✅ Обучение завершено")
    except Exception as e:
        click.echo(f"❌ Ошибка обучения: {e}")
        sys.exit(1)

@cli.command()
@click.option('--config', '-c', default='config/config.yaml', help='Путь к конфигурации')
def status(config):
    """Показать статус бота"""
    try:
        bot = AITradingBot(config)
        status_info = bot.get_status()

        click.echo("📊 Статус AI-торгового бота:")
        click.echo(json.dumps(status_info, indent=2, ensure_ascii=False))

    except Exception as e:
        click.echo(f"❌ Ошибка получения статуса: {e}")
        sys.exit(1)

@cli.command()
@click.option('--config', '-c', default='config/config.yaml', help='Путь к конфигурации')
@click.option('--symbols', '-s', multiple=True, default=['SBER'], help='Торговые инструменты')
def analyze(config, symbols):
    """Провести разовый анализ рынка"""
    click.echo(f"🔍 Анализ рынка для: {', '.join(symbols)}")

    try:
        bot = AITradingBot(config)

        # Загружаем модели если есть
        if os.path.exists('models'):
            bot.load_or_train_models(list(symbols))

        # Анализируем
        results = bot.analyze_market(list(symbols))

        for symbol, analysis in results.items():
            click.echo(f"\n📈 {symbol}:")
            click.echo(f"   Цена: {analysis['current_price']:.2f}")
            click.echo(f"   Сигнал: {analysis['signal']}")
            click.echo(f"   RSI: {analysis['technical_indicators']['rsi']:.1f}")

            if analysis['predictions']:
                click.echo("   Прогнозы ИИ:")
                for model, pred in analysis['predictions'].items():
                    if isinstance(pred, dict):
                        click.echo(f"     {model}: {pred.get('prediction_label', 'N/A')}")
                    else:
                        click.echo(f"     {model}: {pred}")

    except Exception as e:
        click.echo(f"❌ Ошибка анализа: {e}")
        sys.exit(1)

@cli.command()
def setup():
    """Первоначальная настройка проекта"""
    click.echo("🛠️ Настройка AI-торгового бота...")

    # Создаем необходимые папки
    folders = ['config', 'logs', 'data', 'models', 'backtest_results']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        click.echo(f"📁 Создана папка: {folder}")

    # Проверяем наличие .env файла
    if not os.path.exists('.env'):
        click.echo("📝 Создайте файл .env с вашими API ключами")
        click.echo("   Используйте .env.example как шаблон")

    click.echo("✅ Настройка завершена!")
    click.echo("\n📋 Следующие шаги:")
    click.echo("   1. Заполните .env файл своими API ключами")
    click.echo("   2. Настройте config/config.yaml под ваши нужды")
    click.echo("   3. Запустите: python cli.py train для обучения моделей")
    click.echo("   4. Запустите: python cli.py backtest для тестирования")
    click.echo("   5. Запустите: python cli.py start --demo для демо-режима")

if __name__ == '__main__':
    cli()
