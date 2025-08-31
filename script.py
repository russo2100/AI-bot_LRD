#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
УНИВЕРСАЛЬНЫЙ УТИЛИТАРНЫЙ СКРИПТ ДЛЯ AI-Торгового Бота T-Bank
Единая точка входа для всех задач: настройка, данные, тестирование, поддержка

Использование:
    python script.py <command> [options]

Команды:
    setup        - Первоначальная настройка проекта
    data         - Работа с данными (загрузка, очистка, валидация)
    validate     - Валидация данных и конфигурации
    test         - Запуск тестов
    optimize     - Оптимизация стратегий
    monitor      - Мониторинг системы
    backup       - Создание резервной копии
    restore      - Восстановление из резервной копии
    debug        - Инструменты отладки

Примеры:
    python script.py setup
    python script.py data download --symbols SBER GAZP --days 365
    python script.py data clean --file data/raw/stock_data.csv
    python script.py validate config --path config/settings.py
    python script.py test unit --coverage
    python script.py optimize backtest --strategy ma_crossover
    python script.py monitor system
    python script.py debug logs --tail 50
"""

import sys
import os
import argparse
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import logging
import json

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Добавляем текущую директорию в путь для импортов
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def load_data_fetcher():
    """Загрузка модуля работы с данными"""
    try:
        from data_fetcher import DataFetcher
        return DataFetcher()
    except ImportError as e:
        logger.error(f"DataFetcher недоступен: {e}")
        return None


def load_preprocessor():
    """Загрузка модуля предобработки"""
    try:
        from preprocess import (
            clean_data,
            validate_csv_format,
            load_and_validate_csv,
            clean_data as preprocess_data
        )
        return {
            'clean_data': clean_data,
            'validate_csv_format': validate_csv_format,
            'load_and_validate_csv': load_and_validate_csv,
            'preprocess_data': preprocess_data
        }
    except ImportError as e:
        logger.error(f"Preprocessor недоступен: {e}")
        return None


class ScriptRunner:
    """Главный класс для выполнения команд скрипта"""

    def __init__(self):
        self.data_fetcher = load_data_fetcher()
        self.preprocessor = load_preprocessor()

    def run_setup(self, args):
        """Настройка проекта"""
        print("🛠️ Настройка AI-торгового бота...")

        # Создание необходимых директорий
        directories = [
            "data/raw",
            "data/processed",
            "data/models",
            "config",
            "logs",
            "backtest_results",
            "templates",
            ".github/workflows"
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✅ Создано: {directory}")

        # Проверка наличия файлов-конфигураций
        required = [".env.example", ".gitignore", "pytest.ini"]
        for file in required:
            if Path(file).exists():
                print(f"✅ Есть: {file}")
            else:
                print(f"⚠️ Нет: {file}")

        # Создание базовых конфигурационных файлов
        if not Path("config/strategies.yaml").exists():
            with open("config/strategies.yaml", 'w') as f:
                f.write("""strategies:
  trendbot_sma_rsi:
    module: strategies.trendbot
    class: SMACrossRSIStrategy
    params:
      fast_sma: 50
      slow_sma: 200
      rsi_period: 14

  basic_ma_crossover:
    module: tbank_strategies
    class: MovingAverageCrossoverStrategy
    params:
      fast_period: 10
      slow_period: 20
""")
            print("✅ Создан: config/strategies.yaml")

        print("\n📋 Следующие шаги:")
        print("1. Скопируйте .env.example в .env и заполните API ключи")
        print("2. Настройте config/settings.py под свои нужды")
        print("3. Запустите: python script.py data download")
        print("4. Протестируйте: python script.py test unit")

    def run_data_command(self, args):
        """Команды работы с данными"""
        if args.subcommand == "download":
            self._data_download(args)
        elif args.subcommand == "clean":
            self._data_clean(args)
        elif args.subcommand == "validate":
            self._data_validate(args)
        elif args.subcommand == "preprocess":
            self._data_preprocess(args)
        else:
            print(f"Неизвестная подкоманда: {args.subcommand}")
            print("Доступные: download, clean, validate, preprocess")

    def _data_download(self, args):
        """Загрузка данных"""
        if not self.data_fetcher:
            print("❌ DataFetcher недоступен")
            return

        symbols = getattr(args, 'symbols', ['SBER'])
        days = getattr(args, 'days', 365)

        print(f"📈 Загрузка данных: {symbols} за {days} дней")

        for symbol in symbols:
            try:
                result = self.data_fetcher.fetch_historical_data(
                    symbol, f"{datetime.now().year - 1}-01-01",
                    datetime.now().strftime("%Y-%m-%d")
                )

                if result:
                    filename = f"data/raw/{symbol}_{days}d.csv"
                    result.to_csv(filename)
                    print(f"✅ {symbol}: {len(result)} записей в {filename}")
                else:
                    print(f"⚠️ {symbol}: данные не получены")

            except Exception as e:
                print(f"❌ {symbol}: {e}")

    def _data_clean(self, args):
        """Очистка данных"""
        if not self.preprocessor:
            print("❌ Preprocessor недоступен")
            return

        file_path = getattr(args, 'file', None)
        if not file_path:
            print("❌ Укажите файл: --file data/raw/stock_data.csv")
            return

        if not Path(file_path).exists():
            print(f"❌ Файл не найден: {file_path}")
            return

        try:
            print(f"🧹 Очистка данных: {file_path}")

            # Загрузка данных
            df = pd.read_csv(file_path)
            print(f"📊 Загружено: {len(df)} записей")

            # Очистка
            cleaned_df = self.preprocessor['clean_data'](df)
            print(f"📊 После очистки: {len(cleaned_df)} записей")

            # Сохранение
            cleaned_file = file_path.replace('.csv', '_cleaned.csv')
            cleaned_df.to_csv(cleaned_file, index=False)
            print(f"💾 Сохранено: {cleaned_file}")

        except Exception as e:
            print(f"❌ Ошибка очистки: {e}")

    def _data_validate(self, args):
        """Валидация данных"""
        if not self.preprocessor:
            print("❌ Preprocessor недоступен")
            return

        file_path = getattr(args, 'file', None)
        if not file_path:
            print("❌ Укажите файл: --file data/raw/stock_data.csv")
            return

        if not Path(file_path).exists():
            print(f"❌ Файл не найден: {file_path}")
            return

        try:
            print(f"🔍 Валидация: {file_path}")
            is_valid, errors = self.preprocessor['validate_csv_format'](file_path)

            if is_valid:
                print("✅ Данные валидны!")
            else:
                print("❌ Найдены ошибки валидации:")
                for error in errors:
                    print(f"  - {error}")

        except Exception as e:
            print(f"❌ Ошибка валидации: {e}")

    def _data_preprocess(self, args):
        """Предобработка данных"""
        if not self.preprocessor:
            print("❌ Preprocessor недоступен")
            return

        file_path = getattr(args, 'file', None)
        if not file_path:
            print("❌ Укажите файл: --file data/raw/stock_data.csv")
            return

        print(f"✨ Предобработка данных: {file_path}")

        try:
            # Валидация и загрузка
            is_valid, errors = self.preprocessor['validate_csv_format'](file_path)
            if not is_valid:
                print("❌ Данные невалидны:")
                for error in errors:
                    print(f"  - {error}")
                return

            df, load_errors = self.preprocessor['load_and_validate_csv'](file_path)
            if df is None:
                print("❌ Ошибка загрузки:")
                for error in load_errors:
                    print(f"  - {error}")
                return

            print(f"📊 Загружено: {len(df)} записей")

            # Предобработка
            from preprocess import create_returns_features, add_lagged_features
            processed_df = create_returns_features(df)
            processed_df = add_lagged_features(processed_df, ['close', 'volume'])

            print(f"📊 После предобработки: {len(processed_df)} записей, {len(processed_df.columns)} колонок")

            # Сохранение
            processed_file = file_path.replace('.csv', '_processed.csv')
            processed_df.to_csv(processed_file)
            print(f"💾 Сохранено: {processed_file}")

        except Exception as e:
            print(f"❌ Ошибка предобработки: {e}")

    def run_validate_command(self, args):
        """Команды валидации"""
        if args.subcommand == "config":
            self._validate_config(args)
        elif args.subcommand == "data":
            self._validate_data(args)
        else:
            print(f"Неизвестная подкоманда: {args.subcommand}")

    def _validate_config(self, args):
        """Валидация конфигурации"""
        config_path = getattr(args, 'path', 'config/settings.py')

        if not Path(config_path).exists():
            print(f"❌ Файл конфигурации не найден: {config_path}")
            return

        try:
            print(f"🔍 Валидация конфигурации: {config_path}")

            # Импорт и проверка настроек
            from config.settings import settings

            # Проверка критических настроек
            issues = []

            if not settings.api.tinkoff_token:
                issues.append("TINKOFF_TOKEN не указан (работа в демо режиме)")

            if settings.api.sandbox:
                issues.append("Используется песочница T-Bank API")

            if settings.risk.max_drawdown > 0.5:
                issues.append(".3f")

            if issues:
                print("⚠️ Найдены замечания:")
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print("✅ Конфигурация валидна!")

            # Показать ключевые настройки
            print(f"🎯 Торг. инструменты: {settings.trading.symbols}")
            print(f"💰 Нач. капитал: ${settings.trading.initial_capital}")
            print(f"📊 Комиссия: {settings.trading.commission:.3%}")
            print(f"📈 Стратегия: {settings.backtest.default_strategy}")

        except Exception as e:
            print(f"❌ Ошибка валидации конфигурации: {e}")

    def _validate_data(self, args):
        """Валидация структуры данных"""
        file_path = getattr(args, 'file', None)
        if file_path:
            self._data_validate(args)
        else:
            print("❌ Укажите файл: --file data/processed/stock_data.csv")

    def run_test_command(self, args):
        """Команды тестирования"""
        if args.subcommand == "unit":
            self._run_unit_tests(args)
        elif args.subcommand == "integration":
            self._run_integration_tests(args)
        elif args.subcommand == "all":
            self._run_all_tests(args)
        else:
            print(f"Неизвестная подкоманда: {args.subcommand}")

    def _run_unit_tests(self, args):
        """Запуск unit-тестов"""
        coverage = getattr(args, 'coverage', False)

        cmd = ["pytest", "tests/", "-v"]
        if coverage:
            cmd.extend(["--cov=.", "--cov-report=term-missing"])

        print("🧪 Запуск unit-тестов...")
        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode == 0

    def _run_integration_tests(self, args):
        """Запуск интеграционных тестов"""
        print("🔗 Запуск интеграционных тестов...")
        # Для интеграционных тестов можно добавить отдельные маркеры
        result = subprocess.run([
            "pytest", "tests/", "-v", "-k", "integration",
            "--cov=.||", "--cov-report=term-missing"
        ], cwd=project_root)
        return result.returncode == 0

    def _run_all_tests(self, args):
        """Запуск всех тестов"""
        print("🧪 Запуск всех тестов...")
        result = subprocess.run([
            "pytest", "tests/", "-v",
            "--cov=.", "--cov-report=term-missing"
        ], cwd=project_root)
        return result.returncode == 0

    def run_optimize_command(self, args):
        """Команды оптимизации"""
        if args.subcommand == "backtest":
            self._optimize_backtest(args)
        elif args.subcommand == "hyperparams":
            self._optimize_hyperparams(args)
        else:
            print(f"Неизвестная подкоманда: {args.subcommand}")

    def _optimize_backtest(self, args):
        """Оптимизация стратегии через бэктестирование"""
        strategy = getattr(args, 'strategy', 'ma_crossover')
        print(f"🔬 Оптимизация через бэктестирование: {strategy}")

        # Здесь можно добавить логику оптимизации параметров стратегии
        print("✅ Оптимизация завершена (заглушка)")

    def _optimize_hyperparams(self, args):
        """Оптимизация гиперпараметров"""
        print("🎯 Оптимизация гиперпараметров с Optuna...")
        print("✅ Оптимизация завершена (заглушка)")

    def run_monitor_command(self, args):
        """Команды мониторинга"""
        if args.subcommand == "system":
            self._monitor_system(args)
        elif args.subcommand == "bot":
            self._monitor_bot(args)
        else:
            print(f"Неизвестная подкоманда: {args.subcommand}")

    def _monitor_system(self, args):
        """Мониторинг системы"""
        print("📊 Мониторинг системы...")

        # Использование RAM
        import psutil
        memory = psutil.virtual_memory()
        print(f"💾 RAM: {memory.percent}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")

        # Дисковое пространство
        disk = psutil.disk_usage('/')
        print(f"💿 Диск: {disk.percent}% ({disk.used/1024**3:.1f}GB/{disk.total/1024**3:.1f}GB)")

        # CPU
        cpu = psutil.cpu_percent(interval=1)
        print(f"🖥️ CPU: {cpu}%")

        # Файлы проекта
        data_files = list(Path("data").rglob("*.csv"))
        print(f"📁 CSV файлов: {len(data_files)}")
        for file in data_files[:3]:  # показать первые 3
            size = file.stat().st_size / 1024  # KB
            print(f"  - {file.name}: {size:.1f}KB")

    def _monitor_bot(self, args):
        """Мониторинг бота"""
        print("🤖 Мониторинг бота...")

        try:
            from app import TradingBotApp
            bot = TradingBotApp()

            # Получение статуса
            status = bot.get_status()
            if status.get('error'):
                print(f"❌ {status['error']}")
            else:
                print(f"📊 Бот запущен: {status.get('is_running', 'Неизвестно')}")
                print(f"💰 Трейдинг: {status.get('is_trading', 'Неизвестно')}")
                print(f"🎯 Моделей: {len(status.get('models_loaded', []))}")
                print(f"📈 Анализ: {status.get('last_analysis', 'Не выполнялся')}")

        except Exception as e:
            print(f"❌ Ошибка мониторинга: {e}")

    def run_debug_command(self, args):
        """Команды отладки"""
        if args.subcommand == "logs":
            self._debug_logs(args)
        elif args.subcommand == "config":
            self._debug_config(args)
        elif args.subcommand == "data":
            self._debug_data(args)
        else:
            print(f"Неизвестная подкоманда: {args.subcommand}")

    def _debug_logs(self, args):
        """Просмотр логов"""
        lines = getattr(args, 'tail', 20)
        log_file = "trading_bot.log"

        if Path(log_file).exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.readlines()
                for line in content[-lines:]:
                    print(line.rstrip())
        else:
            print("❌ Файл логов не найден")

    def _debug_config(self, args):
        """Отладка конфигурации"""
        print("🔧 Отладка конфигурации...")

        try:
            from config.settings import settings

            # Показать все настройки
            config_dict = {
                'trading': {
                    'symbols': settings.trading.symbols,
                    'initial_capital': settings.trading.initial_capital,
                    'commission': settings.trading.commission
                },
                'api': {
                    'token_set': bool(settings.api.tinkoff_token),
                    'sandbox': settings.api.sandbox
                },
                'risk': {
                    'max_drawdown': settings.risk.max_drawdown,
                    'stop_loss': settings.risk.stop_loss_percent,
                    'take_profit': settings.risk.take_profit_percent
                }
            }

            print(json.dumps(config_dict, indent=2, ensure_ascii=False))

        except Exception as e:
            print(f"❌ Ошибка отладки конфигурации: {e}")

    def _debug_data(self, args):
        """Отладка данных"""
        data_dir = Path("data")
        if not data_dir.exists():
            print("❌ Директория data не существует")
            return

        print("💾 Отладка данных...")

        # Статистика файлов
        csv_files = list(data_dir.rglob("*.csv"))
        print(f"📁 Всего CSV файлов: {len(csv_files)}")

        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                relative_path = file_path.relative_to(project_root)
                print(f"📊 {relative_path}: {len(df)} строк, {len(df.columns)} колонок")
            except Exception as e:
                print(f"❌ {file_path}: {e}")


def create_parser():
    """Создание парсера аргументов"""
    parser = argparse.ArgumentParser(
        description="УНИВЕРСАЛЬНЫЙ УТИЛИТАРНЫЙ СКРИПТ ДЛЯ AI-Торгового Бота T-Bank",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='command', help='Доступные команды')

    # Setup команда
    subparsers.add_parser('setup', help='Первоначальная настройка проекта')

    # Data команды
    data_parser = subparsers.add_parser('data', help='Работа с данными')
    data_subparsers = data_parser.add_subparsers(dest='subcommand')
    data_subparsers.required = True

    # data download
    download_parser = data_subparsers.add_parser('download', help='Загрузка данных')
    download_parser.add_argument('--symbols', '-s', nargs='+', default=['SBER'],
                                help='Торговые символы')
    download_parser.add_argument('--days', type=int, default=365,
                                help='Количество дней')

    # data clean
    clean_parser = data_subparsers.add_parser('clean', help='Очистка данных')
    clean_parser.add_argument('--file', required=True,
                             help='Путь к CSV файлу для очистки')

    # data validate
    validate_parser = data_subparsers.add_parser('validate', help='Валидация данных')
    validate_parser.add_argument('--file', required=True,
                                help='Путь к CSV файлу для валидации')

    # data preprocess
    preprocess_parser = data_subparsers.add_parser('preprocess', help='Предобработка данных')
    preprocess_parser.add_argument('--file', required=True,
                                  help='Путь к CSV файлу для предобработки')

    # Validate команда
    validate_cmd_parser = subparsers.add_parser('validate', help='Валидация')
    validate_subparsers = validate_cmd_parser.add_subparsers(dest='subcommand')
    validate_subparsers.required = True

    # validate config
    config_valid_parser = validate_subparsers.add_parser('config', help='Валидация конфигурации')
    config_valid_parser.add_argument('--path', default='config/settings.py',
                                   help='Путь к файлу конфигурации')

    # validate data
    data_valid_parser = validate_subparsers.add_parser('data', help='Валидация данных')
    data_valid_parser.add_argument('--file', help='Путь к CSV файлу')

    # Test команда
    test_parser = subparsers.add_parser('test', help='Запуск тестов')
    test_subparsers = test_parser.add_subparsers(dest='subcommand')
    test_subparsers.required = True

    test_subparsers.add_parser('unit', help='Unit тесты')
    test_subparsers.add_parser('integration', help='Интеграционные тесты')
    test_subparsers.add_parser('all', help='Все тесты')

    for p in [test_subparsers.choices['unit'], test_subparsers.choices['integration'], test_subparsers.choices['all']]:
        p.add_argument('--coverage', action='store_true',
                      help='Показать покрытие кода')

    # Optimize команда
    optimize_parser = subparsers.add_parser('optimize', help='Оптимизация')
    optimize_subparsers = optimize_parser.add_subparsers(dest='subcommand')
    optimize_subparsers.required = True

    # optimize backtest
    backtest_opt_parser = optimize_subparsers.add_parser('backtest', help='Оптимизация через бэктест')
    backtest_opt_parser.add_argument('--strategy', default='ma_crossover',
                                   help='Стратегия для оптимизации')

    # optimize hyperparams
    optimize_subparsers.add_parser('hyperparams', help='Оптимизация гиперпараметров')

    # Monitor команда
    monitor_parser = subparsers.add_parser('monitor', help='Мониторинг')
    monitor_subparsers = monitor_parser.add_subparsers(dest='subcommand')
    monitor_subparsers.required = True

    monitor_subparsers.add_parser('system', help='Мониторинг системы')
    monitor_subparsers.add_parser('bot', help='Мониторинг бота')

    # Debug команда
    debug_parser = subparsers.add_parser('debug', help='Инструменты отладки')
    debug_subparsers = debug_parser.add_subparsers(dest='subcommand')
    debug_subparsers.required = True

    # debug logs
    logs_debug_parser = debug_subparsers.add_parser('logs', help='Просмотр логов')
    logs_debug_parser.add_argument('--tail', type=int, default=20,
                                  help='Количество строк с конца')

    debug_subparsers.add_parser('config', help='Отладка конфигурации')
    debug_subparsers.add_parser('data', help='Отладка данных')

    return parser


def main():
    """Главная функция"""
    parser = create_parser()
    args = parser.parse_args()

    if not hasattr(args, 'command'):
        parser.print_help()
        return

    runner = ScriptRunner()

    try:
        # Выполнение команд
        if args.command == 'setup':
            runner.run_setup(args)

        elif args.command == 'data':
            runner.run_data_command(args)

        elif args.command == 'validate':
            runner.run_validate_command(args)

        elif args.command == 'test':
            runner.run_test_command(args)

        elif args.command == 'optimize':
            runner.run_optimize_command(args)

        elif args.command == 'monitor':
            runner.run_monitor_command(args)

        elif args.command == 'debug':
            runner.run_debug_command(args)

        else:
            print(f"❌ Неизвестная команда: {args.command}")
            parser.print_help()

    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()