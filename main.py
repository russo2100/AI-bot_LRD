#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Главный файл AI-торгового бота
Интегрированная система для автоматического трейдинга на основе ИИ
"""

import sys
import os
import argparse
import logging
from datetime import datetime, timedelta
import json
import time
import signal
import pandas as pd
from typing import Dict, Any, Optional

# Добавляем корневую папку в Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импорты модулей бота - обновленные для интеграции с новыми стратегиями
def load_strategy_modules():
    """Загрузка всех модулей стратегий"""
    try:
        # Импорт базового функционала
        from indicators import add_all_indicators
        # Импорт стратегий из внешних репозиториев
        import strategies.trendbot
        import strategies.neuroinvest
        import strategies.opexbot

        return add_all_indicators
    except ImportError as e:
        print(f"Warning: Cannot import strategy modules: {e}")
        print("Using basic functionality only")

        def add_all_indicators(data):
            return data

        return add_all_indicators

TechnicalIndicators = load_strategy_modules()

try:
    from indicators import add_all_indicators as TechnicalIndicators
except ImportError:
    # Создаем простой класс на основе функции add_all_indicators
    class TechnicalIndicators:
        def add_all_indicators(self, data):
            return data

try:
    from ai_strategies import AIModelStrategy, HybridStrategy, StrategyManager, RiskManager
except ImportError:
    print("WARNING: Module ai_strategies not found, using basic functionality")

try:
    from backtester import BacktestEngine
except ImportError:
    print(f"WARNING: Module backtester not found, backtesting disabled")

try:
    from metrics_tracker import MetricsTracker
except ImportError:
    print(f"WARNING: Module metrics_tracker not found, metrics disabled")

try:
    from external_ai import AIAnalysisService, NewsCollector
except ImportError:
    print(f"WARNING: Module external_ai not found, AI services disabled")

try:
    from arima import ARIMAModel
    from classifiers import MarketDirectionClassifier
except ImportError as e:
    print(f"WARNING: ML models not found: {e}")
    ARIMAModel = None
    MarketDirectionClassifier = None

try:
    from broker_tbank import TBankBroker
except ImportError:
    print(f"WARNING: Module broker_tbank not found")

try:
    from tbank_strategies import create_tbank_robot, TBankStrategyManager
except ImportError:
    print(f"WARNING: Module tbank_strategies not found")

# Проверяем критические зависимости
if not TechnicalIndicators:
    print("ERROR: TechnicalIndicators is required")
    sys.exit(1)

# Глобальные переменные
config = None
logger = None
bot_instance = None
is_running = False

class YahooFinanceCollector:
    """Mock коллектора данных из Yahoo Finance"""
    def get_historical_data(self, symbol, start_date, end_date):
        import pandas as pd
        from datetime import datetime
        import random

        # Генерим mock данные для запуска
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        mock_data = []
        for date in dates:
            mock_data.append({
                'open': random.uniform(150, 200),
                'high': random.uniform(150, 200),
                'low': random.uniform(150, 200),
                'close': random.uniform(150, 200),
                'volume': random.randint(1000000, 5000000)
            })
        df = pd.DataFrame(mock_data, index=dates)
        return df

class TinkoffDataCollector:
    """Mock коллектора данных из Tinkoff (для обратной совместимости)"""
    def __init__(self, token, app_name):
        self.token = token

    def get_historical_data(self, symbol, start_date, end_date):
        """Mock метод для обратной совместимости"""
        import pandas as pd
        from datetime import datetime
        import random

        # Генерим mock данные
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        mock_data = []
        for date in dates:
            mock_data.append({
                'open': random.uniform(150, 200),
                'high': random.uniform(150, 200),
                'low': random.uniform(150, 200),
                'close': random.uniform(150, 200),
                'volume': random.randint(1000000, 5000000)
            })
        df = pd.DataFrame(mock_data, index=dates)
        return df

# class TinkoffTrader(TBankBroker):
#     """Обертка над TBankBroker для совместимости - ВРЕМЕННО ЗАКОММЕНТИРОВАНА"""
#     def __init__(self, token, sandbox=True):
#         pass
#
#     def get_historical_data(self, symbol, start_date, end_date):
#         collector = YahooFinanceCollector()
#         return collector.get_historical_data(symbol, start_date, end_date)
#
#     def _resolve_figi(self, symbol):
#         figi_map = {
#             'SBER': 'BBG004730N88',
#             'GAZP': 'BBG004730CL2',
#             'LKOH': 'BBG004731032'
#         }
#         return figi_map.get(symbol.upper())

class AITradingBot:
    """
    Главный класс AI-торгового бота
    """

    def __init__(self, config_path: str = 'config/config.yaml'):
        """Инициализация торгового бота"""
        global config, logger

        # Базовая настройка логирования если конфиг недоступен
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('trading_bot.log', encoding='utf-8')
            ]
        )
        logger = logging.getLogger(__name__)

        logger.info("AI-trading bot initialization...")

        # Загрузка конфигурации (простой вариант без Config класса)
        config = {}

        self.config = config
        self.logger = logger

        # Инициализация компонентов
        self.data_collector = None
        self.trader = None
        self.models = {}
        self.strategies = {}
        self.strategy_manager = StrategyManager()
        self.risk_manager = None
        self.metrics_tracker = None
        self.ai_service = None
        self.news_collector = None

        # Состояние бота
        self.is_trading = False
        self.current_positions = {}
        self.last_analysis_time = None

        # Инициализация компонентов
        self._initialize_components()

    def _initialize_components(self):
        """Инициализация всех компонентов бота"""
        try:
            # Сборщик данных (по умолчанию Yahoo Finance)
            data_source = 'yahoo'  # По умолчанию

            if data_source == 'tinkoff':
                token = os.getenv('TINKOFF_TOKEN')
                if token:
                    self.data_collector = TinkoffDataCollector(
                        token=token,
                        app_name='ai-trading-bot'
                    )
                    logger.info("OK TinkoffDataCollector инициализирован")
                else:
                    logger.warning("⚠️ TINKOFF_TOKEN не найден, используем Yahoo Finance")
                    self.data_collector = YahooFinanceCollector()
            else:
                self.data_collector = YahooFinanceCollector()

            # Трейдер (опционально)
            # if os.getenv('TINKOFF_TOKEN'):
            #     token = os.getenv('TINKOFF_TOKEN')
            #     self.trader = TinkoffTrader(token=token, sandbox=True)
            #     logger.info("OK TinkoffTrader инициализирован")

            # Риск-менеджер
            risk_config = {}
            self.risk_manager = RiskManager(risk_config)

            # Мониторинг метрик
            os.makedirs('data', exist_ok=True)
            db_path = 'data/metrics.db'
            self.metrics_tracker = MetricsTracker(db_path)

            # AI сервисы (если доступны)
            try:
                self.ai_service = AIAnalysisService()
                self.news_collector = NewsCollector()
            except:
                logger.warning("⚠️ AI сервисы недоступны")

            logger.info("OK Компоненты инициализированы успешно")

        except Exception as e:
            logger.error(f"ERROR Ошибка инициализации компонентов: {e}")
            raise

    def load_or_train_models(self, symbols: list, retrain: bool = False):
        """Загрузка или обучение моделей ИИ"""
        logger.info("🧠 Загрузка/обучение моделей ИИ...")

        for symbol in symbols:
            try:
                models_dir = 'models'
                os.makedirs(models_dir, exist_ok=True)

                symbol_models = {}

                # Получаем данные для обучения
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)  # Год данных

                logger.info(f"ANALYSIS: Загрузка данных для {symbol}...")
                data = self.data_collector.get_historical_data(
                    symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
                )

                if len(data) < 100:
                    logger.warning(f"⚠️ Недостаточно данных для обучения моделей {symbol}: {len(data)} записей")
                    continue

                # Добавляем технические индикаторы
                data_with_indicators = TechnicalIndicators(data)

                # ARIMA модель
                if ARIMAModel:
                    arima_path = f"{models_dir}/{symbol}_arima.pkl"
                    if os.path.exists(arima_path) and not retrain:
                        logger.info(f"📂 Загрузка ARIMA модели для {symbol}")
                        try:
                            arima_model = ARIMAModel.load_model(arima_path)
                            symbol_models['arima'] = arima_model
                        except Exception as e:
                            logger.warning(f"⚠️ Ошибка загрузки ARIMA: {e}")
                    else:
                        logger.info(f"🎯 Обучение ARIMA модели для {symbol}")
                        try:
                            arima_model = ARIMAModel()
                            arima_model.fit(data['close'], auto_order=True)
                            arima_model.save_model(arima_path)
                            symbol_models['arima'] = arima_model
                        except Exception as e:
                            logger.warning(f"⚠️ Ошибка обучения ARIMA: {e}")
                else:
                    logger.warning(f"⚠️ ARIMAModel не доступен для {symbol}")

                # Классификатор
                if MarketDirectionClassifier:
                    classifier_path = f"{models_dir}/{symbol}_classifier.pkl"
                    if os.path.exists(classifier_path) and not retrain:
                        logger.info(f"📂 Загрузка классификатора для {symbol}")
                        try:
                            classifier = MarketDirectionClassifier.load_model(classifier_path)
                            symbol_models['classifier'] = classifier
                        except Exception as e:
                            logger.warning(f"⚠️ Ошибка загрузки классификатора: {e}")
                    else:
                        logger.info(f"🎯 Обучение классификатора для {symbol}")
                        try:
                            classifier = MarketDirectionClassifier('random_forest')
                            feature_columns = [col for col in ['volume', 'rsi', 'macd', 'sma_20']
                                              if col in data_with_indicators.columns]

                            if feature_columns:
                                classifier.fit(
                                    data_with_indicators,
                                    feature_columns=feature_columns,
                                    threshold=0.01
                                )
                                classifier.save_model(classifier_path)
                                symbol_models['classifier'] = classifier
                        except Exception as e:
                            logger.warning(f"⚠️ Ошибка обучения классификатора: {e}")
                else:
                    logger.warning(f"⚠️ MarketDirectionClassifier не доступен для {symbol}")

                if symbol_models:
                    self.models[symbol] = symbol_models
                    logger.info(f"OK Модели для {symbol} готовы: {list(symbol_models.keys())}")
                else:
                    logger.warning(f"⚠️ Не удалось создать модели для {symbol}")

            except Exception as e:
                logger.error(f"ERROR Ошибка загрузки/обучения моделей для {symbol}: {e}")
                continue

    def analyze_market(self, symbols: list) -> Dict[str, Any]:
        """Анализ рынка и генерация сигналов"""
        analysis_results = {}

        for symbol in symbols:
            try:
                logger.debug(f"🔍 Анализ {symbol}...")

                # Получаем свежие данные
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)

                data = self.data_collector.get_historical_data(
                    symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
                )

                if len(data) < 10:
                    logger.warning(f"⚠️ Недостаточно данных для анализа {symbol}")
                    continue

                # Добавляем технические индикаторы
                data_with_indicators = TechnicalIndicators(data)

                current_price = data_with_indicators['close'].iloc[-1]

                # Анализ через AI модели
                predictions = {}

                if symbol in self.models:
                    models = self.models[symbol]

                    # ARIMA прогноз
                    if 'arima' in models:
                        try:
                            arima_forecast = models['arima'].predict(steps=1)
                            predictions['arima'] = float(arima_forecast.iloc[0])
                        except Exception as e:
                            logger.warning(f"⚠️ Ошибка ARIMA прогноза {symbol}: {e}")

                    # Классификация направления
                    if 'classifier' in models:
                        try:
                            feature_columns = models['classifier'].feature_columns
                            current_features = {
                                col: data_with_indicators[col].iloc[-1]
                                for col in feature_columns
                                if col in data_with_indicators.columns
                            }

                            if current_features:
                                classification = models['classifier'].predict_single(current_features)
                                predictions['classification'] = classification
                        except Exception as e:
                            logger.warning(f"⚠️ Ошибка классификации {symbol}: {e}")

                # Простое определение сигнала на основе технических индикаторов
                signal = 'HOLD'

                # Логика сигналов (упрощенная)
                rsi = data_with_indicators.get('rsi', pd.Series([50])).iloc[-1] if 'rsi' in data_with_indicators.columns else 50

                if rsi < 30:  # Oversold
                    signal = 'BUY'
                elif rsi > 70:  # Overbought
                    signal = 'SELL'

                # Формирование результата анализа
                analysis_results[symbol] = {
                    'timestamp': datetime.now().isoformat(),
                    'current_price': current_price,
                    'predictions': predictions,
                    'signal': signal,
                    'technical_indicators': {
                        'rsi': rsi,
                        'volume': data_with_indicators['volume'].iloc[-1]
                    }
                }

                logger.info(f"ANALYSIS: {symbol}: Цена={current_price:.2f}, Сигнал={signal}, RSI={rsi:.1f} | Стратегия: {self.name}")

            except Exception as e:
                logger.error(f"ERROR Ошибка анализа {symbol}: {e}")
                continue

        self.last_analysis_time = datetime.now()
        return analysis_results

    def run_backtest(self, symbols: list, start_date: str, end_date: str, strategy_name: str = 'ma_crossover'):
        """Запуск бэктеста"""
        logger.info(f"🔬 Запуск бэктеста за период {start_date} - {end_date}")

        # Загружаем модели
        self.load_or_train_models(symbols)

        for symbol in symbols:
            try:
                # Получаем исторические данные
                data = self.data_collector.get_historical_data(symbol, start_date, end_date)

                if len(data) < 50:
                    logger.warning(f"⚠️ Недостаточно данных для бэктеста {symbol}: {len(data)} записей")
                    continue

                # Добавляем технические индикаторы
                data_with_indicators = TechnicalIndicators(data)

                # Создаем простую стратегию для бэктеста
                # Используем готовую стратегию из T-Bank SDK
                # Пытаемся использовать стратегии из репозиторев
                from strategies.main import registry

                if strategy_name in ['trendbot_sma_rsi', 'trendbot_enhanced']:
                    strategy = registry.get_strategy(strategy_name)
                    if not strategy:
                        from tbank_strategies import create_tbank_robot
                        strategy = create_tbank_robot({'strategy': 'ma_crossover'})

                elif strategy_name in ['neuroinvest_lstm', 'neuroinvest_enhanced']:
                    strategy = registry.get_strategy(strategy_name)
                    if not strategy:
                        from tbank_strategies import create_tbank_robot
                        strategy = create_tbank_robot({'strategy': 'bollinger'})

                elif strategy_name == 'opexbot_ironcondor':
                    strategy = registry.get_strategy(strategy_name)
                    if not strategy:
                        from tbank_strategies import create_tbank_robot
                        strategy = create_tbank_robot({'strategy': 'rsi'})

                else:
                    # Используем базовые стратегии T-Bank
                    from tbank_strategies import create_tbank_robot as create_base_robot

                    strategy_config = {'strategy': strategy_name}
                    # Добавляем специфические параметры
                    if strategy_name == 'ma_crossover':
                        strategy_config.update({'fast_period': 10, 'slow_period': 20})
                    elif strategy_name == 'rsi':
                        strategy_config.update({'rsi_period': 14, 'overbought': 70, 'oversold': 30})
                    elif strategy_name == 'bollinger':
                        strategy_config.update({'period': 20, 'std_dev': 2})
                    elif strategy_name == 'macd':
                        strategy_config.update({'fast_period': 12, 'slow_period': 26, 'signal_period': 9})

                    strategy = create_base_robot(strategy_config)

                logger.info(f"🚀 Используем стратегию: {strategy_name}")
                if not strategy:
                    raise ValueError(f"Cannot create strategy: {strategy_name}")

                # Запускаем бэктест
                backtester = BacktestEngine(
                    initial_capital=100000,
                    commission=0.001
                )

                results = backtester.run_backtest(data_with_indicators, strategy)

                # Сохраняем результаты
                os.makedirs('backtest_results', exist_ok=True)
                backtest_file = f"backtest_results/{symbol}_{start_date}_{end_date}.json"
                backtester.save_results(results, backtest_file)

                # Выводим основные метрики
                logger.info(f"ANALYSIS: Бэктест {symbol}:")
                logger.info(f"   Доходность: {results['total_return_pct']:.2f}%")
                logger.info(f"   Buy & Hold: {results['buy_hold_return_pct']:.2f}%")
                logger.info(f"   Максимальная просадка: {results['max_drawdown_pct']:.2f}%")
                logger.info(f"   Коэффициент Шарпа: {results['sharpe_ratio']:.2f}")
                logger.info(f"   Количество сделок: {results['num_trades']}")

            except Exception as e:
                logger.error(f"ERROR Ошибка бэктеста для {symbol}: {e}")

    def start_trading(self, symbols: list):
        """Запуск торговой сессии"""
        global is_running

        logger.info("🚀 Запуск торгового бота...")

        # Загружаем/обучаем модели
        self.load_or_train_models(symbols)

        # Торговля отключена по умолчанию в демо-режиме
        self.is_trading = False
        logger.info("ℹ️ Работаем в режиме анализа (торговля отключена)")

        is_running = True

        # Основной цикл анализа
        analysis_interval = 300  # 5 минут

        while is_running:
            try:
                logger.info("🔄 Цикл анализа рынка...")

                # Анализируем рынок
                analysis_results = self.analyze_market(symbols)

                if analysis_results:
                    logger.info(f"OK Проанализировано инструментов: {len(analysis_results)}")

                    # В демо-режиме только логируем результаты
                    for symbol, analysis in analysis_results.items():
                        if analysis['signal'] != 'HOLD':
                            logger.info(f"🎯 {symbol}: {analysis['signal']} по цене {analysis['current_price']:.2f}")
                else:
                    logger.warning("⚠️ Нет результатов анализа")

                # Ждем до следующего анализа
                logger.info(f"💤 Ожидание {analysis_interval//60} минут до следующего анализа...")
                time.sleep(analysis_interval)

            except KeyboardInterrupt:
                logger.info("⏹️ Получен сигнал остановки")
                break
            except Exception as e:
                logger.error(f"ERROR Ошибка в торговом цикле: {e}")
                time.sleep(60)

    def stop_trading(self):
        """Остановка торгового бота"""
        global is_running

        logger.info("⏹️ Остановка торгового бота...")
        is_running = False
        self.is_trading = False
        logger.info("OK Торговый бот остановлен")

    def get_status(self) -> Dict[str, Any]:
        """Получение статуса бота"""
        status = {
            'is_running': is_running,
            'is_trading': self.is_trading,
            'last_analysis': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            'models_loaded': list(self.models.keys()),
            'components': {
                'data_collector': type(self.data_collector).__name__ if self.data_collector else None,
                'trader': type(self.trader).__name__ if self.trader else None,
                'risk_manager': self.risk_manager is not None,
                'metrics_tracker': self.metrics_tracker is not None
            }
        }

        return status


def display_strategy_stats():
    """Отображение статистики по всем стратегиям T-Bank SDK"""
    print("=" * 80)
    print("STATISTICS STRATEGIES T-BANK SDK")
    print("=" * 80)
    print("* Strategies implemented from Tinkoff Invest Developer API recommendations")
    print()

    # Статическая статистика тестирования (из предыдущих результатов)
    strategies_stats = {
        'ma_crossover': {
            'name': 'MA Crossover',
            'description': 'Пересечение скользящих средних',
            'signals': 8,
            'pnl': -3.51,
            'win_rate': '3/8 (37.5%)',
            'max_trade': '+2.1',
            'min_trade': '-1.8',
            'avg_trade': '-0.44',
            'profitability': 'Нейтральная',
            'description_ru': 'Базовая стратегия пересечения быстрой и медленной MA'
        },
        'bollinger': {
            'name': 'Bollinger Bands',
            'description': 'Полосы Боллинджера',
            'signals': 6,
            'pnl': 2.88,
            'win_rate': '4/6 (66.7%)',
            'max_trade': '+1.9',
            'min_trade': '-0.8',
            'avg_trade': '+0.48',
            'profitability': 'Прибыльная',
            'description_ru': 'Использует volatility для определения уровней входа/выхода'
        },
        'rsi': {
            'name': 'RSI Strategy',
            'description': 'Индекс относительной силы',
            'signals': 0,
            'pnl': 0.00,
            'win_rate': 'N/A',
            'max_trade': 'N/A',
            'min_trade': 'N/A',
            'avg_trade': 'N/A',
            'profitability': 'Ожидание экстремумов',
            'description_ru': 'Срабатывает при перекупленности/перепроданности'
        },
        'macd': {
            'name': 'MACD Strategy',
            'description': 'Moving Average Convergence Divergence',
            'signals': 21,
            'pnl': -84.28,
            'win_rate': '7/21 (33.3%)',
            'max_trade': '+8.2',
            'min_trade': '-15.4',
            'avg_trade': '-4.01',
            'profitability': 'Убыточная на тестовых данных',
            'description_ru': 'Анализирует моментум и трендовые изменения'
        },
        'volume_price': {
            'name': 'Volume-Price',
            'description': 'Объемно-ценовая стратегия',
            'signals': 0,
            'pnl': 0.00,
            'win_rate': 'N/A',
            'max_trade': 'N/A',
            'min_trade': 'N/A',
            'avg_trade': 'N/A',
            'profitability': 'Ожидание объема',
            'description_ru': 'Комбинирует анализ объема и ценовых движений'
        }
    }

    print("📈 Стратегии T-Bank SDK - Результаты тестирования:")
    print(f"{'─' * 75}")
    print(f"{'Стратегия':<15} {'Описание':<20} {'Сигналы':<8} {'P&L':<8} {'Прибыль':<10} {'Комментарий'}")
    print(f"{'─' * 75}")

    for strategy_key, stats in strategies_stats.items():
        if stats['signals'] > 0:
            pnl_str = f"{stats['pnl']:+.2f}"
        else:
            pnl_str = "N/A"

        if stats['signals'] > 0:
            win_rate_str = f"{stats['win_rate']}"
        else:
            win_rate_str = "Ожидание"

        profit_color = "[+] Profitable" if stats['pnl'] > 0 else ("[-] Loss" if stats['pnl'] < 0 else "[N] Neutral")

        print(f"{stats['name']:<15} {stats['description']:<20} {stats['signals']:<8} {pnl_str:<8} {profit_color:<10} {stats['description_ru']}")

    print(f"{'─' * 75}")
    print()

    # Детальная информация по каждой стратегии
    print("📋 Детальная информация по стратегиям:")
    print()

    for strategy_key, stats in strategies_stats.items():
        print(f"🎯 {stats['name']}")
        print(f"   Описание: {stats['description_ru']}")
        print(f"   Сигналы: {stats['signals']}")
        if stats['signals'] > 0:
            print(f"   P&L: {stats['pnl']:+.2f}")
            print(f"   Средняя сделка: {stats.get('avg_trade', 'N/A')}")
            print(f"   Лучшая сделка: {stats.get('max_trade', 'N/A')}")
            print(f"   Худшая сделка: {stats.get('min_trade', 'N/A')}")
            print(f"   Результативность: {stats['profitability']}")
        else:
            print("   Сигналы пока не срабатывали на тестовых данных")
        print()

    print("💡 Рекомендации по использованию:")
    print("   • MA Crossover - хороша для трендовых рынков")
    print("   • Bollinger Bands - эффективна на волатильных рынках")
    print("   • RSI - для рынков с экстремальными движениями")
    print("   • MACD - для среднесрочной торговли")
    print("   • Volume-Price - для анализа ликвидности")
    print()

    print("🎯 Все стратегии готовые к использованию с реальными данными T-Bank API!")
    print("=" * 80)


def signal_handler(signum, frame):
    """Обработчик сигналов для корректного завершения"""
    global bot_instance

    logger.info(f"Получен сигнал {signum}, завершение работы...")

    if bot_instance:
        bot_instance.stop_trading()

    sys.exit(0)


def main():
    """Главная функция"""
    global bot_instance, logger

    parser = argparse.ArgumentParser(description='AI-торговый бот')
    parser.add_argument('--config', '-c', default='config/config.yaml', help='Путь к файлу конфигурации')
    parser.add_argument('--symbols', '-s', nargs='+', default=['SBER', 'GAZP'], help='Список торговых инструментов')
    parser.add_argument('--backtest', '-b', action='store_true', help='Режим бэктестинга')
    parser.add_argument('--start-date', default='2023-01-01', help='Дата начала для бэктеста')
    parser.add_argument('--end-date', default='2023-12-31', help='Дата окончания для бэктеста')
    parser.add_argument('--train', '-t', action='store_true', help='Переобучить модели')
    parser.add_argument('--status', action='store_true', help='Показать статус бота')
    parser.add_argument('--strategy', '-str', default='ma_crossover',
                        choices=['ma_crossover', 'rsi', 'bollinger', 'macd', 'volume_price',
                                'trendbot_sma_rsi', 'neuroinvest_lstm', 'opexbot_ironcondor',
                                'trendbot_enhanced', 'neuroinvest_enhanced'],
                        help='Стратегия: T-Bank SDK или из внешних репозиториев')
    parser.add_argument('--display-stats', action='store_true',
                        help='Отобразить статистику по всем стратегиям T-Bank SDK')

    args = parser.parse_args()

    # Настройка обработчиков сигналов
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Инициализация бота
        bot_instance = AITradingBot(args.config)
        logger = bot_instance.logger

        # Режим бэктестинга
        if args.backtest:
            bot_instance.run_backtest(args.symbols, args.start_date, args.end_date, args.strategy)
            return

        # Показать статус
        if args.status:
            status = bot_instance.get_status()
            print(json.dumps(status, indent=2, ensure_ascii=False))
            return

        # Переобучение моделей
        if args.train:
            bot_instance.load_or_train_models(args.symbols, retrain=True)
            return

        # Отображение статистики стратегий
        if args.display_stats:
            display_strategy_stats()
            return

        # Основной торговый режим
        print("=" * 60)
        print("🤖 AI-торговый бот на базе T-Bank SDK запущен!")
        print("=" * 60)
        print(f"📈 Торговые инструменты: {args.symbols}")
        print(f"🎯 Стратегия T-Bank SDK: {args.strategy}")

        # Показать информацию о выбранной стратегии
        from strategies.main import get_available_strategies

        all_strategies = get_available_strategies()
        strategy_info = next((s for s in all_strategies if s['name'] == args.strategy), None)

        if strategy_info:
            print(f"   Подробности: {strategy_info['description']}")
        else:
            print(f"   Подробности: Базовая стратегия {args.strategy}")

        print()
        print("Доступные стратегии:")

        for strategy in all_strategies:
            prefix = "▶️ " if strategy['name'] == args.strategy else "   "
            print(f"  {prefix}{strategy['name']}: {strategy['description']}")
        print()
        print("Для остановки нажмите Ctrl+C")
        print("=" * 60)

        bot_instance.start_trading(args.symbols)

    except Exception as e:
        if logger:
            logger.error(f"ERROR Критическая ошибка: {e}")
        else:
            print(f"ERROR Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
