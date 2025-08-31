#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main module for managing trading strategies
Интеграция различных торговых стратегий для T-Bank API
"""

import sys
import os
import logging
import yaml
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

# Добавляем корневую папку в Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tbank_strategies import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """
    Реестр торговых стратегий
    """

    def __init__(self):
        self._strategies = {}
        self._active_strategies = {}

    def register(self, name: str, strategy_class: type, config: Dict[str, Any] = None):
        """Регистрация стратегии"""
        self._strategies[name] = {
            'class': strategy_class,
            'config': config or {},
            'enabled': True
        }
        logger.info(f"Strategy '{name}' registered with config: {config}")

    def get_strategy(self, name: str, config: Dict[str, Any] = None) -> Optional[BaseStrategy]:
        """Получить экземпляр стратегии"""
        if name not in self._strategies:
            logger.error(f"Strategy '{name}' not found")
            return None

        strategy_info = self._strategies[name]
        if not strategy_info['enabled']:
            logger.warning(f"Strategy '{name}' is disabled")
            return None

        # Объединяем конфигурации
        strategy_config = {**strategy_info['config']}
        if config:
            strategy_config.update(config)

        try:
            return strategy_info['class'](config=strategy_config)
        except Exception as e:
            logger.error(f"Failed to create strategy '{name}': {e}")
            return None

    def list_strategies(self) -> List[str]:
        """Список всех зарегистрированных стратегий"""
        return list(self._strategies.keys())

    def get_config(self, name: str) -> Dict[str, Any]:
        """Получить конфигурацию стратегии"""
        if name in self._strategies:
            return self._strategies[name]['config']
        return {}

    def enable_strategy(self, name: str):
        """Включить стратегию"""
        if name in self._strategies:
            self._strategies[name]['enabled'] = True
            logger.info(f"Strategy '{name}' enabled")

    def disable_strategy(self, name: str):
        """Отключить стратегию"""
        if name in self._strategies:
            self._strategies[name]['enabled'] = False
            logger.info(f"Strategy '{name}' disabled")


# Глобальный реестр стратегий
registry = StrategyRegistry()


def register_strategy(name: str, strategy_class: type, config: Dict[str, Any] = None):
    """
    Регистрация стратегии в глобальном реестре
    """
    registry.register(name, strategy_class, config)


def load_strategies_from_config(config_file: str = "config/strategies.yaml"):
    """
    Загрузка стратегий из конфигурационного файла
    """
    try:
        if not os.path.exists(config_file):
            logger.warning(f"Config file {config_file} not found")
            return

        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if 'strategies' in config:
            for strategy_name, strategy_config in config['strategies'].items():
                if 'module' in strategy_config and 'class' in strategy_config:
                    # Динамический импорт
                    try:
                        module_name = strategy_config['module']
                        class_name = strategy_config['class']

                        __import__(module_name)
                        module = sys.modules[module_name]
                        strategy_class = getattr(module, class_name)

                        register_strategy(
                            strategy_name,
                            strategy_class,
                            strategy_config.get('params', {})
                        )

                    except Exception as e:
                        logger.error(f"Failed to load strategy {strategy_name}: {e}")

        logger.info(f"Loaded strategies from {config_file}")

    except Exception as e:
        logger.error(f"Failed to load strategies config: {e}")


def test_strategy(strategy_name: str, test_data: Any = None) -> Dict[str, Any]:
    """
    Тестирование стратегии
    """
    strategy = registry.get_strategy(strategy_name)
    if not strategy:
        return {'error': f'Strategy {strategy_name} not found'}

    # Здесь можно добавить логику тестирования
    try:
        # Тестовые данные для демонстрации
        if test_data is None:
            import pandas as pd
            import numpy as np
            np.random.seed(42)
            dates = pd.date_range('2024-01-01', periods=100, freq='D')
            test_data = pd.DataFrame({
                'close': np.random.uniform(100, 200, 100),
                'volume': np.random.randint(1000, 10000, 100)
            }, index=dates)

        signals = []
        for i in range(30, len(test_data)):
            signal = strategy.generate_signal(test_data, i)
            signals.append(signal)

        # Простая симуляция торговли
        pnl = 0
        position = None
        for i, signal in enumerate(signals):
            if signal == Signal.BUY and position != 'long':
                position = 'long'
                entry_price = test_data.iloc[i+30]['close']
            elif signal == Signal.SELL and position == 'long':
                exit_price = test_data.iloc[i+30]['close']
                pnl += exit_price - entry_price
                position = None

        return {
            'strategy': strategy_name,
            'signals_count': len(signals),
            'final_pnl': round(pnl, 2),
            'final_position': position,
            'status': 'completed'
        }

    except Exception as e:
        logger.error(f"Strategy test failed: {e}")
        return {'error': str(e)}


def get_available_strategies() -> List[Dict[str, Any]]:
    """
    Получить список доступных стратегий
    """
    result = []
    for name in registry.list_strategies():
        config = registry.get_config(name)
        result.append({
            'name': name,
            'config': config,
            'description': config.get('description', 'No description')
        })
    return result


# Инициализация стратегий при запуске модуля
def _initialize_strategies():
    """Инициализация встроенных стратегий"""
    try:
        # Загружаем конфигурацию если она существует
        config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
        config_file = os.path.join(config_dir, 'strategies.yaml')
        load_strategies_from_config(config_file)

        # Регистрируем базовую стратегию если не загружена из файла
        if not registry.list_strategies():
            # Импортируем и регистрируем встроенные стратегии
            from tbank_strategies import MovingAverageCrossoverStrategy
            register_strategy('tbank_ma_crossover', MovingAverageCrossoverStrategy)

    except Exception as e:
        logger.error(f"Failed to initialize strategies: {e}")


# Автоматическая инициализация
_initialize_strategies()


if __name__ == "__main__":
    # Тестирование модуля
    print("=== Trading Strategies Manager ===")
    print(f"Available strategies: {registry.list_strategies()}")

    # Тестируем каждую стратегию
    for strategy_name in registry.list_strategies():
        result = test_strategy(strategy_name)
        print(f"Test result for {strategy_name}: {result}")