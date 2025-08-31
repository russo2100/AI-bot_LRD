#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Готовые стратегии торговли на основе T-Bank SDK
Реализации типичных роботов из документации Tinkoff Invest API
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from ai_strategies import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Стратегия пересечения скользящих средних
    Одна из самых популярных стратегий в T-Bank SDK
    """

    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        super().__init__("TBank_MA_Crossover", config)
        self.fast_period = config.get('fast_period', 10)
        self.slow_period = config.get('slow_period', 20)

    def generate_signal(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """
        Генерация сигнала на основе пересечения MA
        """
        try:
            if current_idx < self.slow_period:
                return Signal.HOLD

            current_data = data.iloc[:current_idx + 1]

            # Расчет скользящих средних
            fast_ma = current_data['close'].rolling(self.fast_period).mean().iloc[-1]
            slow_ma = current_data['close'].rolling(self.slow_period).mean().iloc[-1]

            # Предыдущие значения для определения пересечения
            prev_fast_ma = current_data['close'].rolling(self.fast_period).mean().iloc[-2]
            prev_slow_ma = current_data['close'].rolling(self.slow_period).mean().iloc[-2]

            current_price = current_data['close'].iloc[-1]

            # Логика сигналов
            if fast_ma > slow_ma and prev_fast_ma <= prev_slow_ma:
                signal = Signal.BUY
                logger.info(f"BUY: {self.name}: Быстрая MA пересекла медленную вверх - сигнал BUY")
            elif fast_ma < slow_ma and prev_fast_ma >= prev_slow_ma:
                signal = Signal.SELL
                logger.info(f"SELL: {self.name}: Быстрая MA пересекла медленную вниз - сигнал SELL")
            else:
                signal = Signal.HOLD

            return signal

        except Exception as e:
            logger.error(f"ERROR Ошибка в стратегии MA Crossover: {e}")
            return Signal.HOLD


class RSIStrategy(BaseStrategy):
    """
    Стратегия на основе RSI индикатора
    Классическая стратегия из примеров T-Bank SDK
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("TBank_RSI", config)
        config = config or {}
        self.rsi_period = config.get('rsi_period', 14)
        self.overbought_level = config.get('overbought', 70)
        self.oversold_level = config.get('oversold', 30)

    def _calculate_rsi(self, prices: pd.Series, period: int) -> float:
        """Расчет RSI индикатора"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if not rsi.empty else 50

    def generate_signal(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """
        Генерация сигнала на основе RSI
        """
        try:
            if current_idx < self.rsi_period:
                return Signal.HOLD

            current_data = data.iloc[:current_idx + 1]

            # Расчет RSI
            rsi = self._calculate_rsi(current_data['close'], self.rsi_period)
            current_price = current_data['close'].iloc[-1]

            # Логика сигналов
            if rsi <= self.oversold_level:
                signal = Signal.BUY
                logger.info(f"BUY: {self.name}: RSI={rsi:.2f} (<= {self.oversold_level}) - oversold, сигнал BUY")
            elif rsi >= self.overbought_level:
                signal = Signal.SELL
                logger.info(f"SELL: {self.name}: RSI={rsi:.2f} (>= {self.overbought_level}) - overbought, сигнал SELL")
            else:
                signal = Signal.HOLD

            return signal

        except Exception as e:
            logger.error(f"ERROR Ошибка в стратегии RSI: {e}")
            return Signal.HOLD


class BollingerBandsStrategy(BaseStrategy):
    """
    Стратегия Bollinger Bands
    Популярная стратегия из примеров T-Bank SDK
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("TBank_Bollinger", config)
        config = config or {}
        self.period = config.get('period', 20)
        self.std_dev = config.get('std_dev', 2)

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int, std_dev: float):
        """Расчет Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        return upper_band.iloc[-1], sma.iloc[-1], lower_band.iloc[-1]

    def generate_signal(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """
        Генерация сигнала на основе Bollinger Bands
        """
        try:
            if current_idx < self.period:
                return Signal.HOLD

            current_data = data.iloc[:current_idx + 1]

            # Расчет полос Боллинджера
            upper, middle, lower = self._calculate_bollinger_bands(
                current_data['close'], self.period, self.std_dev
            )

            current_price = current_data['close'].iloc[-1]

            # Логика сигналов
            if current_price <= lower:
                signal = Signal.BUY
                logger.info(f"BUY: {self.name}: Цена {current_price:.2f} коснулась нижней полосы {lower:.2f} - сигнал BUY")
            elif current_price >= upper:
                signal = Signal.SELL
                logger.info(f"SELL: {self.name}: Цена {current_price:.2f} коснулась верхней полосы {upper:.2f} - сигнал SELL")
            else:
                signal = Signal.HOLD

            return signal

        except Exception as e:
            logger.error(f"ERROR Ошибка в стратегии Bollinger Bands: {e}")
            return Signal.HOLD


class MACDStrategy(BaseStrategy):
    """
    Стратегия MACD (Moving Average Convergence Divergence)
    Продвинутая стратегия из T-Bank SDK
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("TBank_MACD", config)
        config = config or {}
        self.fast_period = config.get('fast_period', 12)
        self.slow_period = config.get('slow_period', 26)
        self.signal_period = config.get('signal_period', 9)

    def _calculate_macd(self, prices: pd.Series, fast_p: int, slow_p: int, signal_p: int):
        """Расчет MACD"""
        fast_ema = prices.ewm(span=fast_p, adjust=False).mean()
        slow_ema = prices.ewm(span=slow_p, adjust=False).mean()

        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_p, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]

    def generate_signal(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """
        Генерация сигнала на основе MACD
        """
        try:
            if current_idx < self.slow_period:
                return Signal.HOLD

            current_data = data.iloc[:current_idx + 1]

            # Расчет MACD
            macd, signal_line, histogram = self._calculate_macd(
                current_data['close'], self.fast_period, self.slow_period, self.signal_period
            )

            # Предыдущие значения для определения пересечения
            prev_macd_data = current_data.iloc[:current_idx]
            if len(prev_macd_data) >= self.slow_period:
                prev_macd, prev_signal, _ = self._calculate_macd(
                    prev_macd_data['close'], self.fast_period, self.slow_period, self.signal_period
                )

                # Логика сигналов
                if macd > signal_line and prev_macd <= prev_signal:
                    signal = Signal.BUY
                    logger.info(f"BUY: {self.name}: MACD пересек сигнальную линию вверх - сигнал BUY")
                elif macd < signal_line and prev_macd >= prev_signal:
                    signal = Signal.SELL
                    logger.info(f"SELL: {self.name}: MACD пересек сигнальную линию вниз - сигнал SELL")
                else:
                    signal = Signal.HOLD
            else:
                signal = Signal.HOLD

            return signal

        except Exception as e:
            logger.error(f"ERROR Ошибка в стратегии MACD: {e}")
            return Signal.HOLD


class VolumePriceStrategy(BaseStrategy):
    """
    Стратегия на основе объема и цены
    Объемная стратегия из рекомендаций T-Bank
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("TBank_Volume_Price", config)
        config = config or {}
        self.volume_multiplier = config.get('volume_multiplier', 2.0)
        self.price_change_threshold = config.get('price_change_threshold', 0.01)  # 1%

    def generate_signal(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """
        Генерация сигнала на основе объема и движения цены
        """
        try:
            if current_idx < 10:  # Нужно достаточно данных
                return Signal.HOLD

            current_data = data.iloc[:current_idx + 1]

            current_price = current_data['close'].iloc[-1]
            current_volume = current_data['volume'].iloc[-1]

            # Средний объем
            avg_volume = current_data['volume'].iloc[-10:].mean()

            # Изменение цены за последний день
            price_change = (current_price - current_data['close'].iloc[-2]) / current_data['close'].iloc[-2]

            # Логика сигналов
            if current_volume > avg_volume * self.volume_multiplier:
                if price_change > self.price_change_threshold:
                    signal = Signal.BUY
                    logger.info(f"BUY: {self.name}: Высокий объем + рост цены - сигнал BUY")
                elif price_change < -self.price_change_threshold:
                    signal = Signal.SELL
                    logger.info(f"SELL: {self.name}: Высокий объем + падение цены - сигнал SELL")
                else:
                    signal = Signal.HOLD
            else:
                signal = Signal.HOLD

            return signal

        except Exception as e:
            logger.error(f"ERROR Ошибка в стратегии Volume-Price: {e}")
            return Signal.HOLD


class TBankStrategyManager:
    """
    Менеджер для управления стратегиями T-Bank
    Интегрируется с существующей архитектурой
    """

    def __init__(self):
        self.strategies = {
            'ma_crossover': MovingAverageCrossoverStrategy(),
            'rsi': RSIStrategy(),
            'bollinger': BollingerBandsStrategy(),
            'macd': MACDStrategy(),
            'volume_price': VolumePriceStrategy()
        }

    def get_strategy(self, strategy_name: str, config: Dict[str, Any] = None):
        """Получить экземпляр стратегии"""
        if strategy_name in self.strategies:
            strategy_class = self.strategies[strategy_name].__class__
            return strategy_class(config)
        else:
            raise ValueError(f"Стратегия '{strategy_name}' не найдена")

    def get_available_strategies(self) -> Dict[str, str]:
        """Получить список доступных стратегий"""
        return {
            'ma_crossover': 'Пересечение скользящих средних',
            'rsi': 'RSI стратегия',
            'bollinger': 'Полосы Боллинджера',
            'macd': 'MACD стратегия',
            'volume_price': 'Объемно-ценовая стратегия'
        }


def create_tbank_robot(config: Dict[str, Any] = None) -> BaseStrategy:
    """
    Создание готового робота на основе T-Bank SDK стратегий
    """
    default_config = {
        'strategy': 'ma_crossover',
        'fast_period': 10,
        'slow_period': 20,
        'rsi_period': 14,
        'overbought': 70,
        'oversold': 30
    }

    if config:
        default_config.update(config)

    manager = TBankStrategyManager()

    return manager.get_strategy(default_config['strategy'], default_config)


# Тестирование стратегий
if __name__ == "__main__":
    import numpy as np

    print("Тестирование готовых стратегий T-Bank SDK...")

    # Создаем тестовые данные
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')

    # Симулируем тренд с волатильностью
    base_price = 100
    prices = []
    volumes = []

    for i in range(100):
        trend = i * 0.1  # Восходящий тренд
        noise = np.random.normal(0, 2)
        price = base_price + trend + noise
        prices.append(price)

        # Объем с случайной вариацией
        volume = np.random.randint(50000, 200000)
        volumes.append(volume)

    test_data = pd.DataFrame({
        'close': prices,
        'volume': volumes
    }, index=dates)

    # Тестируем все стратегии
    manager = TBankStrategyManager()
    strategies_config = {
        'ma_crossover': {},
        'rsi': {'rsi_period': 14, 'overbought': 70, 'oversold': 30},
        'bollinger': {},
        'macd': {},
        'volume_price': {}
    }

    for strategy_name, config in strategies_config.items():
        print(f"\nANALYSIS: Тестирование стратегии: {strategy_name}")

        try:
            strategy = manager.get_strategy(strategy_name, config)
            signals = []

            for i in range(30, len(test_data)):
                signal = strategy.generate_signal(test_data, i)
                if signal != Signal.HOLD:
                    signals.append(signal)
                    strategy.update_position(signal, test_data['close'].iloc[i], test_data.index[i])

            print(f"  OK Сгенерировано сигналов: {len(signals)}")
            print(f"  OK Общее P&L: {strategy.pnl:.2f}")

        except Exception as e:
            print(f"  ERROR Ошибка: {e}")

    print("\nOK Тестирование стратегий завершено!")