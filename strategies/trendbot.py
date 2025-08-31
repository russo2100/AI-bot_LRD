#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python TrendBot Strategy Implementation
SMA Cross RSI Strategy из репозитория https://github.com/qwertyo1/tinkoff-trading-bot
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

from tbank_strategies import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class SMACrossRSIStrategy(BaseStrategy):
    """
    SMA Cross RSI Strategy - комбинация двух индикаторов

    Логика стратегии:
    - Пересечение скользящих средних для тренда
    - RSI для подтверждения входа на перепроданность/перекупленность
    - Комбинированные сигналы для более точного входа
    """

    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        super().__init__("TrendBot SMA_RSI", config)

        # Параметры SMA
        self.fast_sma_period = config.get('fast_sma', 50)
        self.slow_sma_period = config.get('slow_sma', 200)

        # Параметры RSI
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_threshold = config.get('rsi_threshold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.rsi_oversold = config.get('rsi_oversold', 30)

        logger.info(f"TrendBot SMA RSI initialized with: SMA({self.fast_sma_period},{self.slow_sma_period}), RSI({self.rsi_period})")

    def _calculate_rsi(self, prices: pd.Series, period: int) -> float:
        """Расчет RSI индикатора"""
        if len(prices) < period + 1:
            return 50.0

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs.iloc[-1]))

    def generate_signal(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """
        Генерация сигнала на основе комбинации SMA Cross + RSI

        Стратегия Python TrendBot:
        1. Вход BUY: Быстрая MA пересекает медленную вверх И RSI < threshold
        2. Выход SELL: Быстрая MA пересекает медленную вниз ИЛИ RSI > overbought
        """

        try:
            # Проверяем наличие необходимых данных
            if current_idx < self.slow_sma_period + self.rsi_period:
                return Signal.HOLD

            # Данные для анализа
            current_data = data.iloc[:current_idx + 1]
            current_price = current_data['close'].iloc[-1]

            # === Расчет SMA ===
            fast_sma_ser = current_data['close'].rolling(self.fast_sma_period).mean()
            slow_sma_ser = current_data['close'].rolling(self.slow_sma_period).mean()

            fast_sma = fast_sma_ser.iloc[-1]
            slow_sma = slow_sma_ser.iloc[-1]

            # Предыдущие значения для определения пересечения
            if len(fast_sma_ser) >= 2:
                prev_fast_sma = fast_sma_ser.iloc[-2]
                prev_slow_sma = slow_sma_ser.iloc[-2]
            else:
                prev_fast_sma = prev_slow_sma = fast_sma

            # === Расчет RSI ===
            rsi = self._calculate_rsi(current_data['close'], self.rsi_period)

            # === Логика сигналов (Python TrendBot approach) ===
            signal = Signal.HOLD

            # BUY Signal
            # 1. Быстрая MA пересекает медленную вверх (тренд вверх)
            ma_crossover_up = fast_sma > slow_sma and prev_fast_sma <= prev_slow_sma
            # 2. RSI показывает перепроданность (дополнительное условие)
            rsi_oversold = rsi <= self.rsi_oversold

            if ma_crossover_up and rsi_oversold:
                signal = Signal.BUY
                logger.info(f"TREND UP + RSI <= {self.rsi_oversold}: BUY signal at price {current_price:.2f}")
                self.signals_history.append({
                    'timestamp': current_data.index[-1],
                    'signal': signal,
                    'price': current_price,
                    'fast_sma': fast_sma,
                    'slow_sma': slow_sma,
                    'rsi': rsi,
                    'reason': 'MA crossover up + RSI oversold'
                })
            # SELL Signal
            # 1. Быстрая MA пересекает медленную вниз ИЛИ
            # 2. RSI показывает перекупленность
            elif (fast_sma < slow_sma and prev_fast_sma >= prev_slow_sma) or rsi >= self.rsi_overbought:
                signal = Signal.SELL
                reason = "MA crossover down" if (fast_sma < slow_sma and prev_fast_sma >= prev_slow_sma) else f"RSI >= {self.rsi_overbought}"
                logger.info(f"{reason}: SELL signal at price {current_price:.2f}, RSI={rsi:.1f}")
                self.signals_history.append({
                    'timestamp': current_data.index[-1],
                    'signal': signal,
                    'price': current_price,
                    'fast_sma': fast_sma,
                    'slow_sma': slow_sma,
                    'rsi': rsi,
                    'reason': reason
                })

            return signal

        except Exception as e:
            logger.error(f"TrendBot SMA RSI error: {e}")
            return Signal.HOLD


class TrendBotEnhanced(BaseStrategy):
    """
    Расширенная версия TrendBot с дополнительными фильтрами
    """

    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        super().__init__("TrendBot Enhanced", config)

        # Основные параметры
        self.fast_sma = config.get('fast_sma', 20)
        self.slow_sma = config.get('slow_sma', 50)
        self.rsi_period = config.get('rsi_period', 14)
        self.macd_fast = config.get('macd_fast', 12)
        self.macd_slow = config.get('macd_slow', 26)
        self.macd_signal = config.get('macd_signal', 9)

        # Пороговые значения
        self.rsi_buy_threshold = config.get('rsi_buy_threshold', 30)
        self.rsi_sell_threshold = config.get('rsi_sell_threshold', 70)
        self.confirmation_period = config.get('confirmation_period', 3)

        logger.info(f"TrendBot Enhanced initialized with multiple indicators")

    def generate_signal(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """
        Улучшенная логика с MACD подтверждением
        """
        try:
            if current_idx < max(self.slow_sma, self.rsi_period, self.macd_slow, self.confirmation_period):
                return Signal.HOLD

            current_data = data.iloc[:current_idx + 1]

            # Расчет индикаторов
            fast_ema = current_data['close'].ewm(span=self.macd_fast).mean()
            slow_ema = current_data['close'].ewm(span=self.macd_slow).mean()
            macd_line = fast_ema - slow_ema
            signal_line = macd_line.ewm(span=self.macd_signal).mean()

            rsi = self._calculate_rsi(current_data['close'], self.rsi_period)

            # Простые фильтры
            sma_20 = current_data['close'].rolling(20).mean().iloc[-1]
            sma_50 = current_data['close'].rolling(50).mean().iloc[-1]
            price = current_data['close'].iloc[-1]

            # Многократное подтверждение сигнала
            macd_bullish = macd_line.iloc[-1] > signal_line.iloc[-1]
            rsi_bullish = rsi <= self.rsi_buy_threshold
            sma_bullish = price > sma_20 > sma_50

            if macd_bullish and rsi_bullish and sma_bullish:
                return Signal.BUY

            return Signal.HOLD

        except Exception as e:
            logger.error(f"TrendBot Enhanced error: {e}")
            return Signal.HOLD

    def _calculate_rsi(self, prices: pd.Series, period: int) -> float:
        """Вспомогательный метод для RSI"""
        if len(prices) < period + 1:
            return 50.0

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs.iloc[-1]))


# Регистрация стратегий при импорте
def _register_trendbot_strategies():
    """Регистрация стратегий TrendBot"""
    try:
        from strategies.main import register_strategy

        # Регистрация основной стратегии SMA RSI
        register_strategy(
            'trendbot_sma_rsi',
            SMACrossRSIStrategy,
            {
                'description': 'SMA Cross RSI from TrendBot',
                'fast_sma': 50,
                'slow_sma': 200,
                'rsi_period': 14,
                'rsi_threshold': 30,
                'source': 'https://github.com/qwertyo1/tinkoff-trading-bot'
            }
        )

        # Регистрация расширенной стратегии
        register_strategy(
            'trendbot_enhanced',
            TrendBotEnhanced,
            {
                'description': 'Enhanced TrendBot with MACD',
                'fast_sma': 20,
                'slow_sma': 50,
                'rsi_period': 14,
                'source': 'https://github.com/qwertyo1/tinkoff-trading-bot'
            }
        )

        logger.info("TrendBot strategies registered successfully")

    except Exception as e:
        logger.error(f"Failed to register TrendBot strategies: {e}")


# Автоматическая регистрация при импорте
_register_trendbot_strategies()