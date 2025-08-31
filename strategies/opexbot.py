#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpexBot Strategy Implementation
Options Iron Condor Strategy из репозитория https://github.com/pskucherov/OpexBot
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta

from tbank_strategies import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class IronCondorStrategy(BaseStrategy):
    """
    Options Iron Condor Strategy - адаптация OpexBot

    Логика Iron Condor:
    1. Продажа опционов PUT/Call с отсечкой (верхняя/нижняя точка прибыли)
    2. Покупка защитных PUT/Call для ограничения убытков
    3. Прибыль в диапазоне между отсечками
    4. Максимальный убыток за пределами защитных барьеров
    """

    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        super().__init__("OpexBot Iron_Condor", config)

        # Поддерживаемые базовые активы (российские акции)
        self.underlying_tickers = config.get('underlying_tickers',
                                            ['SBERF', 'Si', 'GAZPF'])

        # Параметры сделки
        self.strike_offset = config.get('strike_offset', 0.05)  # 5% от цены
        self.expiry_days = config.get('expiry_days', 7)  # 7 дней до экспирации
        self.buy_back_threshold = config.get('buy_back_threshold', 0.5)  # 50% прибыли

        # Управление рисками
        self.max_positions = config.get('max_positions', 5)
        self.risk_per_trade = config.get('risk_per_trade', 0.02)  # 2% на сделку

        # Состояние стратегии
        self.open_positions = []  # Список открытых Iron Condor позиций
        self.available_balance = 100000  # Доступный капитал

        logger.info(f"OpexBot Iron Condor initialized for: {self.underlying_tickers}")
        logger.info(f"Parameters: Strike offset {self.strike_offset*100}%, Expiry {self.expiry_days}d")


class IronCondorPosition:
    """
    Класс для одной Iron Condor позиции
    """

    def __init__(self, underlying: str, current_price: float, expiry_date: datetime,
                 strike_range: Dict[str, float], premium_received: float, config: Dict[str, Any] = None):
        self.underlying = underlying
        self.entry_price = current_price
        self.expiry_date = expiry_date
        self.created_time = datetime.now()

        # Страйки (sell strikes)
        self.lower_sell_strike = strike_range['lower_sell']
        self.upper_sell_strike = strike_range['upper_sell']

        # Защитные страйки (buy strikes)
        self.lower_buy_strike = strike_range['lower_buy']
        self.upper_buy_strike = strike_range['upper_buy']

        # Премии и стоимости
        self.premium_received = premium_received
        self.max_profit = premium_received
        self.max_loss = (
            (self.lower_buy_strike - self.lower_sell_strike) - premium_received +
            (self.upper_sell_strike - self.upper_buy_strike) - premium_received
        )

        # Статус позиции
        self.is_open = True
        self.pnl = 0
        self.current_value = premium_received

    def update_value(self, current_price: float, days_to_expiry: int) -> float:
        """Оценка текущей стоимости позиции"""
        if not self.is_open:
            return self.pnl

        # В реальной торговле здесь был бы расчет Грек + оценка опционов
        # Для симуляции используем упрощенную модель

        days_passed = (datetime.now() - self.created_time).days
        time_decay_factor = 1 - (days_passed / self.expiry_date - self.created_time).days

        # Оценка позиций по страйкам
        lower_put_value = max(self.lower_sell_strike - current_price, 0)
        upper_call_value = max(current_price - self.upper_sell_strike, 0)

        total_liability = lower_put_value + upper_call_value
        current_value = self.premium_received - total_liability

        self.current_value = current_value
        self.pnl = current_value - self.premium_received

        return self.pnl

    def close_if_profitable(self, current_price: float, threshold: float = 0.5) -> bool:
        """Закрытие позиции если достигла определенной прибыли"""
        profit_ratio = self.pnl / self.premium_received if self.premium_received > 0 else 0

        if profit_ratio > threshold:
            self.is_open = False
            return True

        return False

    def is_expired(self) -> bool:
        """Проверка на истечение позиции"""
        return datetime.now() >= self.expiry_date

    def close_on_expiry(self, current_price: float):
        """Закрытие позиции по экспирации"""
        self.update_value(current_price, 0)
        self.is_open = False


def calculate_option_premium(current_price: float, strike: float,
                           time_to_expiry: float, volatility: float = 0.2) -> float:
    """
    Упрощенная оценка премии опциона (Black-Scholes approximation)
    """
    try:
        # Время в годах
        t = time_to_expiry / 365

        if t <= 0:
            return 0.0

        # Простая модель премии
        moneyness = abs(strike - current_price) / current_price
        intrinsic_value = max(current_price - strike if strike <= current_price else 0, 0)

        # Временная стоимость
        time_value = (volatility * np.sqrt(t) * current_price * 0.2) / np.sqrt(365 / time_to_expiry)

        return intrinsic_value + time_value

    except Exception:
        return 0.1  # Минимальная премия


def generate_iron_condor_strikes(current_price: float, offset: float = 0.05,
                               width: float = 0.05) -> Dict[str, float]:
    """
    Генерация страйков для Iron Condor позиции
    """
    center = current_price

    # Sell strikes (где продаем опционы)
    lower_sell_strike = center * (1 - offset)
    upper_sell_strike = center * (1 + offset)

    # Buy strikes (где покупаем защиту)
    lower_buy_strike = center * (1 - offset - width)
    upper_buy_strike = center * (1 + offset + width)

    return {
        'lower_sell': lower_sell_strike,
        'upper_sell': upper_sell_strike,
        'lower_buy': lower_buy_strike,
        'upper_buy': upper_buy_strike
    }


def simulate_option_prices(strikes: Dict[str, float], current_price: float,
                          expiry_days: int, volatility: float = 0.2) -> Dict[str, float]:
    """
    Симуляция цен опционов
    """
    call_prices = {}
    put_prices = {}

    for strike_name, strike_price in strikes.items():
        if 'lower' in strike_name:
            # PUT опционы для нижних страйков
            if strike_name == 'lower_sell':
                put_prices['sell'] = calculate_option_premium(
                    current_price, strike_price, expiry_days, volatility)
            elif strike_name == 'lower_buy':
                put_prices['buy'] = calculate_option_premium(
                    current_price, strike_price, expiry_days, volatility)

        elif 'upper' in strike_name:
            # CALL опционы для верхних страйков
            if strike_name == 'upper_sell':
                call_prices['sell'] = calculate_option_premium(
                    current_price, strike_price, expiry_days, volatility)
            elif strike_name == 'upper_buy':
                call_prices['buy'] = calculate_option_premium(
                    current_price, strike_price, expiry_days, volatility)

    return {
        'calls': call_prices,
        'puts': put_prices,
        'net_premium': (call_prices.get('sell', 0) + put_prices.get('sell', 0)
                       - call_prices.get('buy', 0) - put_prices.get('buy', 0))
    }


# Основная стратегия Iron Condor
class IronCondorStrategy(BaseStrategy):
    """
    Подробная реализация Iron Condor стратегии
    """

    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        super().__init__("OpexBot Iron_Condor", config)

        # Формируется базовый актив
        self.underlying_tickers = config.get('underlying_tickers', ['SBERF', 'Si', 'GAZPF'])
        self.strike_offset = config.get('strike_offset', 0.05)  # 5% от цены
        self.expiry_days = config.get('expiry_days', 7)  # Неделя до экспирации
        self.buy_back_threshold = config.get('buy_back_threshold', 0.5)  # 50% прибыли
        self.volatility_adjustment = config.get('volatility_adjustment', True)

        # Состояние стратегии
        self.open_positions = []
        self.total_capital = 100000
        self.available_margin = 50000
        self.position_counter = 0

        logger.info(f"Iron Condor initialized with: {len(self.underlying_tickers)} underlyings")
        logger.info(f"Parameters: offset={self.strike_offset*100}%, expiry={self.expiry_days}d")

    def generate_signal(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """
        Генерация сигнала для открытия/закрытия Iron Condor позиций
        """

        try:
            if current_idx < 30:  # Нужно достаточно данных
                return Signal.HOLD

            current_data = data.iloc[:current_idx + 1]
            current_price = current_data['close'].iloc[-1]

            # Управление существующими позициями
            self._update_positions(current_price)

            # Проверка возможности открытия новой позиции
            can_open_position = (
                len(self.open_positions) < 3  # Максимум 3 позиции
                and self._calculate_available_margin() > 0.1 * self.total_capital
            )

            # Условия для открытия Iron Condor:
            # 1. Низкая волатильность (хорошее время для продаж опционов)
            # 2. Цена в среднем диапазоне (достаточно от цены до страков)
            # 3. Достаточно времени до экспирации
            # 4. Прибыльные условия

            if can_open_position and self._should_open_position(current_data):
                # Генерация параметров позиции
                strike_range = generate_iron_condor_strikes(current_price, self.strike_offset)
                expiry_date = datetime.now() + timedelta(days=self.expiry_days)

                # Симуляция вычисления премий
                option_prices = simulate_option_prices(
                    strike_range, current_price, self.expiry_days)

                net_premium = option_prices['net_premium']

                if net_premium > 0:  # Только выгодные позиции
                    position = IronCondorPosition(
                        underlying=self._get_best_underlying(current_data),
                        current_price=current_price,
                        expiry_date=expiry_date,
                        strike_range=strike_range,
                        premium_received=net_premium
                    )

                    self.open_positions.append(position)
                    self.position_counter += 1

                    logger.info(".2f")
                    logger.info(f"  Position #{self.position_counter}: Strikes {strike_range['lower_sell']:.1f}/{strike_range['upper_sell']:.1f}")
                    logger.info(".2f")

                    self.signals_history.append({
                        'timestamp': current_data.index[-1],
                        'signal': Signal.BUY,  # BUY = открытие позиции
                        'price': current_price,
                        'position_id': self.position_counter,
                        'premium': net_premium,
                        'strikes': strike_range
                    })

                    return Signal.BUY

            return Signal.HOLD

        except Exception as e:
            logger.error(f"OpexBot Iron Condor error: {e}")
            return Signal.HOLD

    def _update_positions(self, current_price: float):
        """Обновление состояния всех позиций"""
        positions_to_close = []

        for i, position in enumerate(self.open_positions):
            try:
                # Обновление стоимости позиции
                position.update_value(current_price,
                                    (position.expiry_date - datetime.now()).days)

                # Закрытие по тайм-декэю или профиту
                if position.close_if_profitable(current_price, self.buy_back_threshold):
                    positions_to_close.append((i, 'take_profit', position.pnl))
                elif position.is_expired():
                    position.close_on_expiry(current_price)
                    positions_to_close.append((i, 'expiry', position.pnl))
                elif position.pnl < -0.8 * position.premium_received:  # Стоп-лосс
                    positions_to_close.append((i, 'stop_loss', position.pnl))

            except Exception as e:
                logger.error(f"Position update error: {e}")
                positions_to_close.append((i, 'error', 0))

        # Закрытие позиций
        for pos_idx, reason, pnl in sorted(positions_to_close, reverse=True):
            closed_position = self.open_positions.pop(pos_idx)
            logger.info(".2f")
            logger.info(".2f")

    def _should_open_position(self, data: pd.DataFrame) -> bool:
        """Проверка условий для открытия новой позиции"""
        try:
            # Анализ волатильности (низкая волатильность нужна для продажи опционов)
            recent_volatility = data['close'].pct_change().std() * np.sqrt(252)  # Годовая волатильность

            # Анализ тренда (избегать сильных трендов)
            sma_20 = data['close'].rolling(20).mean().iloc[-1]
            sma_50 = data['close'].rolling(50).mean().iloc[-1]
            current_price = data['close'].iloc[-1]

            trend_strength = abs(current_price - (sma_20 + sma_50) / 2) / ((sma_20 + sma_50) / 2)

            # Условия открытия:
            # 1. Волатильность умеренная (не слишком высокая)
            low_volatility = recent_volatility < 0.3
            # 2. Цена в боковике (не сильный тренд)
            sideway_trend = trend_strength < 0.05
            # 3. Достаточное расстояние до последних позиций
            time_since_last_position = self._time_since_last_position()

            return low_volatility and sideway_trend and time_since_last_position > timedelta(days=1)

        except Exception:
            return False

    def _time_since_last_position(self) -> timedelta:
        """Время с момента открытия последней позиции"""
        if not self.open_positions:
            return timedelta(days=365)  # Большой интервал если нет позиций

        last_position = max(self.open_positions, key=lambda p: p.created_time)
        return datetime.now() - last_position.created_time

    def _get_best_underlying(self, data: pd.DataFrame) -> str:
        """Выбор наилучшего базового актива"""
        # Простая логика - каждый раз чередовать
        tickers = self.underlying_tickers
        return tickers[self.position_counter % len(tickers)]

    def _calculate_available_margin(self) -> float:
        """Расчет доступного маржи"""
        used_margin = sum(
            position.max_loss for position in self.open_positions
            if position.is_open
        )
        return max(0, self.available_margin - used_margin)


# Регистрация стратегии OpexBot
def _register_opexbot_strategies():
    """Регистрация стратегий OpexBot"""
    try:
        from strategies.main import register_strategy

        # Основная Iron Condor стратегия
        register_strategy(
            'opexbot_ironcondor',
            IronCondorStrategy,
            {
                'description': 'Options Iron Condor Strategy from OpexBot',
                'underlying_tickers': ['SBERF', 'Si', 'GAZPF'],
                'strike_offset': 0.05,
                'expiry_days': 7,
                'buy_back_threshold': 0.5,
                'max_positions': 5,
                'risk_per_trade': 0.02,
                'source': 'https://github.com/pskucherov/OpexBot'
            }
        )

        logger.info("OpexBot strategies registered successfully")

    except Exception as e:
        logger.error(f"Failed to register OpexBot strategies: {e}")


# Автоматическая регистрация
_register_opexbot_strategies()