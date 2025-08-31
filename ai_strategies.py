# src/strategies/ai_strategies.py
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from abc import ABC, abstractmethod
from enum import Enum
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class Signal(Enum):
    """Торговые сигналы"""
    BUY = 1
    SELL = -1
    HOLD = 0

class Position(Enum):
    """Позиции"""
    LONG = 1
    SHORT = -1
    FLAT = 0

class BaseStrategy(ABC):
    """
    Базовый класс для торговых стратегий
    Реализует подходы из книги "ИИ в трейдинге"
    """

    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.position = Position.FLAT
        self.entry_price = 0.0
        self.entry_time = None
        self.pnl = 0.0
        self.trades = []
        self.signals_history = []

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """Генерация торгового сигнала"""
        pass

    def update_position(self, signal: Signal, price: float, timestamp: datetime):
        """Обновление позиции на основе сигнала"""
        if signal == Signal.BUY and self.position != Position.LONG:
            if self.position == Position.SHORT:
                self._close_position(price, timestamp)
            self._open_long_position(price, timestamp)

        elif signal == Signal.SELL and self.position != Position.SHORT:
            if self.position == Position.LONG:
                self._close_position(price, timestamp)
            self._open_short_position(price, timestamp)

        elif signal == Signal.HOLD and self.position != Position.FLAT:
            # Опционально: закрытие позиции при сигнале HOLD
            pass

    def _open_long_position(self, price: float, timestamp: datetime):
        """Открытие длинной позиции"""
        self.position = Position.LONG
        self.entry_price = price
        self.entry_time = timestamp

        trade = {
            'action': 'OPEN_LONG',
            'price': price,
            'timestamp': timestamp,
            'position_size': self.config.get('position_size', 1.0)
        }
        self.trades.append(trade)

        logger.info(f"🟢 {self.name}: Открыта LONG позиция по {price:.4f}")

    def _open_short_position(self, price: float, timestamp: datetime):
        """Открытие короткой позиции"""
        self.position = Position.SHORT
        self.entry_price = price
        self.entry_time = timestamp

        trade = {
            'action': 'OPEN_SHORT',
            'price': price,
            'timestamp': timestamp,
            'position_size': self.config.get('position_size', 1.0)
        }
        self.trades.append(trade)

        logger.info(f"🔴 {self.name}: Открыта SHORT позиция по {price:.4f}")

    def _close_position(self, price: float, timestamp: datetime):
        """Закрытие текущей позиции"""
        if self.position == Position.FLAT:
            return

        # Расчет P&L
        if self.position == Position.LONG:
            pnl = price - self.entry_price
        else:  # SHORT
            pnl = self.entry_price - price

        position_size = self.config.get('position_size', 1.0)
        pnl *= position_size
        self.pnl += pnl

        trade = {
            'action': 'CLOSE',
            'price': price,
            'timestamp': timestamp,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time,
            'pnl': pnl,
            'position': self.position.name
        }
        self.trades.append(trade)

        logger.info(f"⚪ {self.name}: Закрыта позиция. P&L: {pnl:.4f}, Общий P&L: {self.pnl:.4f}")

        self.position = Position.FLAT
        self.entry_price = 0.0
        self.entry_time = None

    def get_performance_metrics(self) -> Dict[str, float]:
        """Расчет метрик производительности стратегии"""
        if not self.trades:
            return {}

        closed_trades = [t for t in self.trades if 'pnl' in t]

        if not closed_trades:
            return {'total_trades': len(self.trades), 'total_pnl': self.pnl}

        pnls = [t['pnl'] for t in closed_trades]
        win_trades = [pnl for pnl in pnls if pnl > 0]
        loss_trades = [pnl for pnl in pnls if pnl < 0]

        metrics = {
            'total_trades': len(closed_trades),
            'total_pnl': sum(pnls),
            'win_rate': len(win_trades) / len(closed_trades) if closed_trades else 0,
            'avg_win': np.mean(win_trades) if win_trades else 0,
            'avg_loss': np.mean(loss_trades) if loss_trades else 0,
            'profit_factor': abs(sum(win_trades) / sum(loss_trades)) if loss_trades else float('inf') if win_trades else 0,
            'max_win': max(pnls) if pnls else 0,
            'max_loss': min(pnls) if pnls else 0
        }

        return metrics


class AIModelStrategy(BaseStrategy):
    """
    Стратегия на основе прогнозов ИИ моделей
    """

    def __init__(self, models: Dict[str, Any], config: Dict[str, Any] = None):
        super().__init__("AI_Model_Strategy", config)
        self.models = models  # Словарь с обученными моделями
        self.prediction_threshold = config.get('prediction_threshold', 0.005)
        self.confidence_threshold = config.get('confidence_threshold', 0.6)

    def generate_signal(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """
        Генерация сигнала на основе ансамбля ИИ моделей
        """
        try:
            current_data = data.iloc[:current_idx + 1]

            predictions = {}
            confidences = {}

            # Получаем прогнозы от всех моделей
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'predict'):
                        # Модели прогнозирования цен
                        pred = model.predict(current_data.tail(1))
                        current_price = current_data['close'].iloc[-1]

                        # Преобразуем в процентное изменение
                        price_change = (pred - current_price) / current_price
                        predictions[model_name] = price_change
                        confidences[model_name] = 1.0  # Базовая уверенность

                    elif hasattr(model, 'predict_single'):
                        # Классификаторы
                        features = {
                            col: current_data[col].iloc[-1] 
                            for col in current_data.columns 
                            if col in model.feature_columns
                        }

                        result = model.predict_single(features)
                        prediction = result['prediction']
                        confidence = result['confidence']

                        # Преобразуем в цифровое значение
                        if prediction == 1:  # UP
                            predictions[model_name] = self.prediction_threshold * 1.5
                        elif prediction == -1:  # DOWN
                            predictions[model_name] = -self.prediction_threshold * 1.5
                        else:  # FLAT
                            predictions[model_name] = 0

                        confidences[model_name] = confidence

                except Exception as e:
                    logger.warning(f"⚠️ Ошибка модели {model_name}: {e}")
                    continue

            if not predictions:
                return Signal.HOLD

            # Взвешенное усреднение прогнозов
            weighted_prediction = 0
            total_weight = 0

            for model_name, pred in predictions.items():
                weight = confidences.get(model_name, 0.5)
                weighted_prediction += pred * weight
                total_weight += weight

            if total_weight > 0:
                final_prediction = weighted_prediction / total_weight
            else:
                final_prediction = 0

            # Средняя уверенность
            avg_confidence = np.mean(list(confidences.values())) if confidences else 0

            # Генерация сигнала
            signal = Signal.HOLD

            if avg_confidence >= self.confidence_threshold:
                if final_prediction > self.prediction_threshold:
                    signal = Signal.BUY
                elif final_prediction < -self.prediction_threshold:
                    signal = Signal.SELL

            # Сохраняем историю сигналов
            signal_info = {
                'timestamp': current_data.index[-1],
                'signal': signal,
                'prediction': final_prediction,
                'confidence': avg_confidence,
                'models_used': len(predictions)
            }
            self.signals_history.append(signal_info)

            return signal

        except Exception as e:
            logger.error(f"❌ Ошибка генерации сигнала: {e}")
            return Signal.HOLD


class TechnicalStrategy(BaseStrategy):
    """
    Стратегия на основе технических индикаторов
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Technical_Strategy", config)
        self.rsi_buy_threshold = config.get('rsi_buy_threshold', 30)
        self.rsi_sell_threshold = config.get('rsi_sell_threshold', 70)
        self.ma_fast_period = config.get('ma_fast_period', 10)
        self.ma_slow_period = config.get('ma_slow_period', 20)

    def generate_signal(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """
        Генерация сигнала на основе технических индикаторов
        """
        try:
            if current_idx < self.ma_slow_period:
                return Signal.HOLD

            current_data = data.iloc[:current_idx + 1]

            # Получаем последние значения индикаторов
            current_rsi = current_data.get('rsi', pd.Series([50])).iloc[-1]
            current_price = current_data['close'].iloc[-1]

            # Скользящие средние
            ma_fast = current_data['close'].rolling(self.ma_fast_period).mean().iloc[-1]
            ma_slow = current_data['close'].rolling(self.ma_slow_period).mean().iloc[-1]

            # Предыдущие значения для определения пересечений
            prev_ma_fast = current_data['close'].rolling(self.ma_fast_period).mean().iloc[-2]
            prev_ma_slow = current_data['close'].rolling(self.ma_slow_period).mean().iloc[-2]

            # Логика сигналов
            signal = Signal.HOLD

            # Пересечение MA сверху вниз + RSI oversold
            if (ma_fast > ma_slow and prev_ma_fast <= prev_ma_slow and 
                current_rsi <= self.rsi_buy_threshold):
                signal = Signal.BUY

            # Пересечение MA снизу вверх + RSI overbought
            elif (ma_fast < ma_slow and prev_ma_fast >= prev_ma_slow and 
                  current_rsi >= self.rsi_sell_threshold):
                signal = Signal.SELL

            return signal

        except Exception as e:
            logger.error(f"❌ Ошибка технической стратегии: {e}")
            return Signal.HOLD


class HybridStrategy(BaseStrategy):
    """
    Гибридная стратегия: комбинация ИИ и технического анализа
    """

    def __init__(self, ai_models: Dict[str, Any], config: Dict[str, Any] = None):
        super().__init__("Hybrid_Strategy", config)
        self.ai_strategy = AIModelStrategy(ai_models, config)
        self.technical_strategy = TechnicalStrategy(config)
        self.ai_weight = config.get('ai_weight', 0.7)
        self.technical_weight = config.get('technical_weight', 0.3)

    def generate_signal(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """
        Комбинированная генерация сигнала
        """
        try:
            # Получаем сигналы от обеих стратегий
            ai_signal = self.ai_strategy.generate_signal(data, current_idx)
            technical_signal = self.technical_strategy.generate_signal(data, current_idx)

            # Взвешенное голосование
            ai_vote = ai_signal.value * self.ai_weight
            technical_vote = technical_signal.value * self.technical_weight

            combined_vote = ai_vote + technical_vote

            # Определяем финальный сигнал
            if combined_vote > 0.5:
                signal = Signal.BUY
            elif combined_vote < -0.5:
                signal = Signal.SELL
            else:
                signal = Signal.HOLD

            # Сохраняем детали для анализа
            signal_info = {
                'timestamp': data.index[current_idx],
                'signal': signal,
                'ai_signal': ai_signal,
                'technical_signal': technical_signal,
                'combined_vote': combined_vote
            }
            self.signals_history.append(signal_info)

            return signal

        except Exception as e:
            logger.error(f"❌ Ошибка гибридной стратегии: {e}")
            return Signal.HOLD


class RiskManager:
    """
    Модуль управления рисками
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.max_drawdown = config.get('max_drawdown', 0.1)  # 10%
        self.position_size_limit = config.get('position_size_limit', 0.05)  # 5% капитала
        self.stop_loss_pct = config.get('stop_loss_pct', 0.02)  # 2%
        self.take_profit_pct = config.get('take_profit_pct', 0.04)  # 4%

        self.initial_capital = config.get('initial_capital', 100000)
        self.current_capital = self.initial_capital
        self.max_capital = self.initial_capital

    def should_limit_position(self, current_pnl: float) -> bool:
        """Проверка на ограничение позиций из-за просадки"""
        self.current_capital = self.initial_capital + current_pnl

        if self.current_capital > self.max_capital:
            self.max_capital = self.current_capital

        current_drawdown = (self.max_capital - self.current_capital) / self.max_capital

        return current_drawdown >= self.max_drawdown

    def calculate_position_size(self, price: float, volatility: float = 0.02) -> float:
        """Расчет размера позиции на основе волатильности"""
        # Kelly Criterion упрощенная версия
        risk_per_trade = self.current_capital * 0.01  # 1% риска на сделку
        stop_loss_amount = price * self.stop_loss_pct

        if stop_loss_amount > 0:
            position_size = risk_per_trade / stop_loss_amount

            # Ограничиваем размер позиции
            max_position = self.current_capital * self.position_size_limit / price
            position_size = min(position_size, max_position)
        else:
            position_size = self.current_capital * 0.01 / price

        return max(position_size, 0)

    def check_stop_loss(self, entry_price: float, current_price: float, position: Position) -> bool:
        """Проверка стоп-лосса"""
        if position == Position.LONG:
            return current_price <= entry_price * (1 - self.stop_loss_pct)
        elif position == Position.SHORT:
            return current_price >= entry_price * (1 + self.stop_loss_pct)

        return False

    def check_take_profit(self, entry_price: float, current_price: float, position: Position) -> bool:
        """Проверка тейк-профита"""
        if position == Position.LONG:
            return current_price >= entry_price * (1 + self.take_profit_pct)
        elif position == Position.SHORT:
            return current_price <= entry_price * (1 - self.take_profit_pct)

        return False


class StrategyManager:
    """
    Менеджер для управления несколькими стратегиями
    """

    def __init__(self):
        self.strategies = {}
        self.risk_manager = None

    def add_strategy(self, strategy: BaseStrategy, weight: float = 1.0):
        """Добавление стратегии в портфель"""
        self.strategies[strategy.name] = {
            'strategy': strategy,
            'weight': weight,
            'enabled': True
        }

    def set_risk_manager(self, risk_manager: RiskManager):
        """Установка менеджера рисков"""
        self.risk_manager = risk_manager

    def generate_combined_signal(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """
        Генерация комбинированного сигнала от всех стратегий
        """
        if not self.strategies:
            return Signal.HOLD

        signals = {}
        total_weight = 0

        for name, strategy_info in self.strategies.items():
            if not strategy_info['enabled']:
                continue

            try:
                strategy = strategy_info['strategy']
                weight = strategy_info['weight']

                signal = strategy.generate_signal(data, current_idx)
                signals[name] = signal.value * weight
                total_weight += weight

            except Exception as e:
                logger.error(f"❌ Ошибка стратегии {name}: {e}")
                continue

        if total_weight == 0:
            return Signal.HOLD

        # Взвешенное голосование
        combined_signal = sum(signals.values()) / total_weight

        if combined_signal > 0.3:
            return Signal.BUY
        elif combined_signal < -0.3:
            return Signal.SELL
        else:
            return Signal.HOLD

    def get_portfolio_performance(self) -> Dict[str, Any]:
        """Получение общей производительности портфеля стратегий"""
        total_pnl = 0
        total_trades = 0
        strategy_metrics = {}

        for name, strategy_info in self.strategies.items():
            strategy = strategy_info['strategy']
            metrics = strategy.get_performance_metrics()
            strategy_metrics[name] = metrics

            total_pnl += metrics.get('total_pnl', 0)
            total_trades += metrics.get('total_trades', 0)

        return {
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'strategies': strategy_metrics
        }


if __name__ == "__main__":
    # Тестирование стратегий
    print("🧪 Тестирование торговых стратегий")

    # Создаем тестовые данные
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='D')

    prices = 100 + np.random.randn(200).cumsum() * 2
    test_data = pd.DataFrame({
        'close': prices,
        'volume': np.random.randint(1000, 10000, 200),
        'rsi': np.random.uniform(20, 80, 200),
        'macd': np.random.randn(200) * 0.5,
    }, index=dates)

    print(f"📊 Тестовые данные: {len(test_data)} записей")

    # Тестирование технической стратегии
    print("\n📈 Тестирование технической стратегии...")
    tech_strategy = TechnicalStrategy({
        'rsi_buy_threshold': 30,
        'rsi_sell_threshold': 70
    })

    signals = []
    for i in range(20, len(test_data)):
        signal = tech_strategy.generate_signal(test_data, i)
        if signal != Signal.HOLD:
            tech_strategy.update_position(signal, test_data['close'].iloc[i], test_data.index[i])
            signals.append((test_data.index[i], signal, test_data['close'].iloc[i]))

    # Закрываем последнюю позицию если есть
    if tech_strategy.position != Position.FLAT:
        tech_strategy._close_position(test_data['close'].iloc[-1], test_data.index[-1])

    tech_metrics = tech_strategy.get_performance_metrics()
    print(f"✅ Технические сигналы: {len(signals)}")
    print(f"📊 P&L: {tech_metrics.get('total_pnl', 0):.2f}")
    print(f"📊 Сделки: {tech_metrics.get('total_trades', 0)}")
    print(f"📊 Win Rate: {tech_metrics.get('win_rate', 0)*100:.1f}%")

    # Тестирование риск-менеджера
    print("\n⚠️ Тестирование риск-менеджера...")
    risk_manager = RiskManager({
        'max_drawdown': 0.1,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.04
    })

    # Проверка стоп-лосса
    entry_price = 100
    current_price = 97
    should_stop = risk_manager.check_stop_loss(entry_price, current_price, Position.LONG)
    print(f"🛑 Стоп-лосс сработал: {should_stop}")

    # Расчет размера позиции
    position_size = risk_manager.calculate_position_size(100, 0.02)
    print(f"💰 Размер позиции: {position_size:.2f}")

    print("✅ Тестирование стратегий завершено!")
