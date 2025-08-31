#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroInvest Strategy Implementation
LSTM Trend Following Strategy из репозитория https://github.com/VladimirTalyzin/NeuroInvest
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

from tbank_strategies import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class SimpleLSTM:
    """
    Упрощенная LSTM реализация для торговли
    Вместо сложной нейронной сети использует комбинацию линейной регрессии и трендовых индикаторов
    """

    def __init__(self, lookback_window: int = 60, forecast_period: int = 5):
        self.lookback_window = lookback_window
        self.forecast_period = forecast_period
        self.model = LinearRegression()
        self.scaler = MinMaxScaler()
        self.trained = False

    def fit(self, data: pd.DataFrame, target_col: str = 'close'):
        """Обучение модели"""
        if len(data) < self.lookback_window + self.forecast_period:
            logger.warning("Not enough data for LSTM training")
            return False

        try:
            # Простая подготовка данных
            features = data.copy()
            if 'close' in features.columns:
                # Создание лаговых признаков (идея исходной LSTM)
                for i in range(1, min(self.lookback_window, len(data))):
                    features[f'close_lag_{i}'] = data['close'].shift(i)

                # Добавление технических индикаторов
                features['close_ma_5'] = data['close'].rolling(5).mean()
                features['close_ma_20'] = data['close'].rolling(20).mean()
                features['close_ma_50'] = data['close'].rolling(50).mean()

                # Целевая переменная - будущая доходность
                features[target_col + '_future'] = data[target_col].shift(-self.forecast_period) / data[target_col] - 1

                # Удаление NaN
                features = features.dropna()

                if len(features) > 0:
                    # Февриковые столбцы
                    feature_cols = [col for col in features.columns
                                   if col != target_col + '_future' and not col.startswith(target_col)]
                    feature_cols.remove('high') if 'high' in feature_cols else None
                    feature_cols.remove('low') if 'low' in feature_cols else None

                    X = features[feature_cols]
                    y = features[target_col + '_future']

                    # Нормализация
                    X_scaled = self.scaler.fit_transform(X)

                    # Обучение простого модели
                    self.model.fit(X_scaled, y)
                    self.feature_cols = feature_cols
                    self.trained = True

                    logger.info(f"LSTM trained on {len(X)} samples with {len(feature_cols)} features")
                    return True

        except Exception as e:
            logger.error(f"LSTM training error: {e}")

        return False

    def predict(self, data: pd.DataFrame) -> Optional[float]:
        """Предсказание будущей доходности"""
        if not self.trained:
            return None

        try:
            if len(data) < len(self.feature_cols):
                return None

            # Подготовка последних данных
            latest_data = data.iloc[-1:]

            # Создание тех же признаков, что и при обучении
            for i in range(1, min(self.lookback_window, len(data))):
                latest_data[f'close_lag_{i}'] = data['close'].iloc[-i-1] if len(data) > i else data['close'].iloc[-1]

            latest_data['close_ma_5'] = data['close'].iloc[-5:].mean() if len(data) >= 5 else data['close'].iloc[-1]
            latest_data['close_ma_20'] = data['close'].iloc[-20:].mean() if len(data) >= 20 else data['close'].iloc[-1]
            latest_data['close_ma_50'] = data['close'].iloc[-50:].mean() if len(data) >= 50 else data['close'].iloc[-1]

            # Выбор признаков
            X = latest_data[self.feature_cols]
            X_scaled = self.scaler.transform(X)

            # Предсказание
            prediction = self.model.predict(X_scaled)[0]

            logger.debug(".4f")
            return prediction

        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            return None


class ATRCalculator:
    """
    ATR (Average True Range) Calculator для NeuroInvest
    """

    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> float:
        """Расчет ATR"""
        try:
            if len(data) < self.period:
                return 0.0

            high = data['high']
            low = data['low']
            close = data['close']

            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))

            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # ATR как среднее True Range
            atr = true_range.rolling(window=self.period).mean().iloc[-1]

            if pd.isna(atr):
                return 0.0

            return float(atr)

        except Exception as e:
            logger.error(f"ATR calculation error: {e}")
            return 0.0


class LSTMTrendFollowingStrategy(BaseStrategy):
    """
    LSTM Trend Following Strategy - основанная на NeuroInvest

    Логика:
    1. LSTM модель предсказывает будущую доходность
    2. ATR используется для динамического стоп-лосса
    3. Комбинация AI и риск-менеджмента
    """

    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        super().__init__("NeuroInvest LSTM_ATR", config)

        # Параметры LSTM
        self.model_type = config.get('model_type', 'LSTM')
        self.lookback_window = config.get('lookback_window', 60)
        self.forecast_period = config.get('forecast_period', 5)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)

        # Параметры ATR
        self.atr_period = config.get('atr_period', 14)
        self.atr_multiplier = config.get('atr_multiplier', 1.5)

        # Инициализация компонентов
        self.lstm_model = SimpleLSTM(self.lookback_window, self.forecast_period)
        self.atr_calculator = ATRCalculator(self.atr_period)

        # Состояние стратегии
        self.model_trained = False
        self.last_signal = Signal.HOLD
        self.position_size = 0.0

        logger.info(f"NeuroInvest LSTM ATR initialized: {self.lookback_window}d lookback, {self.forecast_period}d forecast")

    def generate_signal(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """
        Генерация сигнала на основе LSTM предсказания и ATR
        """

        try:
            if current_idx < max(self.lookback_window, self.atr_period):
                return Signal.HOLD

            # Текущие данные
            current_data = data.iloc[:current_idx + 1]
            current_price = current_data['close'].iloc[-1]

            # Обучение модели (если необходимо)
            if not self.model_trained:
                training_data = current_data.iloc[:-self.forecast_period]
                if len(training_data) >= self.lookback_window + self.forecast_period:
                    self.model_trained = self.lstm_model.fit(training_data)
                    if self.model_trained:
                        logger.info("NeuroInvest LSTM model trained successfully")
                    else:
                        logger.warning("Failed to train LSTM model")
                        return Signal.HOLD

            # Получение предсказания LSTM
            prediction = self.lstm_model.predict(current_data)
            if prediction is None:
                return Signal.HOLD

            # Расчет ATR для волатильности
            atr = self.atr_calculator.calculate(current_data)
            atr_stop = atr * self.atr_multiplier

            # Анализ предсказания
            abs_prediction = abs(prediction)

            # Логика сигналов NeuroInvest
            signal = Signal.HOLD

            # BUY: LSTM предсказывает рост и уверенность выше порога
            if prediction > self.confidence_threshold:
                signal = Signal.BUY
                atr_level = current_price + atr_stop
                logger.info(".4f")
                logger.info(".2f")

                self.signals_history.append({
                    'timestamp': current_data.index[-1],
                    'signal': signal,
                    'price': current_price,
                    'atr': atr,
                    'atr_stop': atr_stop,
                    'lstm_prediction': prediction,
                    'confidence': abs_prediction,
                    'reason': 'LSTM positive + confidence above threshold'
                })

            # SELL: LSTM предсказывает падение или стоп-лосс по ATR
            elif prediction < -self.confidence_threshold:
                signal = Signal.SELL
                atr_level = current_price - atr_stop
                logger.info(".4f")
                logger.info(".2f")

                self.signals_history.append({
                    'timestamp': current_data.index[-1],
                    'signal': signal,
                    'price': current_price,
                    'atr': atr,
                    'atr_stop': atr_stop,
                    'lstm_prediction': prediction,
                    'confidence': abs_prediction,
                    'reason': 'LSTM negative + confidence above threshold'
                })

            # Проверка стоп-лосса по ATR для активных позиций
            elif abs_prediction > self.confidence_threshold:
                logger.debug(".4f")

            return signal

        except Exception as e:
            logger.error(f"NeuroInvest LSTM ATR error: {e}")
            return Signal.HOLD


class EnhancedNeuroInvest(BaseStrategy):
    """
    Расширенная версия NeuroInvest с многомодельным подходом
    """

    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        super().__init__("Enhanced NeuroInvest", config)

        # Используем несколько моделей
        self.models = {
            'lstm': SimpleLSTM(config.get('lookback_window', 60), config.get('forecast_period', 5)),
            'trend': LinearRegression()  # Простая линейная регрессия для тренда
        }

        self.atr_calculator = ATRCalculator(config.get('atr_period', 14))
        self.ensemble_weight_lstm = config.get('ensemble_weight_lstm', 0.7)
        self.ensemble_weight_trend = config.get('ensemble_weight_trend', 0.3)
        self.adaptive_threshold = config.get('adaptive_threshold', True)

        self.trained = False
        self.current_threshold = config.get('confidence_threshold', 0.7)

    def generate_signal(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """
        Ансамблевая стратегия с LSTM и трендовой моделью
        """
        try:
            if current_idx < 100:  # Нужно больше данных для обучения
                return Signal.HOLD

            current_data = data.iloc[:current_idx + 1]

            # Обучение моделей
            if not self.trained:
                self._train_models(current_data)

            # Получение предсказаний от всех моделей
            predictions = {}
            confidences = {}

            # LSTM предсказание
            lstm_pred = self.models['lstm'].predict(current_data)
            predictions['lstm'] = lstm_pred if lstm_pred is not None else 0.0
            confidences['lstm'] = 1.0 if lstm_pred is not None else 0.0

            # Тренд предсказание
            trend_pred = self._trend_prediction(current_data)
            predictions['trend'] = trend_pred
            confidences['trend'] = 0.8  # Более низкая уверенность для тренда

            # Ансамблевое предсказание
            ensemble_pred = (
                predictions['lstm'] * self.ensemble_weight_lstm * confidences['lstm'] +
                predictions['trend'] * self.ensemble_weight_trend * confidences['trend']
            ) / (self.ensemble_weight_lstm + self.ensemble_weight_trend)

            # Адаптивный порог
            if self.adaptive_threshold:
                volatility = current_data['close'].pct_change().std() * 100
                self.current_threshold = min(0.9, max(0.5, volatility * 0.1))

            # Генерация сигнала
            if ensemble_pred > self.current_threshold:
                signal = Signal.BUY
                logger.info(".4f")
            elif ensemble_pred < -self.current_threshold:
                signal = Signal.SELL
                logger.info(".4f")
            else:
                signal = Signal.HOLD

            return signal

        except Exception as e:
            logger.error(f"Enhanced NeuroInvest error: {e}")
            return Signal.HOLD

    def _train_models(self, data: pd.DataFrame):
        """Обучение всех моделей"""
        try:
            training_data = data.iloc[:-self.models['lstm'].forecast_period]
            self.trained = self.models['lstm'].fit(training_data)
            # TODO: Обучение трендовой модели
            logger.info("Enhanced NeuroInvest models trained")
        except Exception as e:
            logger.error(f"Model training error: {e}")

    def _trend_prediction(self, data: pd.DataFrame) -> float:
        """Простое предсказание тренда"""
        try:
            # Линейный тренд по последним 20 точкам
            recent_data = data.iloc[-20:]
            x = np.arange(len(recent_data))
            y = recent_data['close'].values

            slope, intercept = np.polyfit(x, y, 1)

            # Нормализованный наклон как предсказание
            current_price = data['close'].iloc[-1]
            if current_price > 0:
                return slope / current_price  # Нормализованная сила тренда
            else:
                return 0.0

        except Exception:
            return 0.0


# Регистрация стратегий NeuroInvest
def _register_neuroinvest_strategies():
    """Регистрация стратегий NeuroInvest"""
    try:
        from strategies.main import register_strategy

        # Основная LSTM ATR стратегия
        register_strategy(
            'neuroinvest_lstm',
            LSTMTrendFollowingStrategy,
            {
                'description': 'LSTM Trend Following with ATR from NeuroInvest',
                'lookback_window': 60,
                'atr_period': 14,
                'atr_multiplier': 1.5,
                'forecast_period': 5,
                'confidence_threshold': 0.7,
                'source': 'https://github.com/VladimirTalyzin/NeuroInvest'
            }
        )

        # Расширенная ансамблевая стратегия
        register_strategy(
            'neuroinvest_enhanced',
            EnhancedNeuroInvest,
            {
                'description': 'Enhanced NeuroInvest with ensemble models',
                'lookback_window': 60,
                'atr_period': 14,
                'confidence_threshold': 0.7,
                'ensemble_weight_lstm': 0.7,
                'ensemble_weight_trend': 0.3,
                'source': 'https://github.com/VladimirTalyzin/NeuroInvest'
            }
        )

        logger.info("NeuroInvest strategies registered successfully")

    except Exception as e:
        logger.error(f"Failed to register NeuroInvest strategies: {e}")


# Автоматическая регистрация
_register_neuroinvest_strategies()