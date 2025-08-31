# src/features/indicators.py
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def sma(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Simple Moving Average - простое скользящее среднее
    Базовый индикатор тренда из книги
    """
    if len(series) < window:
        logger.warning(f"⚠️ Недостаточно данных для SMA({window}): {len(series)} < {window}")
        return pd.Series([np.nan] * len(series), index=series.index)

    return series.rolling(window=window, min_periods=1).mean()


def ema(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Exponential Moving Average - экспоненциальное скользящее среднее
    Более чувствительно к последним ценам
    """
    return series.ewm(span=window, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index - индекс относительной силы
    Осциллятор от 0 до 100, показывает перекупленность/перепроданность
    """
    if len(series) < window + 1:
        logger.warning(f"⚠️ Недостаточно данных для RSI({window}): {len(series)} < {window + 1}")
        return pd.Series([50.0] * len(series), index=series.index)

    # Вычисление изменений цены
    delta = series.diff()

    # Разделение на прибыли и убытки
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Сглаженные средние прибылей и убытков
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    # Для более точного RSI используем экспоненциальное сглаживание
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()

    # Вычисление RSI
    rs = avg_gain / (avg_loss + 1e-10)  # избегаем деления на ноль
    rsi_values = 100 - (100 / (1 + rs))

    return rsi_values


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    MACD - схождение-расхождение скользящих средних
    Возвращает DataFrame с колонками: macd, signal, histogram
    """
    if len(series) < slow + signal:
        logger.warning(f"⚠️ Недостаточно данных для MACD: {len(series)} < {slow + signal}")
        return pd.DataFrame({
            'macd': [0.0] * len(series),
            'signal': [0.0] * len(series), 
            'histogram': [0.0] * len(series)
        }, index=series.index)

    # Быстрая и медленная EMA
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)

    # Линия MACD
    macd_line = ema_fast - ema_slow

    # Сигнальная линия (EMA от MACD)
    signal_line = ema(macd_line, signal)

    # Гистограмма
    histogram = macd_line - signal_line

    return pd.DataFrame({
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }, index=series.index)


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2) -> pd.DataFrame:
    """
    Bollinger Bands - полосы Боллинжера
    Возвращает верхнюю, среднюю и нижнюю полосы
    """
    if len(series) < window:
        logger.warning(f"⚠️ Недостаточно данных для Bollinger Bands({window}): {len(series)}")
        middle = pd.Series([series.iloc[0] if len(series) > 0 else 0] * len(series), index=series.index)
        return pd.DataFrame({
            'upper': middle,
            'middle': middle,
            'lower': middle,
            'width': [0.0] * len(series),
            'position': [0.5] * len(series)
        }, index=series.index)

    # Средняя линия (SMA)
    middle_band = sma(series, window)

    # Стандартное отклонение
    std_dev = series.rolling(window=window, min_periods=1).std()

    # Верхняя и нижняя полосы
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)

    # Ширина полос (нормализованная)
    band_width = (upper_band - lower_band) / middle_band

    # Позиция цены в полосах (0 = нижняя полоса, 1 = верхняя полоса)
    band_position = (series - lower_band) / (upper_band - lower_band)
    band_position = band_position.clip(0, 1)  # ограничиваем 0-1

    return pd.DataFrame({
        'upper': upper_band,
        'middle': middle_band,
        'lower': lower_band,
        'width': band_width,
        'position': band_position
    }, index=series.index)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Average True Range - средний истинный диапазон
    Индикатор волатильности
    """
    if len(high) < window + 1:
        logger.warning(f"⚠️ Недостаточно данных для ATR({window}): {len(high)}")
        return pd.Series([0.0] * len(high), index=high.index)

    # True Range компоненты
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    # True Range = максимум из трех компонентов
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR = экспоненциальное скользящее среднее True Range
    atr_values = true_range.ewm(span=window, adjust=False).mean()

    return atr_values


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
               k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
    """
    Stochastic Oscillator - стохастический осциллятор
    Возвращает %K и %D линии
    """
    if len(high) < k_window:
        logger.warning(f"⚠️ Недостаточно данных для Stochastic({k_window}): {len(high)}")
        return pd.DataFrame({
            'k_percent': [50.0] * len(high),
            'd_percent': [50.0] * len(high)
        }, index=high.index)

    # Максимум и минимум за период
    highest_high = high.rolling(window=k_window, min_periods=1).max()
    lowest_low = low.rolling(window=k_window, min_periods=1).min()

    # %K линия
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)

    # %D линия (сглаженная %K)
    d_percent = k_percent.rolling(window=d_window, min_periods=1).mean()

    return pd.DataFrame({
        'k_percent': k_percent,
        'd_percent': d_percent
    }, index=high.index)


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Williams %R - индикатор Уильямса
    Осциллятор от -100 до 0
    """
    if len(high) < window:
        logger.warning(f"⚠️ Недостаточно данных для Williams %R({window}): {len(high)}")
        return pd.Series([-50.0] * len(high), index=high.index)

    highest_high = high.rolling(window=window, min_periods=1).max()
    lowest_low = low.rolling(window=window, min_periods=1).min()

    williams_r_values = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)

    return williams_r_values


def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """
    Commodity Channel Index - индекс товарного канала
    Осциллятор без ограничений, обычно колеблется между -100 и +100
    """
    if len(high) < window:
        logger.warning(f"⚠️ Недостаточно данных для CCI({window}): {len(high)}")
        return pd.Series([0.0] * len(high), index=high.index)

    # Типичная цена
    typical_price = (high + low + close) / 3

    # Скользящее среднее типичной цены
    sma_tp = typical_price.rolling(window=window, min_periods=1).mean()

    # Среднее отклонение
    mean_deviation = typical_price.rolling(window=window, min_periods=1).apply(
        lambda x: abs(x - x.mean()).mean(), raw=False
    )

    # CCI = (Типичная цена - SMA) / (0.015 * Среднее отклонение)
    cci_values = (typical_price - sma_tp) / (0.015 * mean_deviation + 1e-10)

    return cci_values


def roc(series: pd.Series, window: int = 12) -> pd.Series:
    """
    Rate of Change - скорость изменения
    Показывает процентное изменение цены за период
    """
    if len(series) < window + 1:
        logger.warning(f"⚠️ Недостаточно данных для ROC({window}): {len(series)}")
        return pd.Series([0.0] * len(series), index=series.index)

    roc_values = ((series - series.shift(window)) / series.shift(window)) * 100
    return roc_values.fillna(0)


def momentum(series: pd.Series, window: int = 10) -> pd.Series:
    """
    Momentum - моментум
    Разность между текущей ценой и ценой n периодов назад
    """
    if len(series) < window + 1:
        logger.warning(f"⚠️ Недостаточно данных для Momentum({window}): {len(series)}")
        return pd.Series([0.0] * len(series), index=series.index)

    momentum_values = series - series.shift(window)
    return momentum_values.fillna(0)


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавить все технические индикаторы к DataFrame
    Функция для автоматического обогащения данных всеми индикаторами
    """
    if df.empty:
        return df

    logger.info("✨ Добавление всех технических индикаторов...")

    df_indicators = df.copy()

    # Проверяем наличие необходимых колонок
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"❌ Отсутствуют колонки: {missing_cols}")
        return df

    try:
        # Скользящие средние
        df_indicators['sma_10'] = sma(df['close'], 10)
        df_indicators['sma_20'] = sma(df['close'], 20)
        df_indicators['sma_50'] = sma(df['close'], 50)
        df_indicators['ema_12'] = ema(df['close'], 12)
        df_indicators['ema_26'] = ema(df['close'], 26)

        # Осцилляторы
        df_indicators['rsi'] = rsi(df['close'])
        df_indicators['rsi_oversold'] = (df_indicators['rsi'] < 30).astype(int)
        df_indicators['rsi_overbought'] = (df_indicators['rsi'] > 70).astype(int)

        # MACD
        macd_data = macd(df['close'])
        df_indicators['macd'] = macd_data['macd']
        df_indicators['macd_signal'] = macd_data['signal']
        df_indicators['macd_histogram'] = macd_data['histogram']
        df_indicators['macd_bullish'] = (df_indicators['macd'] > df_indicators['macd_signal']).astype(int)

        # Bollinger Bands
        bb_data = bollinger_bands(df['close'])
        df_indicators['bb_upper'] = bb_data['upper']
        df_indicators['bb_middle'] = bb_data['middle']
        df_indicators['bb_lower'] = bb_data['lower']
        df_indicators['bb_width'] = bb_data['width']
        df_indicators['bb_position'] = bb_data['position']
        df_indicators['bb_squeeze'] = (df_indicators['bb_width'] < df_indicators['bb_width'].rolling(20).quantile(0.2)).astype(int)

        # ATR (волатильность)
        df_indicators['atr'] = atr(df['high'], df['low'], df['close'])
        df_indicators['atr_normalized'] = df_indicators['atr'] / df['close']  # нормализованный ATR

        # Stochastic
        stoch_data = stochastic(df['high'], df['low'], df['close'])
        df_indicators['stoch_k'] = stoch_data['k_percent']
        df_indicators['stoch_d'] = stoch_data['d_percent']
        df_indicators['stoch_oversold'] = (df_indicators['stoch_k'] < 20).astype(int)
        df_indicators['stoch_overbought'] = (df_indicators['stoch_k'] > 80).astype(int)

        # Дополнительные индикаторы
        df_indicators['williams_r'] = williams_r(df['high'], df['low'], df['close'])
        df_indicators['cci'] = cci(df['high'], df['low'], df['close'])
        df_indicators['roc'] = roc(df['close'])
        df_indicators['momentum'] = momentum(df['close'])

        # Ценовые паттерны и сигналы
        df_indicators['ma_cross_bullish'] = (
            (df_indicators['sma_10'] > df_indicators['sma_20']) & 
            (df_indicators['sma_10'].shift(1) <= df_indicators['sma_20'].shift(1))
        ).astype(int)

        df_indicators['ma_cross_bearish'] = (
            (df_indicators['sma_10'] < df_indicators['sma_20']) & 
            (df_indicators['sma_10'].shift(1) >= df_indicators['sma_20'].shift(1))
        ).astype(int)

        # Трендовые фильтры
        df_indicators['uptrend'] = (df['close'] > df_indicators['sma_20']).astype(int)
        df_indicators['strong_uptrend'] = (
            (df['close'] > df_indicators['sma_20']) & 
            (df_indicators['sma_20'] > df_indicators['sma_50'])
        ).astype(int)

        # Заполнить NaN значения
        df_indicators = df_indicators.fillna(method='ffill').fillna(0)

        added_indicators = len(df_indicators.columns) - len(df.columns)
        logger.info(f"✅ Добавлено индикаторов: {added_indicators}")

        return df_indicators

    except Exception as e:
        logger.error(f"❌ Ошибка добавления индикаторов: {e}")
        return df


def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Обнаружение простых ценовых паттернов
    """
    if df.empty or not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        return df

    df_patterns = df.copy()

    # Дожи (открытие ≈ закрытие)
    body_size = abs(df['close'] - df['open'])
    candle_range = df['high'] - df['low']
    df_patterns['doji'] = (body_size <= candle_range * 0.1).astype(int)

    # Молот (длинная нижняя тень, короткая верхняя)
    lower_shadow = df['open'].combine(df['close'], min) - df['low']
    upper_shadow = df['high'] - df['open'].combine(df['close'], max)
    df_patterns['hammer'] = (
        (lower_shadow > 2 * body_size) & 
        (upper_shadow < body_size * 0.5)
    ).astype(int)

    # Висящий (длинная верхняя тень, короткая нижняя)
    df_patterns['hanging_man'] = (
        (upper_shadow > 2 * body_size) & 
        (lower_shadow < body_size * 0.5)
    ).astype(int)

    logger.info("📈 Добавлены паттерны: doji, hammer, hanging_man")

    return df_patterns


if __name__ == "__main__":
    # Тестирование индикаторов
    print("🧪 Тестирование технических индикаторов")

    # Создаем тестовые данные
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)

    # Имитируем реалистичные ценовые движения
    base_price = 100
    price_changes = np.random.randn(100).cumsum() * 2
    closes = base_price + price_changes

    test_data = pd.DataFrame({
        'open': closes * (1 + np.random.randn(100) * 0.01),
        'high': closes * (1 + abs(np.random.randn(100)) * 0.02),
        'low': closes * (1 - abs(np.random.randn(100)) * 0.02),
        'close': closes,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

    # Убеждаемся что high >= max(open, close) и low <= min(open, close)
    test_data['high'] = test_data[['open', 'close', 'high']].max(axis=1)
    test_data['low'] = test_data[['open', 'close', 'low']].min(axis=1)

    print(f"📊 Тестовые данные: {len(test_data)} записей")

    # Тестирование отдельных индикаторов
    print("\n📈 Тестирование индикаторов:")

    rsi_values = rsi(test_data['close'])
    print(f"RSI: {rsi_values.iloc[-1]:.2f} (последнее значение)")

    macd_values = macd(test_data['close'])
    print(f"MACD: {macd_values['macd'].iloc[-1]:.4f}, Signal: {macd_values['signal'].iloc[-1]:.4f}")

    bb_values = bollinger_bands(test_data['close'])
    print(f"Bollinger Position: {bb_values['position'].iloc[-1]:.2f}")

    atr_values = atr(test_data['high'], test_data['low'], test_data['close'])
    print(f"ATR: {atr_values.iloc[-1]:.2f}")

    # Тестирование добавления всех индикаторов
    print("\n✨ Добавление всех индикаторов...")
    enriched_data = add_all_indicators(test_data)

    print(f"📊 Исходные колонки: {len(test_data.columns)}")
    print(f"📊 После обогащения: {len(enriched_data.columns)}")
    print(f"📈 Добавлено индикаторов: {len(enriched_data.columns) - len(test_data.columns)}")

    # Проверка на NaN
    nan_count = enriched_data.isnull().sum().sum()
    print(f"🔍 NaN значений: {nan_count}")

    if nan_count == 0:
        print("✅ Все тесты пройдены!")
    else:
        print("⚠️ Есть NaN значения, проверьте индикаторы")
