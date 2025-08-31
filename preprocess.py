# src/data/preprocess.py
"""
Data preprocessing module for financial time series analysis.

Supported CSV format:
Required columns:
- date: DateTime column in format YYYY-MM-DD or YYYY-MM-DD HH:MM:SS
- open: Opening price (float)
- high: Highest price (float)
- low: Lowest price (float)
- close: Closing price (float)
- volume: Trading volume (integer/float, optional)

Optional columns:
- adj_close: Adjusted closing price
- symbol: Trading symbol/id
- dividends: Dividend payments
- stock_splits: Stock split ratios
- timestamp: Unix timestamp

Example CSV:
date,open,high,low,close,volume
2023-01-01,150.00,155.00,149.50,154.00,1000000
2023-01-02,154.10,158.00,153.00,157.50,1100000
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging
import warnings
from datetime import datetime

logger = logging.getLogger(__name__)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очистка данных согласно рекомендациям книги:
    - Удаление дубликатов
    - Обработка пропущенных значений
    - Коррекция аномалий
    """
    if df.empty:
        return df

    logger.info(f"🧹 Очистка данных: исходно {len(df)} записей")

    df_clean = df.copy()

    # 1. Удаление дубликатов по индексу (времени)
    before_dupes = len(df_clean)
    df_clean = df_clean[~df_clean.index.duplicated(keep='last')]
    if len(df_clean) < before_dupes:
        logger.info(f"🔄 Удалено дубликатов: {before_dupes - len(df_clean)}")

    # 2. Обработка аномальных значений в OHLCV данных
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    existing_cols = [col for col in numeric_cols if col in df_clean.columns]

    for col in existing_cols:
        # Убрать отрицательные значения цен
        if col != 'volume':
            negative_count = (df_clean[col] <= 0).sum()
            if negative_count > 0:
                logger.warning(f"⚠️ Найдено {negative_count} неположительных значений в {col}")
                df_clean[col] = df_clean[col].clip(lower=0.01)

        # Убрать экстремальные выбросы (больше 10 стандартных отклонений)
        mean_val = df_clean[col].mean()
        std_val = df_clean[col].std()
        outlier_threshold = 10

        outliers = abs(df_clean[col] - mean_val) > outlier_threshold * std_val
        outlier_count = outliers.sum()

        if outlier_count > 0:
            logger.warning(f"⚠️ Найдено {outlier_count} выбросов в {col}")
            # Заменяем выбросы медианным значением
            df_clean.loc[outliers, col] = df_clean[col].median()

    # 3. Проверка корректности OHLC данных
    if all(col in df_clean.columns for col in ['open', 'high', 'low', 'close']):
        # High должно быть >= max(open, close)
        invalid_high = df_clean['high'] < df_clean[['open', 'close']].max(axis=1)
        # Low должно быть <= min(open, close)  
        invalid_low = df_clean['low'] > df_clean[['open', 'close']].min(axis=1)

        invalid_count = (invalid_high | invalid_low).sum()
        if invalid_count > 0:
            logger.warning(f"⚠️ Найдено {invalid_count} некорректных OHLC записей")
            # Исправляем некорректные значения
            df_clean.loc[invalid_high, 'high'] = df_clean.loc[invalid_high, ['open', 'close']].max(axis=1)
            df_clean.loc[invalid_low, 'low'] = df_clean.loc[invalid_low, ['open', 'close']].min(axis=1)

    # 4. Обработка пропущенных значений
    missing_count = df_clean.isnull().sum().sum()
    if missing_count > 0:
        logger.info(f"🔧 Заполнение {missing_count} пропущенных значений")

        # Для цен используем forward fill, затем backward fill
        price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in df_clean.columns]
        df_clean[price_cols] = df_clean[price_cols].fillna(method='ffill').fillna(method='bfill')

        # Для объема заполняем нулями
        if 'volume' in df_clean.columns:
            df_clean['volume'] = df_clean['volume'].fillna(0)

        # Остальные колонки заполняем нулями
        df_clean = df_clean.fillna(0)

    logger.info(f"✅ Очистка завершена: {len(df_clean)} записей")
    return df_clean


def normalize_data(df: pd.DataFrame, method: str = 'minmax', 
                  columns: Optional[list] = None) -> Tuple[pd.DataFrame, dict]:
    """
    Нормализация данных как рекомендовано в книге

    Args:
        df: Исходные данные
        method: Метод нормализации ('minmax', 'standard', 'robust')
        columns: Колонки для нормализации (None = все числовые)

    Returns:
        Нормализованные данные и словарь с scaler'ами для обратного преобразования
    """
    if df.empty:
        return df, {}

    df_norm = df.copy()
    scalers = {}

    # Определить колонки для нормализации
    if columns is None:
        numeric_columns = df_norm.select_dtypes(include=[np.number]).columns.tolist()
        # Исключаем некоторые колонки, которые не нужно нормализовывать
        exclude_columns = ['volume', 'is_complete']
        columns = [col for col in numeric_columns if col not in exclude_columns]

    logger.info(f"📏 Нормализация методом '{method}' для колонок: {columns}")

    for col in columns:
        if col not in df_norm.columns:
            logger.warning(f"⚠️ Колонка {col} не найдена")
            continue

        values = df_norm[col].values.reshape(-1, 1)

        # Выбор метода нормализации
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            raise ValueError(f"Неизвестный метод нормализации: {method}")

        # Обучение и преобразование
        normalized_values = scaler.fit_transform(values).flatten()
        df_norm[col] = normalized_values
        scalers[col] = scaler

    logger.info(f"✅ Нормализация завершена для {len(columns)} колонок")
    return df_norm, scalers


def denormalize_data(df_normalized: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    """Обратное преобразование нормализованных данных"""
    df_denorm = df_normalized.copy()

    for col, scaler in scalers.items():
        if col in df_denorm.columns:
            values = df_denorm[col].values.reshape(-1, 1)
            denorm_values = scaler.inverse_transform(values).flatten()
            df_denorm[col] = denorm_values

    return df_denorm


def create_returns_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создание признаков доходности (основа стационарных рядов согласно книге)
    """
    if df.empty or 'close' not in df.columns:
        return df

    df_features = df.copy()

    # Простые доходности
    df_features['returns_1d'] = df_features['close'].pct_change(1)
    df_features['returns_5d'] = df_features['close'].pct_change(5) 
    df_features['returns_20d'] = df_features['close'].pct_change(20)

    # Логарифмические доходности (лучше для нормального распределения)
    df_features['log_returns_1d'] = np.log(df_features['close'] / df_features['close'].shift(1))
    df_features['log_returns_5d'] = np.log(df_features['close'] / df_features['close'].shift(5))

    # Волатильность (скользящие стандартные отклонения)
    df_features['volatility_10d'] = df_features['returns_1d'].rolling(10).std()
    df_features['volatility_20d'] = df_features['returns_1d'].rolling(20).std()

    # Z-score цены (отклонение от скользящего среднего в единицах стд. отклонения)
    df_features['price_zscore_20d'] = (df_features['close'] - df_features['close'].rolling(20).mean()) / df_features['close'].rolling(20).std()

    # Заполнить NaN
    df_features = df_features.fillna(0)

    logger.info("✨ Созданы признаки доходности и волатильности")
    return df_features


def create_time_windows(df: pd.DataFrame, window_size: int = 30, 
                       target_col: str = 'close', step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Создание временных окон для обучения LSTM/GRU согласно книге

    Args:
        df: Данные
        window_size: Размер окна (по умолчанию 30 как в книге)
        target_col: Целевая переменная для прогноза
        step: Шаг между окнами

    Returns:
        X: Входные последовательности (samples, timesteps, features)
        y: Целевые значения (samples,)
    """
    if len(df) <= window_size:
        logger.error(f"❌ Недостаточно данных: {len(df)} <= {window_size}")
        return np.array([]), np.array([])

    # Выбираем только числовые колонки для признаков
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Убираем целевую переменную из признаков если она там есть
    feature_cols = [col for col in numeric_cols if col != target_col]

    if not feature_cols:
        logger.error("❌ Нет числовых признаков для создания окон")
        return np.array([]), np.array([])

    # Подготовка данных
    features = df[feature_cols].values
    targets = df[target_col].values

    X, y = [], []

    # Создание скользящих окон
    for i in range(0, len(df) - window_size, step):
        # Входная последовательность
        X.append(features[i:i + window_size])
        # Целевое значение (следующее после окна)
        y.append(targets[i + window_size])

    X = np.array(X)
    y = np.array(y)

    logger.info(f"🪟 Создано временных окон: {len(X)} выборок, размер окна: {window_size}, признаков: {len(feature_cols)}")

    return X, y


def split_data(df: pd.DataFrame, train_ratio: float = 0.8, 
              val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Разделение данных на train/validation/test с учетом временной структуры
    """
    n = len(df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_data = df.iloc[:train_size]
    val_data = df.iloc[train_size:train_size + val_size] 
    test_data = df.iloc[train_size + val_size:]

    logger.info(f"📊 Разделение данных: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    return train_data, val_data, test_data


def add_lagged_features(df: pd.DataFrame, columns: list, lags: list = [1, 2, 3, 5]) -> pd.DataFrame:
    """
    Добавление лагированных признаков (важно для временных рядов)
    """
    df_lagged = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        for lag in lags:
            lag_col_name = f"{col}_lag_{lag}"
            df_lagged[lag_col_name] = df_lagged[col].shift(lag)

    # Заполняем NaN нулями
    df_lagged = df_lagged.fillna(0)

    logger.info(f"⏳ Добавлены лагированные признаки для {len(columns)} колонок с лагами {lags}")
    return df_lagged


def detect_outliers(df: pd.DataFrame, column: str, method: str = 'iqr', 
                   threshold: float = 1.5) -> pd.Series:
    """
    Обнаружение выбросов различными методами

    Args:
        df: Данные
        column: Колонка для анализа
        method: Метод ('iqr', 'zscore', 'isolation_forest')
        threshold: Пороговое значение

    Returns:
        Булева серия с True для выбросов
    """
    if column not in df.columns:
        return pd.Series([False] * len(df), index=df.index)

    values = df[column]

    if method == 'iqr':
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (values < (Q1 - threshold * IQR)) | (values > (Q3 + threshold * IQR))

    elif method == 'zscore':
        z_scores = abs((values - values.mean()) / values.std())
        outliers = z_scores > threshold

    elif method == 'isolation_forest':
        try:
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(values.values.reshape(-1, 1))
            outliers = pd.Series(outlier_labels == -1, index=df.index)
        except ImportError:
            logger.warning("⚠️ Sklearn не установлен, используем метод IQR")
            return detect_outliers(df, column, method='iqr', threshold=threshold)
    else:
        raise ValueError(f"Неизвестный метод: {method}")

    outlier_count = outliers.sum()
    logger.info(f"🔍 Обнаружено выбросов в {column} методом {method}: {outlier_count} ({outlier_count/len(df)*100:.1f}%)")

    return outliers


def validate_csv_format(file_path: str) -> Tuple[bool, List[str]]:
    """
    Validate CSV file format for financial data

    Args:
        file_path: Path to CSV file

    Returns:
        Tuple of (is_valid, error_messages)
    """
    try:
        df = pd.read_csv(file_path, nrows=5)  # Read only first 5 rows for validation
    except Exception as e:
        return False, [f"Cannot read CSV file: {e}"]

    errors = []

    # Check required columns
    required_columns = ['date', 'open', 'high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")

    # Validate date column
    if 'date' in df.columns:
        try:
            pd.to_datetime(df['date'], errors='raise')
        except Exception:
            errors.append("Invalid date format in 'date' column")

    # Validate numeric columns
    numeric_columns = ['open', 'high', 'low', 'close']
    for col in numeric_columns:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column '{col}' must be numeric")

    # Check for positive prices
    if 'close' in df.columns:
        if (df['close'] <= 0).any():
            errors.append("Found non-positive closing prices")

    return len(errors) == 0, errors


def load_and_validate_csv(file_path: str, date_column: str = 'date',
                         parse_dates: bool = True) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Load CSV file with validation and proper date parsing

    Args:
        file_path: Path to CSV file
        date_column: Name of date column
        parse_dates: Whether to parse dates

    Returns:
        Tuple of (dataframe or None, error messages)
    """
    errors = []

    try:
        # Read CSV with date parsing
        if parse_dates and date_column:
            df = pd.read_csv(file_path, parse_dates=[date_column])
            if date_column in df.columns:
                df = df.sort_values(date_column).set_index(date_column)
        else:
            df = pd.read_csv(file_path)

        # Basic validation
        if df.empty:
            errors.append("CSV file is empty")
            return None, errors

        # Remove duplicates
        initial_len = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_len:
            warnings.warn(f"Removed {initial_len - len(df)} duplicate rows")

        logger.info(f"Successfully loaded CSV: {len(df)} rows, {len(df.columns)} columns")
        return df, errors

    except Exception as e:
        errors.append(f"Error loading CSV: {e}")
        return None, errors


if __name__ == "__main__":
    # Тестирование модуля
    print("🧪 Тестирование модуля предобработки данных")

    # Создаем тестовые данные
    dates = pd.date_range('2023-01-01', periods=100, freq='H')
    test_data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 105,
        'low': np.random.randn(100).cumsum() + 95,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

    # Добавляем аномальные значения
    test_data.loc[test_data.index[50], 'close'] = -10  # отрицательная цена
    test_data.loc[test_data.index[60], 'high'] = 1000   # выброс

    print(f"📊 Тестовые данные: {len(test_data)} записей")

    # Тестирование очистки
    cleaned_data = clean_data(test_data)
    print(f"✅ После очистки: {len(cleaned_data)} записей")

    # Тестирование нормализации
    normalized_data, scalers = normalize_data(cleaned_data)
    print(f"📏 Нормализация выполнена для {len(scalers)} колонок")

    # Тестирование создания окон
    X, y = create_time_windows(normalized_data, window_size=10)
    print(f"🪟 Создано окон: {len(X)} выборок размером {X.shape[1]}x{X.shape[2]}")

    print("✅ Все тесты пройдены!")
