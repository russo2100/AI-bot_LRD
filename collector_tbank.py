# src/data/collector_tbank.py
from __future__ import annotations

import pandas as pd
from datetime import datetime, timezone
from typing import List, Optional
from tinkoff.invest import CandleInterval
from src.execution.broker_tbank import TBankBroker
import logging

logger = logging.getLogger(__name__)

def quotation_to_float(quotation) -> float:
    """Преобразование Quotation в float"""
    if quotation is None:
        return 0.0
    return quotation.units + quotation.nano / 1_000_000_000


def candles_to_df(candles) -> pd.DataFrame:
    """
    Преобразование свечей в pandas DataFrame
    Основано на рекомендациях из книги по структуре данных
    """
    if not candles:
        return pd.DataFrame()

    rows = []
    for candle in candles:
        row = {
            "time": pd.to_datetime(candle.time, utc=True),
            "open": quotation_to_float(candle.open),
            "high": quotation_to_float(candle.high),
            "low": quotation_to_float(candle.low),
            "close": quotation_to_float(candle.close),
            "volume": candle.volume,
            "is_complete": candle.is_complete,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if not df.empty:
        # Установить индекс по времени и отсортировать
        df = df.set_index("time").sort_index()

        # Убрать дубликаты по времени (если есть)
        df = df[~df.index.duplicated(keep='last')]

        logger.info(f"📊 Создан DataFrame: {len(df)} строк, период с {df.index[0]} по {df.index[-1]}")

    return df


def load_history(figi: str, days: int = 365, 
                interval: CandleInterval = CandleInterval.CANDLE_INTERVAL_HOUR,
                save_to: Optional[str] = None) -> pd.DataFrame:
    """
    Загрузить исторические данные для инструмента

    Args:
        figi: Идентификатор инструмента
        days: Количество дней истории
        interval: Интервал свечей
        save_to: Путь для сохранения данных (опционально)

    Returns:
        DataFrame с историческими данными
    """
    try:
        logger.info(f"📈 Загрузка данных {figi}: {days} дней, интервал {interval}")

        broker = TBankBroker()
        candles = broker.get_candles(figi=figi, days=days, interval=interval)
        df = candles_to_df(candles)

        if df.empty:
            logger.warning(f"⚠️ Нет данных для {figi}")
            return df

        # Добавить базовые вычисляемые поля
        df = add_basic_features(df)

        # Сохранить если указан путь
        if save_to:
            df.to_csv(save_to)
            logger.info(f"💾 Данные сохранены в {save_to}")

        return df

    except Exception as e:
        logger.error(f"❌ Ошибка загрузки данных {figi}: {e}")
        raise


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавить базовые признаки согласно книге:
    - Доходности
    - Логарифмические доходности  
    - Волатильность
    - Типичная цена
    """
    if df.empty:
        return df

    df = df.copy()

    # Доходности (как рекомендовано в книге для стационарности)
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = (df['close'] / df['close'].shift(1)).apply(lambda x: pd.np.log(x) if x > 0 else 0)

    # Типичная цена (используется в индикаторах)
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

    # Волатильность (скользящее стандартное отклонение доходностей)
    df['volatility'] = df['returns'].rolling(window=20).std()

    # True Range для ATR
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

    # Убрать вспомогательные колонки
    df = df.drop(['prev_close', 'tr1', 'tr2', 'tr3'], axis=1, errors='ignore')

    # Заполнить NaN нулями в первых строках
    df = df.fillna(0)

    logger.info(f"✨ Добавлены базовые признаки: returns, log_returns, volatility, true_range")

    return df


def get_multiple_instruments(figis: List[str], days: int = 365,
                           interval: CandleInterval = CandleInterval.CANDLE_INTERVAL_HOUR) -> dict:
    """
    Загрузить данные для нескольких инструментов
    Полезно для корреляционного анализа и портфельных стратегий
    """
    results = {}

    for figi in figis:
        try:
            logger.info(f"📊 Загружаем {figi}...")
            df = load_history(figi, days=days, interval=interval)
            if not df.empty:
                results[figi] = df
                logger.info(f"✅ {figi}: {len(df)} записей")
            else:
                logger.warning(f"⚠️ {figi}: данных нет")
        except Exception as e:
            logger.error(f"❌ {figi}: ошибка загрузки - {e}")

    logger.info(f"📈 Загружено инструментов: {len(results)} из {len(figis)}")
    return results


def update_data(existing_file: str, figi: str, 
               interval: CandleInterval = CandleInterval.CANDLE_INTERVAL_HOUR) -> pd.DataFrame:
    """
    Обновить существующие данные новыми свечами
    Загружает только недостающий период
    """
    try:
        # Загрузить существующие данные
        df_existing = pd.read_csv(existing_file, parse_dates=['time'], index_col='time')
        last_date = df_existing.index[-1]

        # Рассчитать сколько дней нужно добавить
        days_to_add = (datetime.now(timezone.utc) - last_date).days + 1

        if days_to_add <= 1:
            logger.info("📊 Данные уже актуальны")
            return df_existing

        logger.info(f"📈 Обновление данных: добавляем {days_to_add} дней")

        # Загрузить новые данные
        df_new = load_history(figi, days=days_to_add, interval=interval)

        if df_new.empty:
            logger.warning("⚠️ Новых данных не получено")
            return df_existing

        # Объединить данные
        df_combined = pd.concat([df_existing, df_new])
        df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
        df_combined = df_combined.sort_index()

        # Сохранить обновленные данные
        df_combined.to_csv(existing_file)

        added_rows = len(df_combined) - len(df_existing)
        logger.info(f"✅ Добавлено {added_rows} новых записей")

        return df_combined

    except Exception as e:
        logger.error(f"❌ Ошибка обновления данных: {e}")
        # В случае ошибки загрузить данные заново
        return load_history(figi, days=30, interval=interval, save_to=existing_file)


# Популярные инструменты для быстрого доступа
POPULAR_FIGIS = {
    "SBER": "BBG004730N88",
    "GAZP": "BBG004730RP0", 
    "LKOH": "BBG004731354",
    "YNDX": "BBG00178PGX3",
    "VTBR": "BBG004730ZJ9",
    "NVTK": "BBG004731032",
    "AFLT": "BBG004730JJ5",
    "ALRS": "BBG004730B03",
    "GMKN": "BBG004730RP0",
    "ROSN": "BBG004731354",
}

def get_popular_stocks_data(days: int = 365, 
                          interval: CandleInterval = CandleInterval.CANDLE_INTERVAL_HOUR) -> dict:
    """Загрузить данные по популярным российским акциям"""
    logger.info("📊 Загрузка данных популярных акций...")
    return get_multiple_instruments(list(POPULAR_FIGIS.values()), days=days, interval=interval)


if __name__ == "__main__":
    # Тестирование модуля
    import sys

    if len(sys.argv) > 1:
        figi = sys.argv[1]
    else:
        figi = POPULAR_FIGIS["SBER"]

    print(f"🧪 Тестирование загрузки данных для {figi}")

    try:
        df = load_history(figi, days=30, interval=CandleInterval.CANDLE_INTERVAL_HOUR)
        print(f"✅ Загружено {len(df)} записей")
        print(f"📊 Период: {df.index[0]} - {df.index[-1]}")
        print(f"📈 Цена: {df['close'].iloc[0]:.2f} -> {df['close'].iloc[-1]:.2f}")
        print("\n📋 Первые 5 записей:")
        print(df.head())

    except Exception as e:
        print(f"❌ Ошибка: {e}")
