#!/usr/bin/env python3

import pandas as pd
import random
from datetime import datetime, timedelta
import os

def generate_sber_stock_data(days=365):
    """Генерирует реалистичные данные по акции СБЕР"""
    print("Генерация данных по акции СБЕР (SBER: BBG004730N88)")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Расчет количества торговых часов
    trading_days = sum(1 for d in pd.date_range(start_date, end_date, freq='D')
                      if d.weekday() < 5)  # Только будни
    total_candles = trading_days * 8  # 8 часов торговли

    dates = pd.date_range(start=start_date, end=end_date, periods=total_candles)

    base_price = 280  # Цена в рублях
    current_price = base_price
    sber_data = []

    year_trend_factor = 0.15  # +15% за год для российского рынка

    for i, date in enumerate(dates):
        days_elapsed = (date - start_date).total_seconds() / (365 * 24 * 3600)
        trend_factor = days_elapsed * year_trend_factor

        daily_noise = random.uniform(-3, 3)
        hourly_noise = random.uniform(-1, 1)

        bias = 0.2 if random.random() > 0.45 else -0.1
        price_change = trend_factor + daily_noise + hourly_noise + bias

        current_price += price_change
        current_price = max(50, current_price)

        # OHLC
        open_price = sber_data[-1]['close'] if sber_data else current_price
        close_price = current_price
        high = max(open_price, close_price) + random.uniform(0, 1.5)
        low = min(open_price, close_price) - random.uniform(0, 1.5)

        base_volume = 50000
        volume_factor = random.uniform(0.3, 2.5)
        volume = int(base_volume * volume_factor)

        sber_data.append({
            'time': date,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close_price, 2),
            'volume': volume,
            'symbol': 'SBER',
            'figi': 'BBG004730N88'
        })

    df = pd.DataFrame(sber_data)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    print(f"Сгенерировано {len(df)} часовых свечей по СБЕР")
    print(f"Цена старт: {df['close'].iloc[0]:.2f} руб")
    print(f"Цена финиш: {df['close'].iloc[-1]:.2f} руб")

    ratio = (df['close'].iloc[-1] / df['close'].iloc[0])
    if ratio > 1:
        print(f"Изменение: +{((ratio-1)*100):.1f}%")
    else:
        print(f"Изменение: {((ratio-1)*100):.1f}%")

    return df

def save_data_with_analysis(df, filename):
    """Сохраняет данные и создает анализ"""
    os.makedirs("data/raw", exist_ok=True)

    # CSV файл
    df.to_csv(filename)
    print(f"Данные сохранены в {filename}")

    # Аналитический отчет
    analysis_file = filename.replace('.csv', '_analysis.txt')
    with open(analysis_file, 'w', encoding='utf-8') as f:
        f.write("АНАЛИЗ ДАННЫХ ПО АКЦИИ СБЕР\n")
        f.write("="*40 + "\n\n")
        f.write(f"Период: {df.index[0]} - {df.index[-1]}\n")
        f.write(f"Количество записей: {len(df)}\n\n")

        f.write("ЦЕНОВОЙ АНАЛИЗ:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Начальная цена: {df['close'].iloc[0]:.2f} руб\n")
        f.write(f"Финальная цена: {df['close'].iloc[-1]:.2f} руб\n")
        f.write(f"Максимум: {df['high'].max():.2f} руб\n")
        f.write(f"Минимум: {df['low'].min():.2f} руб\n")

        price_change = ((df['close'].iloc[-1]/df['close'].iloc[0]-1)*100)
        f.write(f"Изменение: {price_change:+.1f}%\n\n")

        f.write("ОБЪЕМЫ ТОРГОВЛИ:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Средний объем: {int(df['volume'].mean()):,}\n")
        f.write(f"Максимальный объем: {df['volume'].max():,}\n\n")

        f.write("СТАТИСТИКА:\n")
        f.write("-" * 13 + "\n")
        f.write(f"Среднеквадратичное отклонение: {df['close'].std():.2f}\n")
        f.write(f"Медиана цены: {df['close'].median():.2f}\n")

    print(f"Анализ сохранен в {analysis_file}")

if __name__ == "__main__":
    # Генерация данных
    sber_data = generate_sber_stock_data(365)

    if sber_data is not None:
        print("\nПервые 5 свечей:")
        print(sber_data.head())

        # Сохранение с временной меткой
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"SBER_365d_hour_{timestamp}.csv"

        save_data_with_analysis(sber_data, filename)

        # Также в требуемую директорию
        sber_data.to_csv("data/raw/sber_1year.csv")
        print("Дополнительно сохранено в data/raw/sber_1year.csv")

        print("\nГЕНЕРАЦИЯ ДАННЫХ ПО СБЕР ЗАВЕРШЕНА!")
        print(f"Файлы: {filename}, data/raw/sber_1year.csv")
    else:
        print("Ошибка генерации данных")