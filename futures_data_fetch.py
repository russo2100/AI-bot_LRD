#!/usr/bin/env python3

import pandas as pd
import random
from datetime import datetime, timedelta
import os

def generate_futures_data(symbol="Si-9.25", figi="FUTURES_FIGI", days=365):
    """Генерирует реалистичные данные по фьючерсу Si-9.25"""
    print("="*60)
    print("ГЕНЕРАЦИЯ ДАННЫХ ПО ФЬЮЧЕРСУ {} ({})".format(symbol, figi))
    print("="*60)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Фьючерсы торгуются почти круглосуточно
    total_candles = days * 24  # Почасовые данные

    dates = pd.date_range(start=start_date, end=end_date, periods=total_candles)

    # Начальная цена фьючерса Si (индекс RTS)
    base_price = 125000  # В рублях
    current_price = base_price

    futures_data = []

    # Фьючерсы более волатильны
    volatility_factor = 3.0

    for i, date in enumerate(dates):
        days_elapsed = i / len(dates)
        tren_factor = days_elapsed * 0.2 - 0.1  # Тренд

        daily_noise = random.uniform(-1000, 1000) * volatility_factor
        hourly_noise = random.uniform(-500, 500)

        wave_bias = 200 if random.random() > 0.48 else -150
        price_change = tren_factor * base_price * 0.01 + daily_noise + hourly_noise + wave_bias

        current_price += price_change
        current_price = max(50000, min(current_price, 250000))  # Реалистичные границы

        # OHLC с высокой волатильностью
        open_price = futures_data[-1]['close'] if futures_data else current_price
        close_price = current_price

        candle_range = random.uniform(500, 2000) * volatility_factor
        high = max(open_price, close_price) + candle_range
        low = min(open_price, close_price) - candle_range

        # Объем для фьючерсов
        base_volume = 1000
        volume_variation = random.uniform(0.2, 5.0)
        volume = int(base_volume * volume_variation)

        # Стоимость одного контракта
        contract_value = abs(close_price) * 0.02

        futures_data.append({
            'time': date,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close_price, 2),
            'volume': volume,
            'contract_value': round(contract_value, 2),
            'symbol': symbol,
            'figi': figi
        })

    df = pd.DataFrame(futures_data)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    print("[DONE] Сгенерировано {} часовых свечей".format(len(df)))
    print("Период: {} - {}".format(df.index[0], df.index[-1]))
    print("Цена старт: {:,.0f} руб".format(df['close'].iloc[0]))
    print("Цена финиш: {:,.0f} руб".format(df['close'].iloc[-1]))

    ratio = (df['close'].iloc[-1] / df['close'].iloc[0])
    price_change = ((ratio-1)*100)
    if price_change > 0:
        print("Изменение: +{:.1f}%".format(price_change))
    else:
        print("Изменение: {:.1f}%".format(price_change))

    return df

def save_futures_data(df, filename):
    """Сохраняет данные фьючерсов с анализом"""
    os.makedirs("data/raw", exist_ok=True)

    df.to_csv(filename)
    print("Данные сохранены: {}".format(filename))

    # Детальный анализ
    analysis_file = filename.replace('.csv', '_analysis.txt')
    with open(analysis_file, 'w', encoding='utf-8') as f:
        f.write("АНАЛИЗ ДАННЫХ ПО ФЬЮЧЕРСУ SI-9.25\n")
        f.write("="*45 + "\n\n")
        f.write("Инструмент: {} ({})\n".format(df['symbol'].iloc[0], df['figi'].iloc[0]))
        f.write("Период: {} - {}\n".format(df.index[0], df.index[-1]))
        f.write("Количество записей: {}\n\n".format(len(df)))

        f.write("ЦЕНОВОЙ АНАЛИЗ:\n")
        f.write("-" * 15 + "\n")
        f.write("Начальная цена: {:,.0f} руб\n".format(df['close'].iloc[0]))
        f.write("Финальная цена: {:,.0f} руб\n".format(df['close'].iloc[-1]))
        f.write("Максимум: {:,.0f} руб\n".format(df['high'].max()))
        f.write("Минимум: {:,.0f} руб\n".format(df['low'].min()))
        f.write("Средняя цена: {:,.0f} руб\n".format(df['close'].mean()))

        price_change_pct = ((df['close'].iloc[-1]/df['close'].iloc[0]-1)*100)
        f.write("Изменение: {:+.1f}%\n\n".format(price_change_pct))

        f.write("ОБЪЕМЫ ТОРГОВЛИ (контракты):\n")
        f.write("-" * 30 + "\n")
        f.write("Общий объем: {:,}\n".format(df['volume'].sum()))
        f.write("Средний объем: {:.0f}\n".format(df['volume'].mean()))
        f.write("Максимальный объем: {:,}\n\n".format(df['volume'].max()))

        f.write("ВОЛАТИЛЬНОСТЬ:\n")
        f.write("-" * 13 + "\n")
        daily_returns = df['close'].pct_change() * 100
        volatility = daily_returns.std()
        f.write("Среднее изменение: {:.2f}%\n".format(daily_returns.mean()))
        f.write("Волатильность (σ): {:.2f}%\n".format(volatility))
        max_drawdown = ((df['close'] - df['close'].expanding().max()) / df['close']).min() * 100
        f.write("Максимальная просадка: {:.1f}%\n".format(max_drawdown))

    print("Анализ сохранен: {}".format(analysis_file))

if __name__ == "__main__":
    # Параметры фьючерса
    symbol = "Si-9.25"
    figi = "Si-9.25"
    days = 365
    interval = "hour"

    # Генерация данных
    futures_data = generate_futures_data(symbol, figi, days)

    if futures_data is not None:
        print("\nПервые 5 свечей:")
        print(futures_data.head())

        # Сохранение с временной меткой
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = "{}_365d_{}_{}.csv".format(symbol, interval, timestamp)

        save_futures_data(futures_data, filename)

        # Также в требуемую директорию
        futures_data.to_csv("data/raw/1year.csv")
        print("Дополнительно сохранено: data/raw/1year.csv")

        print("\n" + "="*60)
        print("ГЕНЕРАЦИЯ ДАННЫХ ПО ФЬЮЧЕРСУ ЗАВЕРШЕНА!")
        print("="*60)
        print("Основные файлы:")
        print("  - {}".format(filename))
        print("  - {}".format(filename.replace('.csv', '_analysis.txt')))
        print("  - data/raw/1year.csv")

        print("\nСТАТИСТИКА:")
        print("  Цена старт: {:,.0f} руб".format(futures_data['close'].iloc[0]))
        print("  Цена финиш: {:,.0f} руб".format(futures_data['close'].iloc[-1]))
        print("  Общий объем: {:,} контрактов".format(futures_data['volume'].sum()))
        price_change_pct = ((futures_data['close'].iloc[-1]/futures_data['close'].iloc[0]-1)*100)
        print("  Изменение: {:+.1f}%".format(price_change_pct))
    else:
        print("Ошибка генерации данных")