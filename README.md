# 🤖 AI-Торговый Бот

Интеллектуальная система автоматического трейдинга на основе машинного обучения и ИИ, разработанная с использованием принципов и подходов из книги "ИИ в Трейдинге".

## 📋 Содержание

- [Возможности](#возможности)
- [Архитектура](#архитектура)
- [Установка](#установка)
- [Настройка](#настройка)
- [Использование](#использование)
- [Модели ИИ](#модели-ии)
- [Торговые стратегии](#торговые-стратегии)
- [Бэктестинг](#бэктестинг)
- [API интеграции](#api-интеграции)
- [Мониторинг](#мониторинг)
- [Структура проекта](#структура-проекта)

## 🚀 Возможности

### Модели машинного обучения
- **ARIMA модели** для прогнозирования временных рядов
- **LSTM нейронные сети** для анализа последовательностей
- **Классификаторы направления** (Random Forest, Gradient Boosting, SVM)
- **Ансамблевые методы** для повышения точности

### Торговые стратегии
- AI-стратегии на основе прогнозов моделей
- Технические стратегии с индикаторами
- Гибридные стратегии (ИИ + технический анализ)
- Адаптивное управление портфелем стратегий

### Интеграция с внешними ИИ
- **OpenAI GPT** для анализа новостей и генерации инсайтов
- **Anthropic Claude** для альтернативного анализа
- **Локальные LLM** через Ollama
- Автоматический анализ рыночных настроений

### Управление рисками
- Динамическое управление размером позиций
- Автоматические стоп-лоссы и тейк-профиты
- Мониторинг максимальной просадки
- Диверсификация портфеля

### Источники данных
- **Yahoo Finance** для глобальных рынков
- **Tinkoff API** для российских активов
- **Alpha Vantage** для расширенных данных
- Интеграция с новостными API

## 🏗️ Архитектура

```
AI-Trading-Bot/
├── src/
│   ├── data/                 # Сбор и обработка данных
│   │   ├── data_collector.py     # Collectors для разных источников
│   │   └── technical_indicators.py # Технические индикаторы
│   │
│   ├── models/              # Модели машинного обучения
│   │   ├── arima.py             # ARIMA модели
│   │   ├── lstm.py              # LSTM нейронные сети
│   │   └── classifiers.py       # Классификаторы направления
│   │
│   ├── strategies/          # Торговые стратегии
│   │   └── ai_strategies.py     # ИИ и гибридные стратегии
│   │
│   ├── trading/             # Торговые модули
│   │   └── tinkoff_trader.py    # Интеграция с брокером
│   │
│   ├── backtesting/         # Система бэктестинга
│   │   └── backtester.py        # Движок бэктестинга
│   │
│   ├── monitoring/          # Мониторинг и метрики
│   │   └── metrics_tracker.py   # Отслеживание производительности
│   │
│   ├── ai_integration/      # Внешние ИИ сервисы
│   │   └── external_ai.py       # OpenAI, Claude, локальные LLM
│   │
│   └── utils/               # Вспомогательные модули
│       ├── config.py            # Управление конфигурацией
│       └── logger.py            # Логирование
│
├── config/                  # Конфигурационные файлы
├── models/                  # Обученные модели
├── logs/                    # Логи
├── data/                    # База данных метрик
└── backtest_results/        # Результаты бэктестов
```

## 💻 Установка

### 1. Клонирование репозитория
```bash
git clone <repository-url>
cd ai-trading-bot
```

### 2. Создание виртуального окружения
```bash
# Python 3.9+ рекомендуется
python -m venv venv

# Активация (Linux/Mac)
source venv/bin/activate

# Активация (Windows)
venv\Scripts\activate
```

### 3. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 4. Первоначальная настройка
```bash
python cli.py setup
```

## ⚙️ Настройка

### 1. Переменные окружения
Скопируйте `.env.example` в `.env` и заполните своими ключами:

```bash
cp .env.example .env
```

Отредактируйте `.env`:
```env
# Tinkoff API (для российского рынка)
TINKOFF_TOKEN=your_tinkoff_token_here

# OpenAI API (для анализа новостей)
OPENAI_API_KEY=your_openai_api_key_here

# Другие API ключи...
```

### 2. Конфигурация бота
Отредактируйте `config/config.yaml`:

```yaml
# Источник данных
data:
  source: "yahoo"  # yahoo, tinkoff

# Торговые настройки
trading:
  enabled: false  # true для реальной торговли
  broker: "tinkoff"

# Настройки моделей
models:
  training_days: 365
  use_arima: true
  use_lstm: true
  use_classifier: true

# Управление рисками
risk_management:
  initial_capital: 100000
  max_drawdown: 0.15
  stop_loss_pct: 0.02
```

### 3. Получение API ключей

#### Tinkoff Investments API
1. Зарегистрируйтесь в [Tinkoff Investments](https://www.tinkoff.ru/invest/)
2. Получите токен в настройках API
3. Начните с sandbox режима

#### OpenAI API (опционально)
1. Регистрация на [OpenAI Platform](https://platform.openai.com/)
2. Создание API ключа
3. Пополнение баланса для использования GPT-4

#### Другие API
- [News API](https://newsapi.org/) - для сбора новостей
- [Alpha Vantage](https://www.alphavantage.co/) - для расширенных данных

## 🎯 Использование

### CLI Интерфейс

#### Обучение моделей
```bash
# Обучить модели для конкретных инструментов
python cli.py train -s AAPL MSFT GOOGL

# Принудительное переобучение
python cli.py train -s AAPL --force
```

#### Бэктестинг
```bash
# Бэктест за год
python cli.py backtest -s AAPL MSFT --start-date 2023-01-01 --end-date 2023-12-31

# Кастомный период
python cli.py backtest -s SBER GAZP --start-date 2023-06-01 --end-date 2023-12-31
```

#### Анализ рынка
```bash
# Разовый анализ
python cli.py analyze -s AAPL TSLA

# Статус бота
python cli.py status
```

#### Запуск торговли
```bash
# Демо режим (без реальных сделок)
python cli.py start -s AAPL MSFT --demo

# Реальная торговля (ОСТОРОЖНО!)
python cli.py start -s AAPL MSFT
```

### Прямой запуск

#### Основной торговый бот
```bash
# Запуск в режиме анализа
python main.py -s AAPL MSFT GOOGL

# Бэктест
python main.py --backtest -s AAPL --start-date 2023-01-01

# Переобучение моделей
python main.py --train -s AAPL MSFT

# Статус системы
python main.py --status
```

## 🧠 Модели ИИ

### ARIMA Модели
- Автоматический подбор параметров
- Прогнозирование цен на 1-5 периодов
- Анализ трендов и сезонности

```python
# Пример конфигурации ARIMA
models:
  arima_auto_order: true  # Автоподбор (p,d,q)
  arima_seasonal: false   # Сезонная декомпозиция
```

### LSTM Нейронные сети
- Анализ последовательностей временных рядов
- Учет долгосрочных зависимостей
- Многомерные входные данные

```python
# Конфигурация LSTM
models:
  lstm_window_size: 30    # Окно данных
  lstm_units: 64          # Размер скрытого слоя
  lstm_epochs: 50         # Эпохи обучения
```

### Классификаторы направления
- Random Forest для интерпретируемости
- Gradient Boosting для точности
- Ансамблевые методы

```python
# Настройки классификации
models:
  classification_threshold: 0.01  # 1% для UP/DOWN
  confidence_threshold: 0.6       # Минимальная уверенность
```

## 📈 Торговые стратегии

### AI-стратегия
Принимает решения на основе прогнозов моделей:
- Взвешенное голосование между моделями
- Учет уверенности прогнозов
- Адаптивные пороги для сигналов

### Техническая стратегия
Использует классические индикаторы:
- RSI для определения перекупленности/перепроданности
- Пересечения скользящих средних
- Bollinger Bands для волатильности

### Гибридная стратегия
Комбинирует ИИ и технический анализ:
```yaml
strategies:
  ai_weight: 0.7          # Вес ИИ сигналов
  technical_weight: 0.3   # Вес технических индикаторов
```

### Управление портфелем стратегий
- Динамические веса стратегий
- Отключение неэффективных стратегий
- A/B тестирование новых подходов

## 🔬 Бэктестинг

### Возможности системы
- Точный учет комиссий и проскальзывания
- Walk-Forward анализ
- Множественные метрики производительности

### Метрики анализа
- **Доходность** (абсолютная и относительно buy-and-hold)
- **Коэффициент Шарпа** (риск-доходность)
- **Максимальная просадка**
- **Win Rate** (процент прибыльных сделок)
- **Profit Factor** (отношение прибыли к убыткам)

### Пример результатов
```json
{
  "total_return_pct": 12.5,
  "buy_hold_return_pct": 8.3,
  "max_drawdown_pct": 5.2,
  "sharpe_ratio": 1.8,
  "num_trades": 47
}
```

## 🔌 API Интеграции

### Интеграция с внешними ИИ

#### OpenAI GPT
```python
# Анализ настроений новостей
sentiment = ai_service.analyze_market_sentiment(news_texts, service='openai')

# Генерация торговых инсайтов
insights = ai_service.generate_trading_insights(market_data, service='openai')
```

#### Локальные LLM (Ollama)
```bash
# Установка Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Загрузка модели
ollama pull llama2

# Настройка в config.yaml
external_ai:
  local_llm:
    enabled: true
    model: "llama2"
```

### Брокерские API

#### Tinkoff Investments
```python
# Настройка sandbox
tinkoff:
  sandbox: true  # Тестовый режим
  
# Переключение на продакшн
tinkoff:
  sandbox: false  # ОСТОРОЖНО: Реальные деньги!
```

## 📊 Мониторинг

### Метрики в реальном времени
- P&L по сделкам и дням
- Эффективность стратегий
- Использование капитала
- Риск-метрики

### База данных метрик
```sql
-- Структура таблиц
trades          -- Все сделки
daily_metrics   -- Дневные показатели  
alerts          -- Системные уведомления
```

### Экспорт данных
```bash
# Экспорт метрик за последние 30 дней
python -c "
from src.monitoring.metrics_tracker import MetricsTracker
tracker = MetricsTracker()
tracker.export_data('metrics_export.json', days=30)
"
```

### Система алертов
- Превышение максимальной просадки
- Критические ошибки моделей
- Проблемы с подключением к API
- Аномальная активность

## 📁 Структура проекта

```
ai-trading-bot/
├── 📁 src/                    # Исходный код
│   ├── 📁 data/               # Модули данных
│   │   ├── 🐍 data_collector.py
│   │   ├── 🐍 technical_indicators.py
│   │   └── 🐍 __init__.py
│   │
│   ├── 📁 models/             # Модели ИИ
│   │   ├── 🐍 arima.py
│   │   ├── 🐍 lstm.py
│   │   ├── 🐍 classifiers.py
│   │   └── 🐍 __init__.py
│   │
│   ├── 📁 strategies/         # Торговые стратегии
│   │   ├── 🐍 ai_strategies.py
│   │   └── 🐍 __init__.py
│   │
│   ├── 📁 trading/            # Торговые модули
│   │   ├── 🐍 tinkoff_trader.py
│   │   └── 🐍 __init__.py
│   │
│   ├── 📁 backtesting/        # Бэктестинг
│   │   ├── 🐍 backtester.py
│   │   └── 🐍 __init__.py
│   │
│   ├── 📁 monitoring/         # Мониторинг
│   │   ├── 🐍 metrics_tracker.py
│   │   └── 🐍 __init__.py
│   │
│   ├── 📁 ai_integration/     # Внешние ИИ
│   │   ├── 🐍 external_ai.py
│   │   └── 🐍 __init__.py
│   │
│   └── 📁 utils/              # Утилиты
│       ├── 🐍 config.py
│       ├── 🐍 logger.py
│       └── 🐍 __init__.py
│
├── 📁 config/                 # Конфигурации
│   └── 📄 config.yaml
│
├── 📁 models/                 # Обученные модели (создается автоматически)
│   ├── 💾 AAPL_arima.pkl
│   ├── 💾 AAPL_lstm.pkl
│   └── 💾 AAPL_classifier.pkl
│
├── 📁 logs/                   # Логи (создается автоматически)
│   └── 📄 trading_bot.log
│
├── 📁 data/                   # База данных (создается автоматически)
│   └── 🗃️ metrics.db
│
├── 📁 backtest_results/       # Результаты бэктестов (создается автоматически)
│   └── 📊 AAPL_2023-01-01_2023-12-31.json
│
├── 🐍 main.py                 # Главный файл бота
├── 🐍 cli.py                  # CLI интерфейс  
├── 📄 requirements.txt        # Python зависимости
├── 📄 .env.example           # Шаблон переменных окружения
├── 📄 README.md              # Этот файл
└── 📄 .gitignore            # Исключения для Git
```

## ⚠️ Важные предупреждения

### Риски автоматической торговли
1. **Финансовые потери**: Торговля всегда связана с риском
2. **Технические сбои**: Проблемы с API, интернетом, сервером
3. **Переобучение моделей**: Исторические данные не гарантируют будущие результаты
4. **Рыночная волатильность**: Форс-мажорные события могут нарушить работу алгоритмов

### Рекомендации по безопасности
1. **Начните с демо-режима** и sandbox
2. **Ограничьте максимальную просадку** (не более 10-15%)
3. **Регулярно мониторьте** работу системы
4. **Диверсифицируйте** активы и стратегии
5. **Держите стоп-лоссы** на всех позициях

## 🛠️ Разработка и вклад

### Структура разработки
```bash
# Создание новой ветки
git checkout -b feature/new-strategy

# Тестирование изменений
python -m pytest tests/

# Форматирование кода  
black src/
flake8 src/

# Commit и push
git add .
git commit -m "Add new trading strategy"
git push origin feature/new-strategy
```

### Добавление новых моделей
1. Создайте класс в `src/models/`
2. Реализуйте методы `fit()`, `predict()`, `save_model()`, `load_model()`
3. Добавьте интеграцию в `main.py`
4. Создайте тесты

### Добавление новых стратегий  
1. Наследуйтесь от `BaseStrategy`
2. Реализуйте `generate_signal()`
3. Добавьте в `StrategyManager`
4. Протестируйте через бэктестинг

## 📚 Дополнительные ресурсы

### Обучающие материалы
- 📖 Книга "ИИ в Трейдинге" (основа проекта)
- 📺 [Курс по алгоритмическому трейдингу](https://www.coursera.org/specializations/trading-algorithms)
- 🎓 [Machine Learning for Trading](https://www.udacity.com/course/machine-learning-for-trading--ud501)

### Полезные ссылки
- [Tinkoff Invest API документация](https://tinkoff.github.io/investAPI/)
- [Alpha Vantage API](https://www.alphavantage.co/documentation/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Yahoo Finance API](https://python-yahoofinance.readthedocs.io/)

### Сообщества
- 🐛 [Issues и баг-репорты](../../issues)
- 💬 [Discussions для вопросов](../../discussions)
- 📱 Telegram: @ai_trading_community
- 🐦 Twitter: @ai_trading_bot

## 📄 Лицензия

MIT License - см. файл [LICENSE](LICENSE) для подробностей.

## ⭐ Поддержка проекта

Если проект оказался полезным:
- ⭐ Поставьте звезду на GitHub
- 🐛 Сообщите об ошибках в Issues  
- 💡 Предложите улучшения в Discussions
- 🔀 Сделайте форк и внесите свой вклад

---

**⚠️ Дисклеймер**: Данный проект предназначен только для образовательных целей. Авторы не несут ответственности за финансовые потери от использования данного ПО. Торговля финансовыми инструментами сопряжена с высоким риском потери средств.

#   A I - b o t _ L R D  
 