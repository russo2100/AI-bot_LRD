# Т-Банк ИИ Торговый Агент 🤖

## Обзор проекта

Автоматизированная торговая система для «Т-Банк Инвестиции» на основе искусственного интеллекта, созданная по методикам книги «ИИ в трейдинге» с использованием современных технологий 2025 года.

### ✨ Основные возможности

- 🔄 **Гибридные стратегии**: Комбинация технического анализа с LSTM прогнозами
- 🛡️ **Продвинутый риск-менеджмент**: Стоп-лосс, тейк-профит, динамическое управление капиталом  
- 📊 **Множество моделей**: ARIMA, LSTM/GRU, Random Forest, логистическая регрессия
- 🧪 **Полный цикл тестирования**: Backtesting, forward testing, walk-forward analysis
- 📈 **Мониторинг метрик**: Sharpe, Sortino, Max Drawdown в режиме реального времени
- 🌐 **API интеграция**: Официальный Python SDK «Т-Банк Инвестиции» v2
- 🎯 **Песочница**: Безопасное тестирование перед боевым режимом

---

## 🚀 Быстрый старт

### 1. Системные требования

```bash
# Операционная система: Windows 10+, macOS 10.15+, Linux Ubuntu 18.04+
# Python: 3.9-3.12 (рекомендуется 3.11)
# ОЗУ: минимум 8GB, рекомендуется 16GB
# Свободное место: 2GB для установки + место под данные
```

### 2. Установка Python

#### Windows:
```powershell
# Скачать с https://python.org/downloads/
# ✅ Обязательно отметить "Add Python to PATH"
# Проверить установку:
python --version
pip --version
```

#### macOS:
```bash
# Через Homebrew (рекомендуется)
brew install python@3.11

# Или скачать с python.org
# Проверить установку:
python3 --version
pip3 --version
```

#### Linux Ubuntu/Debian:
```bash
sudo apt update
sudo apt install python3.11 python3.11-pip python3.11-venv
# Создать алиасы (добавить в ~/.bashrc)
echo 'alias python=python3.11' >> ~/.bashrc
echo 'alias pip=pip3' >> ~/.bashrc
source ~/.bashrc
```

### 3. Клонирование и настройка проекта

```bash
# Создать директорию проекта
mkdir tbank_ai_trader
cd tbank_ai_trader

# Создать виртуальное окружение
python -m venv venv

# Активировать окружение
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Должна появиться приставка (venv) в терминале
```

### 4. Установка зависимостей

Создайте файл `requirements.txt`:
```txt
# Core trading and data
tinkoff-investments>=1.0.0
pandas>=2.2.0
numpy>=1.26.0

# Machine learning
scikit-learn>=1.5.0
statsmodels>=0.14.0
tensorflow>=2.15.0

# Optimization
optuna>=3.6.0

# Backtesting (optional)
backtrader>=1.9.78.123

# Utils
python-dotenv>=1.0.1
pyyaml>=6.0.1
requests>=2.31.0

# Visualization
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.18.0

# Logging and monitoring
loguru>=0.7.0
```

```bash
# Установить зависимости
pip install -r requirements.txt

# Проверить установку ключевых пакетов
python -c "import tinkoff.invest; import pandas; import sklearn; print('✅ Все пакеты установлены')"
```

### 5. Настройка токенов API

#### Получение токена Т-Банк:
1. Перейти на https://www.tbank.ru/invest/open-api
2. Войти в аккаунт Т-Банк
3. Создать токен для API
4. **Важно**: сначала создать токен для песочницы!

#### Создание .env файла:
```bash
# Скопировать пример конфигурации
cp .env.example .env

# Отредактировать .env (вставить свои токены)
nano .env  # Linux/macOS
notepad .env  # Windows
```

Содержимое `.env`:
```env
# API Токены (получить на https://www.tbank.ru/invest/open-api)
TINKOFF_TOKEN=t.ваш_токен_здесь
TINKOFF_ACCOUNT_ID=ваш_id_аккаунта

# Режим работы (ОБЯЗАТЕЛЬНО начать с true!)
SANDBOX=true

# Настройки логирования
LOG_LEVEL=INFO
LOG_FILE=logs/trading.log

# Настройки стратегий
DEFAULT_RISK_PER_TRADE=0.02
MAX_POSITIONS=5
```

### 6. Создание структуры проекта

```bash
# Создать все необходимые директории и файлы
mkdir -p {src/{data,features,models,strategies,backtesting,execution,monitoring},logs,data,config,tests,notebooks}

touch src/__init__.py
touch src/config.py
touch src/data/{__init__.py,collector_tbank.py,preprocess.py}
touch src/features/{__init__.py,indicators.py}
touch src/models/{__init__.py,arima.py,lstm.py,classifiers.py}
touch src/strategies/{__init__.py,ma_lstm.py,rsi_divergence.py}
touch src/backtesting/{__init__.py,simple_backtest.py}
touch src/execution/{__init__.py,broker_tbank.py}
touch src/monitoring/{__init__.py,metrics.py}
touch src/main.py
```

---

## 📁 Структура проекта

```
tbank_ai_trader/
├── 📄 README.md              # Документация
├── 📄 requirements.txt       # Python зависимости
├── 📄 .env                   # Переменные окружения (НЕ коммитить!)
├── 📄 .env.example          # Пример конфигурации
├── 📄 .gitignore            # Git игнорирует
│
├── 📁 src/                  # Основной код
│   ├── 📄 __init__.py
│   ├── 📄 config.py         # Конфигурация
│   ├── 📄 main.py           # CLI интерфейс
│   │
│   ├── 📁 data/             # Сбор и обработка данных
│   │   ├── 📄 __init__.py
│   │   ├── 📄 collector_tbank.py    # API Т-Банк
│   │   └── 📄 preprocess.py         # Очистка данных
│   │
│   ├── 📁 features/         # Технические индикаторы
│   │   ├── 📄 __init__.py
│   │   └── 📄 indicators.py         # SMA, RSI, MACD и др.
│   │
│   ├── 📁 models/           # ML модели
│   │   ├── 📄 __init__.py
│   │   ├── 📄 arima.py              # Статистические модели
│   │   ├── 📄 lstm.py               # Нейронные сети
│   │   └── 📄 classifiers.py        # Классификация
│   │
│   ├── 📁 strategies/       # Торговые стратегии
│   │   ├── 📄 __init__.py
│   │   ├── 📄 ma_lstm.py           # MA + LSTM гибрид
│   │   └── 📄 rsi_divergence.py    # RSI дивергенция
│   │
│   ├── 📁 backtesting/      # Тестирование стратегий
│   │   ├── 📄 __init__.py
│   │   └── 📄 simple_backtest.py   # Векторизованный бэктест
│   │
│   ├── 📁 execution/        # Исполнение ордеров
│   │   ├── 📄 __init__.py
│   │   └── 📄 broker_tbank.py      # API брокера
│   │
│   └── 📁 monitoring/       # Мониторинг и метрики
│       ├── 📄 __init__.py
│       └── 📄 metrics.py           # Sharpe, Sortino и др.
│
├── 📁 config/               # YAML конфиги
│   └── 📄 strategies.yaml   # Параметры стратегий
│
├── 📁 data/                 # Сохраненные данные
│   ├── 📁 raw/             # Сырые данные
│   ├── 📁 processed/       # Обработанные данные
│   └── 📁 models/          # Сохраненные модели
│
├── 📁 logs/                 # Лог файлы
│   ├── 📄 trading.log      # Основные логи
│   └── 📄 backtest.log     # Логи бэктестов
│
├── 📁 notebooks/            # Jupyter блокноты для анализа
│   ├── 📄 data_exploration.ipynb
│   ├── 📄 strategy_development.ipynb
│   └── 📄 model_training.ipynb
│
└── 📁 tests/                # Тесты
    ├── 📄 test_data.py
    ├── 📄 test_strategies.py
    └── 📄 test_models.py
```

---

## 🔧 Основные команды

### Сбор данных

```bash
# Получить исторические данные по акции на 365 дней (часовые свечи)
python src/main.py collect --figi BBG004730N88 --days 365 --interval hour --out data/raw/sber_1year.csv

# Доступные интервалы: 1min, 5min, 15min, hour, day
# FIGI популярных акций:
# SBER: BBG004730N88
# GAZP: BBG004730RP0  
# LKOH: BBG004731354
# YNDX: BBG00178PGX3
```

### Обучение моделей

```bash
# Обучить LSTM модель на данных (окно 30 дней)
python src/main.py train --in data/raw/sber_1year.csv --window 30 --out data/processed/sber_with_predictions.csv

# Модель автоматически добавит колонку 'lstm_pred' с прогнозами
```

### Запуск бэктеста

```bash
# Протестировать стратегию MA+LSTM на исторических данных
python src/main.py backtest --in data/processed/sber_with_predictions.csv --out data/results/backtest_results.csv

# Результаты включают: equity curve, метрики производительности, сигналы
```

### Торговля в песочнице

```bash
# ⚠️ ВАЖНО: убедитесь, что SANDBOX=true в .env файле!

# Купить 1 акцию Сбера по рыночной цене (песочница)
python src/main.py trade --figi BBG004730N88 --qty 1 --market --buy

# Продать 1 акцию по лимитной цене
python src/main.py trade --figi BBG004730N88 --qty 1 --buy=false

# Проверить состояние счёта
python -c "from src.execution.broker_tbank import TBankBroker; print(TBankBroker().get_accounts())"
```

---

## ⚙️ Детальная настройка

### 1. Настройка стратегий (config/strategies.yaml)

```yaml
# Гибридная стратегия MA + LSTM
ma_lstm_strategy:
  # Параметры скользящих средних
  fast_ma: 10
  slow_ma: 30
  
  # LSTM настройки
  lstm_window: 30
  lstm_units: 64
  lstm_epochs: 10
  
  # Управление рисками
  stop_loss_pct: 0.02     # 2% стоп-лосс
  take_profit_pct: 0.04   # 4% тейк-профит  
  max_risk_per_trade: 0.02 # 2% от капитала на сделку
  
  # Фильтры
  min_volume: 1000000     # Минимальный объём торгов
  volatility_filter: true  # Фильтр по волатильности

# RSI дивергенция стратегия  
rsi_divergence_strategy:
  rsi_period: 14
  divergence_lookback: 20
  min_divergence_strength: 0.7
  stop_loss_pct: 0.015
  take_profit_pct: 0.035
```

### 2. Настройка логирования

```python
# Добавить в src/config.py
import logging
from loguru import logger
import sys

def setup_logging(level="INFO", log_file="logs/trading.log"):
    # Удалить стандартный handler
    logger.remove()
    
    # Консольный вывод с цветами
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True
    )
    
    # Файловый вывод 
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation="10 MB",
        retention="30 days"
    )
```

### 3. Мониторинг производительности

```python
# src/monitoring/dashboard.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def create_performance_dashboard(backtest_results: pd.DataFrame):
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Equity Curve', 'Drawdown',
            'Monthly Returns', 'Trade Distribution', 
            'Rolling Sharpe', 'Position Sizing'
        ],
        vertical_spacing=0.08
    )
    
    # Equity Curve
    fig.add_trace(
        go.Scatter(
            x=backtest_results.index,
            y=backtest_results['equity'],
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # И другие графики...
    
    fig.update_layout(
        title="📊 Торговая Производительность",
        height=800,
        showlegend=True
    )
    
    return fig
```

---

## 🧪 Процедура тестирования

### 1. Полный цикл тестирования

```bash
# Этап 1: Сбор данных (минимум 2 года для надежности)
python src/main.py collect --figi BBG004730N88 --days 730 --interval hour --out data/raw/sber_2years.csv

# Этап 2: Разделение на train/test (80%/20%)
python -c "
import pandas as pd
df = pd.read_csv('data/raw/sber_2years.csv', parse_dates=['time'], index_col='time')
split_idx = int(len(df) * 0.8)
df[:split_idx].to_csv('data/raw/sber_train.csv')
df[split_idx:].to_csv('data/raw/sber_test.csv')
print(f'Train: {len(df[:split_idx])}, Test: {len(df[split_idx:])}')
"

# Этап 3: Обучение на тренировочных данных
python src/main.py train --in data/raw/sber_train.csv --window 30 --out data/processed/sber_train_pred.csv

# Этап 4: Бэктест на тренировочных данных (in-sample)
python src/main.py backtest --in data/processed/sber_train_pred.csv --out data/results/backtest_train.csv

# Этап 5: Forward test на тестовых данных (out-of-sample)  
python src/main.py train --in data/raw/sber_test.csv --window 30 --out data/processed/sber_test_pred.csv
python src/main.py backtest --in data/processed/sber_test_pred.csv --out data/results/backtest_test.csv

# Этап 6: Сравнение результатов
python -c "
import pandas as pd
from src.monitoring.metrics import summary
train_results = pd.read_csv('data/results/backtest_train.csv', parse_dates=['time'], index_col='time')
test_results = pd.read_csv('data/results/backtest_test.csv', parse_dates=['time'], index_col='time')

print('📈 ТРЕНИРОВОЧНЫЕ ДАННЫЕ (In-Sample):')
print(summary(train_results))
print('\n📊 ТЕСТОВЫЕ ДАННЫЕ (Out-of-Sample):') 
print(summary(test_results))
"
```

### 2. Walk-Forward анализ

```python
# src/backtesting/walk_forward.py
def walk_forward_analysis(data, window_size=252, step_size=21):
    """
    Проводит walk-forward анализ:
    - Обучает модель на window_size дней
    - Тестирует на следующих step_size днях  
    - Сдвигает окно и повторяет
    """
    results = []
    
    for start_idx in range(0, len(data) - window_size - step_size, step_size):
        # Окно обучения
        train_end = start_idx + window_size
        test_start = train_end
        test_end = test_start + step_size
        
        train_data = data.iloc[start_idx:train_end]
        test_data = data.iloc[test_start:test_end]
        
        # Обучение модели
        model = train_model(train_data)
        
        # Тестирование
        predictions = model.predict(test_data)
        performance = calculate_metrics(test_data, predictions)
        
        results.append({
            'period': f"{test_data.index[0]} to {test_data.index[-1]}",
            'sharpe': performance['sharpe'],
            'return': performance['total_return'],
            'max_dd': performance['max_drawdown']
        })
    
    return pd.DataFrame(results)

# Запуск walk-forward анализа
wf_results = walk_forward_analysis(data, window_size=252, step_size=21)
print(wf_results.describe())
```

---

## 🎯 Переход к боевому режиму

### ⚠️ Контрольный список перед запуском

- [ ] **Тестирование завершено**: Все стратегии показывают стабильную прибыльность на out-of-sample данных
- [ ] **Метрики приемлемы**: Sharpe > 1.0, Max Drawdown < 20%, Win Rate > 40%
- [ ] **Код протестирован**: Все unit-тесты проходят, нет критических ошибок
- [ ] **Связь стабильна**: API подключение работает без перебоев 
- [ ] **Мониторинг настроен**: Уведомления о критических ошибках и превышении DD
- [ ] **Капитал выделен**: Торгуете только деньгами, которые можете позволить себе потерять
- [ ] **Стоп-лоссы установлены**: Максимальная просадка ограничена на уровне портфеля

### Переход на боевой контур:

```bash
# 1. Изменить .env файл
SANDBOX=false  # ⚠️ ВНИМАНИЕ: теперь реальные деньги!

# 2. Начать с минимальных позиций
python src/main.py trade --figi BBG004730N88 --qty 1 --market --buy

# 3. Мониторить каждую сделку
tail -f logs/trading.log

# 4. Настроить автоматические уведомления при критических событиях
```

---

## 🛠️ Продвинутые возможности

### 1. Оптимизация гиперпараметров

```python
# src/optimization/hyperopt.py
import optuna
from src.strategies.ma_lstm import ma_cross_with_lstm_confirmation
from src.backtesting.simple_backtest import backtest

def optimize_strategy(data, n_trials=100):
    def objective(trial):
        # Параметры для оптимизации
        fast_ma = trial.suggest_int('fast_ma', 5, 20)
        slow_ma = trial.suggest_int('slow_ma', 20, 50) 
        lstm_window = trial.suggest_int('lstm_window', 10, 60)
        stop_loss = trial.suggest_float('stop_loss_pct', 0.01, 0.05)
        take_profit = trial.suggest_float('take_profit_pct', 0.02, 0.08)
        
        # Генерация сигналов
        signals = ma_cross_with_lstm_confirmation(
            data, fast=fast_ma, slow=slow_ma, 
            lstm_pred_col='lstm_pred'
        )
        
        # Бэктест
        results = backtest(
            data, signals, 
            sl_pct=stop_loss, tp_pct=take_profit
        )
        
        # Целевая метрика (Sharpe ratio)
        returns = results['strategy_ret']
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        
        return sharpe
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params

# Запуск оптимизации
best_params = optimize_strategy(data, n_trials=200)
print(f"Лучшие параметры: {best_params}")
```

### 2. Ансамблевые стратегии

```python
# src/strategies/ensemble.py
class EnsembleStrategy:
    def __init__(self):
        self.strategies = [
            'ma_lstm',
            'rsi_divergence', 
            'mean_reversion',
            'breakout_momentum'
        ]
        self.weights = [0.3, 0.25, 0.25, 0.2]  # Веса стратегий
    
    def generate_signals(self, data):
        individual_signals = {}
        
        # Получить сигналы от каждой стратегии
        for strategy in self.strategies:
            individual_signals[strategy] = self.run_strategy(strategy, data)
        
        # Взвешенная комбинация сигналов
        ensemble_signal = sum(
            individual_signals[strategy] * weight 
            for strategy, weight in zip(self.strategies, self.weights)
        )
        
        # Преобразование в дискретные сигналы (-1, 0, 1)
        signals = pd.Series(index=data.index, dtype=int)
        signals[ensemble_signal > 0.5] = 1
        signals[ensemble_signal < -0.5] = -1
        signals = signals.fillna(0)
        
        return signals
```

### 3. Sentiment анализ (расширение для новостей)

```python
# src/features/sentiment.py
import requests
from textblob import TextBlob
import yfinance as yf

def get_stock_news_sentiment(ticker, days=7):
    """Получить настроение из новостей по тикеру"""
    try:
        # Получить новости через yfinance
        stock = yf.Ticker(ticker)
        news = stock.news
        
        sentiments = []
        for article in news[:10]:  # Последние 10 новостей
            text = article.get('title', '') + ' ' + article.get('summary', '')
            sentiment = TextBlob(text).sentiment.polarity
            sentiments.append(sentiment)
        
        # Средний sentiment: от -1 (негативный) до +1 (позитивный)
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        return avg_sentiment
        
    except Exception as e:
        print(f"Ошибка получения новостей: {e}")
        return 0
```

---

## 🔍 Отладка и решение проблем

### Частые проблемы и решения

#### 1. Ошибка подключения к API
```bash
# Проверить интернет соединение
ping google.com

# Проверить токен
python -c "
from src.execution.broker_tbank import TBankBroker
try:
    broker = TBankBroker()
    accounts = broker.get_accounts()
    print(f'✅ API работает. Найдено счетов: {len(accounts)}')
except Exception as e:
    print(f'❌ Ошибка API: {e}')
"
```

#### 2. Недостаточно данных для модели
```python
# Проверить размер данных
import pandas as pd
df = pd.read_csv('data/raw/your_data.csv')
print(f"Строк данных: {len(df)}")
print(f"Минимум нужно: 100 для ARIMA, 500 для LSTM")

# Увеличить период сбора данных если необходимо
python src/main.py collect --figi BBG004730N88 --days 1000 --interval hour --out data/raw/more_data.csv
```

#### 3. Переобучение модели
```python
# Проверить разность между train/test производительностью
train_sharpe = 2.5  # Отлично на тренировочных данных
test_sharpe = 0.3   # Плохо на тестовых данных

if train_sharpe - test_sharpe > 1.0:
    print("⚠️ Возможное переобучение!")
    print("Решения:")
    print("- Упростить модель (меньше параметров)")
    print("- Увеличить regularization") 
    print("- Больше данных для обучения")
    print("- Использовать cross-validation")
```

### Мониторинг в реальном времени

```python
# src/monitoring/real_time.py
import time
import pandas as pd
from datetime import datetime

class RealTimeMonitor:
    def __init__(self):
        self.start_time = datetime.now()
        self.trades_today = 0
        self.pnl_today = 0.0
        self.max_drawdown_today = 0.0
        
    def update_stats(self, new_trade_pnl):
        self.trades_today += 1
        self.pnl_today += new_trade_pnl
        
        # Обновить максимальную просадку
        if new_trade_pnl < 0:
            self.max_drawdown_today = min(
                self.max_drawdown_today, 
                new_trade_pnl
            )
        
        # Проверить лимиты риска
        self.check_risk_limits()
    
    def check_risk_limits(self):
        # Остановить торговлю при превышении дневных лимитов
        if self.max_drawdown_today < -0.05:  # -5% дневная просадка
            print("🛑 КРИТИЧНО: Превышен лимит дневной просадки!")
            self.emergency_stop()
            
        if self.trades_today > 50:  # Слишком много сделок
            print("⚠️ Предупреждение: Слишком активная торговля")
    
    def emergency_stop(self):
        # Закрыть все позиции и остановить торговлю
        broker = TBankBroker()
        broker.cancel_all_orders()
        print("🔴 Экстренная остановка торговли активирована")

# Использование в основном цикле торговли
monitor = RealTimeMonitor()
```

---

## 📚 Дополнительные ресурсы

### Книги для изучения
1. **"Hands-On Machine Learning"** - Aurélien Géron (основы ML)
2. **"Algorithmic Trading"** - Ernest P. Chan (стратегии)  
3. **"Python for Finance"** - Yves Hilpisch (техническая часть)
4. **"Quantitative Portfolio Management"** - Michael Isichenko (портфельная теория)

### Онлайн курсы
- **Coursera**: Machine Learning for Trading (Georgia Tech)
- **edX**: Algorithmic Trading and Finance Models (MIT)
- **YouTube**: Sentdex Python Programming for Finance

### Полезные сайты и форумы
- **QuantConnect Community**: https://www.quantconnect.com/forum
- **Reddit**: r/algotrading, r/SecurityAnalysis  
- **GitHub**: awesome-quant репозиторий
- **Papers**: arxiv.org/list/q-fin/recent

### API документация
- **Т-Банк API**: https://russianinvestments.github.io/invest-python/
- **TensorFlow**: https://www.tensorflow.org/api_docs
- **Pandas**: https://pandas.pydata.org/docs/

---

## 🤝 Поддержка и развитие

### Как получить помощь

1. **Проверьте логи**: `tail -f logs/trading.log`  
2. **Изучите ошибку**: Большинство ошибок описаны в этом README
3. **Тестируйте поэтапно**: Проверьте каждый модуль отдельно
4. **Используйте песочницу**: Никогда не тестируйте на реальных деньгах

### Развитие проекта

Планируемые улучшения:
- [ ] Поддержка криптовалют через Binance API
- [ ] Веб-интерфейс для мониторинга
- [ ] Интеграция с TradingView для сигналов
- [ ] Дополнительные стратегии (статистический арбитраж)
- [ ] Автоматическое переобучение моделей
- [ ] Интеграция с Telegram для уведомлений

---

## ⚖️ Отказ от ответственности

**⚠️ ВАЖНОЕ ПРЕДУПРЕЖДЕНИЕ:**

Этот торговый бот создан исключительно в образовательных целях. Торговля на финансовых рынках связана с высокими рисками и может привести к потере капитала. 

**Перед использованием:**
- Тщательно протестируйте все стратегии в песочнице
- Никогда не инвестируйте деньги, которые не можете позволить себе потерять  
- Изучите основы управления рисками и портфельной теории
- Проконсультируйтесь с финансовыми консультантами

**Авторы не несут ответственности** за любые финансовые потери, возникшие в результате использования данного программного обеспечения.

---

## 📄 Лицензия

MIT License - свободное использование с указанием авторства.

---

*Создано на основе книги "ИИ в трейдинге: обучение и анализ" с использованием современных технологий и лучших практик 2025 года.*