# src/backtesting/backtester.py
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Движок для бэктестинга торговых стратегий
    Реализует подходы из книги "ИИ в трейдинге"
    """

    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        """
        Args:
            initial_capital: Начальный капитал
            commission: Комиссия за сделку (доля от оборота)
        """
        self.initial_capital = initial_capital
        self.commission = commission

        # Состояние портфеля
        self.capital = initial_capital
        self.position = 0  # Количество акций в портфеле
        self.cash = initial_capital

        # История торговли
        self.trades = []
        self.portfolio_history = []
        self.signals_history = []

        # Метрики
        self.max_capital = initial_capital
        self.max_drawdown = 0.0

    def run_backtest(self, data: pd.DataFrame, strategy, 
                    start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        Запуск бэктеста стратегии

        Args:
            data: Исторические данные
            strategy: Торговая стратегия
            start_date: Дата начала (опционально)
            end_date: Дата окончания (опционально)

        Returns:
            Результаты бэктеста
        """
        logger.info(f"🔬 Запуск бэктеста стратегии {strategy.name}")

        # Фильтрация данных по датам
        test_data = data.copy()
        if start_date:
            test_data = test_data[test_data.index >= start_date]
        if end_date:
            test_data = test_data[test_data.index <= end_date]

        logger.info(f"📊 Период тестирования: {test_data.index[0]} - {test_data.index[-1]}")
        logger.info(f"📊 Количество периодов: {len(test_data)}")

        # Сброс состояния
        self._reset_state()

        # Основной цикл бэктестинга
        for i in range(len(test_data)):
            current_timestamp = test_data.index[i]
            current_price = test_data['close'].iloc[i]

            # Генерация сигнала стратегией
            if i > 0:  # Нужна история для генерации сигналов
                try:
                    signal = strategy.generate_signal(test_data, i)
                    self._process_signal(signal, current_price, current_timestamp)
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка генерации сигнала на {current_timestamp}: {e}")
                    continue

            # Обновление состояния портфеля
            self._update_portfolio(current_price, current_timestamp)

        # Финальная очистка - закрываем открытые позиции
        if self.position != 0:
            final_price = test_data['close'].iloc[-1]
            final_timestamp = test_data.index[-1]
            self._close_all_positions(final_price, final_timestamp)

        # Расчет метрик
        results = self._calculate_metrics(test_data)

        logger.info(f"✅ Бэктест завершен. Итоговый капитал: {self.capital:.2f}")
        logger.info(f"📊 Доходность: {((self.capital/self.initial_capital - 1) * 100):.2f}%")

        return results

    def _reset_state(self):
        """Сброс состояния для нового бэктеста"""
        self.capital = self.initial_capital
        self.position = 0
        self.cash = self.initial_capital
        self.trades = []
        self.portfolio_history = []
        self.signals_history = []
        self.max_capital = self.initial_capital
        self.max_drawdown = 0.0

    def _process_signal(self, signal, price: float, timestamp):
        """Обработка торгового сигнала"""
        from src.strategies.ai_strategies import Signal

        if signal == Signal.BUY and self.position <= 0:
            self._execute_buy_order(price, timestamp)
        elif signal == Signal.SELL and self.position >= 0:
            self._execute_sell_order(price, timestamp)
        # Signal.HOLD - не делаем ничего

        self.signals_history.append({
            'timestamp': timestamp,
            'signal': signal.name,
            'price': price,
            'position_before': self.position,
            'cash_before': self.cash
        })

    def _execute_buy_order(self, price: float, timestamp):
        """Исполнение ордера на покупку"""
        if self.cash <= 0:
            return

        # Рассчитываем максимальное количество акций
        max_shares = int(self.cash / (price * (1 + self.commission)))

        if max_shares > 0:
            # Если у нас была короткая позиция, сначала закрываем её
            if self.position < 0:
                shares_to_cover = min(max_shares, abs(self.position))
                self._cover_short_position(shares_to_cover, price, timestamp)
                max_shares -= shares_to_cover

            # Покупаем акции, если ещё остался cash
            if max_shares > 0 and self.cash > 0:
                cost = max_shares * price * (1 + self.commission)

                if cost <= self.cash:
                    self.position += max_shares
                    self.cash -= cost

                    trade = {
                        'timestamp': timestamp,
                        'action': 'BUY',
                        'shares': max_shares,
                        'price': price,
                        'cost': cost,
                        'commission': max_shares * price * self.commission,
                        'cash_after': self.cash,
                        'position_after': self.position
                    }
                    self.trades.append(trade)

                    logger.debug(f"🟢 BUY: {max_shares} акций по {price:.4f}, осталось cash: {self.cash:.2f}")

    def _execute_sell_order(self, price: float, timestamp):
        """Исполнение ордера на продажу"""
        if self.position > 0:
            # Продаём все длинные позиции
            self._sell_long_position(self.position, price, timestamp)

        # Открываем короткую позицию на оставшийся cash
        max_short_shares = int(self.cash / (price * (1 + self.commission)))

        if max_short_shares > 0:
            proceeds = max_short_shares * price * (1 - self.commission)

            self.position -= max_short_shares
            self.cash += proceeds

            trade = {
                'timestamp': timestamp,
                'action': 'SELL_SHORT',
                'shares': max_short_shares,
                'price': price,
                'proceeds': proceeds,
                'commission': max_short_shares * price * self.commission,
                'cash_after': self.cash,
                'position_after': self.position
            }
            self.trades.append(trade)

            logger.debug(f"🔴 SELL_SHORT: {max_short_shares} акций по {price:.4f}, cash: {self.cash:.2f}")

    def _sell_long_position(self, shares: int, price: float, timestamp):
        """Продажа длинной позиции"""
        proceeds = shares * price * (1 - self.commission)

        self.position -= shares
        self.cash += proceeds

        trade = {
            'timestamp': timestamp,
            'action': 'SELL',
            'shares': shares,
            'price': price,
            'proceeds': proceeds,
            'commission': shares * price * self.commission,
            'cash_after': self.cash,
            'position_after': self.position
        }
        self.trades.append(trade)

    def _cover_short_position(self, shares: int, price: float, timestamp):
        """Закрытие короткой позиции"""
        cost = shares * price * (1 + self.commission)

        self.position += shares
        self.cash -= cost

        trade = {
            'timestamp': timestamp,
            'action': 'COVER',
            'shares': shares,
            'price': price,
            'cost': cost,
            'commission': shares * price * self.commission,
            'cash_after': self.cash,
            'position_after': self.position
        }
        self.trades.append(trade)

    def _close_all_positions(self, price: float, timestamp):
        """Закрытие всех открытых позиций в конце бэктеста"""
        if self.position > 0:
            self._sell_long_position(self.position, price, timestamp)
        elif self.position < 0:
            self._cover_short_position(abs(self.position), price, timestamp)

    def _update_portfolio(self, current_price: float, timestamp):
        """Обновление стоимости портфеля"""
        portfolio_value = self.cash + self.position * current_price

        portfolio_record = {
            'timestamp': timestamp,
            'price': current_price,
            'cash': self.cash,
            'position': self.position,
            'portfolio_value': portfolio_value,
            'return': (portfolio_value / self.initial_capital - 1) * 100
        }
        self.portfolio_history.append(portfolio_record)

        # Обновление максимального капитала и просадки
        if portfolio_value > self.max_capital:
            self.max_capital = portfolio_value

        current_drawdown = (self.max_capital - portfolio_value) / self.max_capital
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

        self.capital = portfolio_value

    def _calculate_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Расчет метрик производительности"""
        if not self.portfolio_history:
            return {}

        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df.set_index('timestamp', inplace=True)

        # Базовые метрики
        total_return = (self.capital / self.initial_capital - 1) * 100

        # Доходность buy-and-hold (бенчмарк)
        initial_price = data['close'].iloc[0]
        final_price = data['close'].iloc[-1]
        buy_hold_return = (final_price / initial_price - 1) * 100

        # Волатильность доходности
        returns = portfolio_df['return'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0  # Аннуализированная

        # Коэффициент Шарпа
        risk_free_rate = 0.02  # 2% годовых
        if volatility > 0:
            sharpe_ratio = (total_return / 100 - risk_free_rate) / (volatility / 100)
        else:
            sharpe_ratio = 0

        # Коэффициент Сортино
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_volatility = negative_returns.std() * np.sqrt(252)
            sortino_ratio = (total_return / 100 - risk_free_rate) / (downside_volatility / 100)
        else:
            sortino_ratio = float('inf') if total_return > risk_free_rate * 100 else 0

        # Метрики торговли
        num_trades = len(self.trades)

        # Анализ сделок
        if num_trades > 0:
            buy_trades = [t for t in self.trades if t['action'] in ['BUY', 'COVER']]
            sell_trades = [t for t in self.trades if t['action'] in ['SELL', 'SELL_SHORT']]

            total_commission = sum(t.get('commission', 0) for t in self.trades)
        else:
            total_commission = 0

        metrics = {
            # Основные метрики
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return_pct': total_return,
            'buy_hold_return_pct': buy_hold_return,
            'excess_return_pct': total_return - buy_hold_return,

            # Риски
            'max_drawdown_pct': self.max_drawdown * 100,
            'volatility_pct': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,

            # Торговля
            'num_trades': num_trades,
            'total_commission': total_commission,
            'commission_pct_of_capital': (total_commission / self.initial_capital) * 100,

            # Временные метрики
            'start_date': data.index[0].strftime('%Y-%m-%d'),
            'end_date': data.index[-1].strftime('%Y-%m-%d'),
            'days_tested': len(data),

            # Данные для анализа
            'portfolio_history': self.portfolio_history,
            'trades': self.trades,
            'signals_history': self.signals_history
        }

        return metrics

    def save_results(self, results: Dict[str, Any], filepath: str):
        """Сохранение результатов бэктеста"""
        try:
            # Подготавливаем данные для сериализации
            serializable_results = results.copy()

            # Конвертируем pandas timestamp в строки
            if 'portfolio_history' in serializable_results:
                for record in serializable_results['portfolio_history']:
                    if 'timestamp' in record:
                        record['timestamp'] = record['timestamp'].isoformat()

            if 'trades' in serializable_results:
                for trade in serializable_results['trades']:
                    if 'timestamp' in trade:
                        trade['timestamp'] = trade['timestamp'].isoformat()

            if 'signals_history' in serializable_results:
                for signal in serializable_results['signals_history']:
                    if 'timestamp' in signal:
                        signal['timestamp'] = signal['timestamp'].isoformat()

            # Сохраняем в JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)

            logger.info(f"💾 Результаты бэктеста сохранены: {filepath}")

        except Exception as e:
            logger.error(f"❌ Ошибка сохранения результатов: {e}")


class WalkForwardAnalysis:
    """
    Анализ с переобучением (Walk-Forward Analysis)
    Важный метод валидации из книги
    """

    def __init__(self, training_period_months: int = 12, 
                 testing_period_months: int = 1,
                 step_months: int = 1):
        """
        Args:
            training_period_months: Период обучения в месяцах
            testing_period_months: Период тестирования в месяцах  
            step_months: Шаг сдвига в месяцах
        """
        self.training_period = training_period_months
        self.testing_period = testing_period_months
        self.step_months = step_months

    def run_analysis(self, data: pd.DataFrame, strategy_factory, 
                    initial_capital: float = 100000) -> Dict[str, Any]:
        """
        Запуск Walk-Forward анализа

        Args:
            data: Исторические данные
            strategy_factory: Функция создания и обучения стратегии
            initial_capital: Начальный капитал
        """
        logger.info("🔄 Запуск Walk-Forward анализа")

        results = {
            'periods': [],
            'total_return': 0,
            'win_rate': 0,
            'avg_monthly_return': 0
        }

        # Определяем периоды для анализа
        start_date = data.index[0]
        end_date = data.index[-1]

        current_date = start_date
        period_results = []

        while current_date < end_date:
            # Определяем границы обучающего периода
            train_start = current_date
            train_end = train_start + pd.DateOffset(months=self.training_period)

            # Определяем границы тестового периода
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.testing_period)

            if test_end > end_date:
                break

            # Данные для обучения и тестирования
            train_data = data[(data.index >= train_start) & (data.index < train_end)]
            test_data = data[(data.index >= test_start) & (data.index < test_end)]

            if len(train_data) < 30 or len(test_data) < 5:
                current_date += pd.DateOffset(months=self.step_months)
                continue

            try:
                # Создание и обучение стратегии
                strategy = strategy_factory(train_data)

                # Бэктест на тестовых данных
                backtester = BacktestEngine(initial_capital=initial_capital)
                period_result = backtester.run_backtest(test_data, strategy)

                period_info = {
                    'train_start': train_start.strftime('%Y-%m-%d'),
                    'train_end': train_end.strftime('%Y-%m-%d'),
                    'test_start': test_start.strftime('%Y-%m-%d'),
                    'test_end': test_end.strftime('%Y-%m-%d'),
                    'return_pct': period_result.get('total_return_pct', 0),
                    'max_drawdown_pct': period_result.get('max_drawdown_pct', 0),
                    'num_trades': period_result.get('num_trades', 0),
                    'sharpe_ratio': period_result.get('sharpe_ratio', 0)
                }

                period_results.append(period_info)

                logger.info(f"📊 Период {test_start.strftime('%Y-%m')}: {period_info['return_pct']:.2f}%")

            except Exception as e:
                logger.error(f"❌ Ошибка в периоде {current_date}: {e}")

            # Переходим к следующему периоду
            current_date += pd.DateOffset(months=self.step_months)

        # Агрегированные результаты
        if period_results:
            returns = [p['return_pct'] for p in period_results]

            results.update({
                'periods': period_results,
                'total_periods': len(period_results),
                'total_return': sum(returns),
                'avg_monthly_return': np.mean(returns),
                'win_rate': sum(1 for r in returns if r > 0) / len(returns),
                'best_period': max(returns),
                'worst_period': min(returns),
                'volatility': np.std(returns),
                'sharpe_ratio': np.mean([p['sharpe_ratio'] for p in period_results])
            })

        logger.info(f"✅ Walk-Forward анализ завершен. Средняя доходность: {results.get('avg_monthly_return', 0):.2f}%/месяц")

        return results


if __name__ == "__main__":
    # Тестирование бэктестера
    print("🧪 Тестирование системы бэктестинга")

    # Создаем тестовые данные
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')  # Год торговых дней

    # Имитируем реалистичные цены
    returns = np.random.randn(252) * 0.02  # 2% дневная волатильность
    prices = 100 * (1 + returns).cumprod()

    test_data = pd.DataFrame({
        'close': prices,
        'volume': np.random.randint(1000, 10000, 252),
        'rsi': np.random.uniform(20, 80, 252),
    }, index=dates)

    print(f"📊 Тестовые данные: {len(test_data)} дней")

    # Простая стратегия для теста
    from src.strategies.ai_strategies import TechnicalStrategy

    strategy = TechnicalStrategy({
        'rsi_buy_threshold': 30,
        'rsi_sell_threshold': 70
    })

    # Запуск бэктеста
    print("🔬 Запуск бэктеста...")
    backtester = BacktestEngine(initial_capital=100000, commission=0.001)

    try:
        results = backtester.run_backtest(test_data, strategy)

        print(f"✅ Бэктест завершен!")
        print(f"📊 Начальный капитал: ${results['initial_capital']:,.2f}")
        print(f"📊 Итоговый капитал: ${results['final_capital']:,.2f}")
        print(f"📊 Доходность: {results['total_return_pct']:.2f}%")
        print(f"📊 Buy & Hold доходность: {results['buy_hold_return_pct']:.2f}%")
        print(f"📊 Максимальная просадка: {results['max_drawdown_pct']:.2f}%")
        print(f"📊 Коэффициент Шарпа: {results['sharpe_ratio']:.2f}")
        print(f"📊 Количество сделок: {results['num_trades']}")

        # Сохранение результатов
        backtester.save_results(results, 'backtest_results.json')
        print("💾 Результаты сохранены в backtest_results.json")

    except Exception as e:
        print(f"❌ Ошибка бэктеста: {e}")
        import traceback
        traceback.print_exc()

    print("✅ Тестирование бэктестера завершено!")
