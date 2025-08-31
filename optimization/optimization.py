#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оптимизация стратегий AI-торгового бота
A/B тестирование стратегий и Optuna оптимизация гиперпараметров
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass
import threading
import time

import pandas as pd
import numpy as np

from config.settings import settings

# Проверка наличия Optuna
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna не установлен, оптимизация гиперпараметров недоступна")

logger = logging.getLogger(__name__)


@dataclass
class ABTestResult:
    """Результаты A/B тестирования"""
    strategy_a: str
    strategy_b: str
    winner: str
    confidence_level: float
    performance_diff: float
    test_duration: timedelta
    sample_size_a: int
    sample_size_b: int
    metrics: Dict[str, Dict[str, float]]


@dataclass
class OptimizationResult:
    """Результаты оптимизации Optuna"""
    strategy_name: str
    best_params: Dict[str, Any]
    best_value: float
    trials_count: int
    optimization_time: timedelta
    convergence: bool


class ABTestingFramework:
    """Фреймворк для A/B тестирования стратегий"""

    def __init__(self):
        self.test_history = []
        self.active_tests = []
        self.confidence_threshold = 0.95
        self.min_sample_size = 100

    def start_ab_test(self, strategy_a: str, strategy_b: str,
                     test_duration_hours: int = 24) -> str:
        """Запуск A/B тестирования двух стратегий"""
        test_id = f"ab_test_{int(datetime.now().timestamp())}"

        test_config = {
            'test_id': test_id,
            'strategy_a': strategy_a,
            'strategy_b': strategy_b,
            'start_time': datetime.now(),
            'duration': timedelta(hours=test_duration_hours),
            'results_a': {'trades': [], 'pnl': 0.0, 'win_rate': 0.0},
            'results_b': {'trades': [], 'pnl': 0.0, 'win_rate': 0.0},
            'status': 'running'
        }

        self.active_tests.append(test_config)

        logger.info(f"Started A/B test: {strategy_a} vs {strategy_b}")
        logger.info(f"Test duration: {test_duration_hours} hours")

        # Запуск автоматического завершения
        timer = threading.Timer(test_duration_hours * 3600, self._complete_ab_test, args=[test_id])
        timer.start()

        return test_id

    def record_trade(self, test_id: str, strategy: str,
                    trade_result: Dict[str, Any]):
        """Запись результата сделки для тестирования"""
        for test in self.active_tests:
            if test['test_id'] == test_id and test['status'] == 'running':
                results_key = f"results_{'a' if strategy == test['strategy_a'] else 'b'}"
                test[results_key]['trades'].append(trade_result)
                break

    def _complete_ab_test(self, test_id: str):
        """Завершение A/B тестирования"""
        test = self._find_active_test(test_id)
        if not test:
            return

        test['status'] = 'completed'
        test['end_time'] = datetime.now()

        # Анализ результатов
        results = self._analyze_ab_test(test)
        results.test_duration = test['end_time'] - test['start_time']

        self.test_history.append(results)

        logger.info(f"A/B test completed: {results.winner} wins with {results.confidence_level:.2f} confidence")

        # Рекомендация по стратегии
        self._apply_recommendation(results)

    def _analyze_ab_test(self, test_config: dict) -> ABTestResult:
        """Анализ результатов A/B тестирования"""
        results_a = test_config['results_a']
        results_b = test_config['results_b']

        # Рассчет метрик
        metrics_a = self._calculate_test_metrics(results_a['trades'])
        metrics_b = self._calculate_test_metrics(results_b['trades'])

        # Сравнительный анализ
        pnl_diff = metrics_b['total_pnl'] - metrics_a['total_pnl']
        win_rate_diff = metrics_b['win_rate'] - metrics_a['win_rate']

        # Определение победителя (простая логика - можно расширить)
        winner_pnl = test_config['strategy_b'] if pnl_diff > 0 else test_config['strategy_a']
        performance_diff = abs(pnl_diff)

        # Уровень доверительности (упрощенная оценка)
        confidence_level = self._calculate_confidence_level(
            len(results_a['trades']), len(results_b['trades']),
            performance_diff
        )

        return ABTestResult(
            strategy_a=test_config['strategy_a'],
            strategy_b=test_config['strategy_b'],
            winner=winner_pnl,
            confidence_level=confidence_level,
            performance_diff=performance_diff,
            test_duration=test_config['end_time'] - test_config['start_time'],
            sample_size_a=len(results_a['trades']),
            sample_size_b=len(results_b['trades']),
            metrics={'strategy_a': metrics_a, 'strategy_b': metrics_b}
        )

    def _calculate_test_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Расчет метрик для списка сделок"""
        if not trades:
            return {'total_pnl': 0.0, 'win_rate': 0.0, 'max_drawdown': 0.0, 'sharpe_ratio': 0.0}

        pnls = [trade.get('pnl', 0.0) for trade in trades]
        total_pnl = sum(pnls)

        winning_trades = sum(1 for pnl in pnls if pnl > 0)
        win_rate = winning_trades / len(pnls) if pnls else 0.0

        # Max drawdown (упрощенный расчет)
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_drawdown = drawdowns.max() if len(drawdowns) > 0 else 0.0

        # Sharpe ratio (упрощенный)
        returns = np.array(pnls)
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(365)  # Годовая доходность
        else:
            sharpe_ratio = 0.0

        return {
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }

    def _calculate_confidence_level(self, size_a: int, size_b: int, diff: float) -> float:
        """Расчет уровня доверительности (упрощенная модель)"""
        min_samples = min(size_a, size_b)

        if min_samples < self.min_sample_size:
            return 0.7  # Низкий уровень уверенности

        # Увеличиваем уверенность с размером выборки
        base_confidence = 0.85
        sample_bonus = min(min_samples / self.min_sample_size, 1.0) * 0.1

        return min(base_confidence + sample_bonus, 0.95)

    def _apply_recommendation(self, result: ABTestResult):
        """Применение рекомендаций после тестирования"""
        if result.confidence_level >= self.confidence_threshold:
            # Автоматическое переключение на лучшую стратегию
            logger.info(f"Switching to winning strategy: {result.winner}")

            try:
                # Импорт и применение
                from strategy_manager import get_strategy_manager
                strategy_mgr = get_strategy_manager()
                strategy_mgr.set_active_strategy(result.winner)

                # Запись в конфиг
                self._save_strategy_recommendation(result)

            except Exception as e:
                logger.error(f"Failed to apply strategy change: {e}")

    def _find_active_test(self, test_id: str) -> Optional[dict]:
        """Найти активный тест"""
        for test in self.active_tests:
            if test['test_id'] == test_id:
                return test
        return None

    def _save_strategy_recommendation(self, result: ABTestResult):
        """Сохрание рекомендации по стратегии"""
        try:
            config_path = 'config/strategy_recommendations.json'

            recommendation = {
                'timestamp': datetime.now().isoformat(),
                'winner': result.winner,
                'loser': result.strategy_a if result.strategy_b == result.winner else result.strategy_b,
                'confidence_level': result.confidence_level,
                'performance_diff': result.performance_diff,
                'reason': 'A/B testing result'
            }

            # Загрузка существующих рекомендаций
            recommendations = []
            if 'data' in Path() and Path('data/strategy_recommendations.json').exists():
                with open('data/strategy_recommendations.json', 'r') as f:
                    recommendations = json.load(f)

            recommendations.append(recommendation)

            # Сохранение
            with open('data/strategy_recommendations.json', 'w') as f:
                json.dump(recommendations, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Failed to save strategy recommendation: {e}")

    def get_test_results(self, test_id: str = None) -> List[ABTestResult]:
        """Получение результатов тестирования"""
        if test_id:
            return [r for r in self.test_history if r.test_id == test_id]

        return self.test_history

    def stop_test(self, test_id: str):
        """Принудительное завершение теста"""
        test = self._find_active_test(test_id)
        if test:
            self._complete_ab_test(test_id)


class OptunaOptimizer:
    """Optuna оптимизация гиперпараметров стратегий"""

    def __init__(self):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna не установлен. Установите: pip install optuna")

        self.study = None
        self.best_params = {}
        self.objective_function = None

    def optimize_strategy(self, strategy_name: str,
                         data: pd.DataFrame,
                         param_space: Dict[str, Any],
                         metric: str = 'sharpe_ratio',
                         n_trials: int = 100) -> OptimizationResult:
        """
        Оптимизация стратегии через Optuna

        Args:
            strategy_name: Имя стратегии для оптимизации
            data: Исторические данные для тестирования
            param_space: Пространство параметров для оптимизации
            metric: Метрика оптимизации ('sharpe_ratio', 'total_return', 'win_rate')
            n_trials: Количество испытаний

        Returns:
            OptimizationResult с лучшими параметрами
        """
        logger.info(f"Starting Optuna optimization for {strategy_name}")

        start_time = datetime.now()

        def objective(trial):
            # Определение параметров из trial
            params = {}
            for param_name, param_config in param_space.items():
                param_type = param_config.get('type', 'float')

                if param_type == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['min'],
                        param_config['max']
                    )
                elif param_type == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['min'],
                        param_config['max']
                    )
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )

            # Оценка параметров
            score = self._evaluate_params(strategy_name, params, data, metric)

            # Обратный знак для минимизации (Optuna максимизирует)
            return -score if metric in ['sharpe_ratio', 'total_return', 'win_rate'] else score

        # Создание study
        study_name = f"{strategy_name}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        self.study = optuna.create_study(direction='maximize', study_name=study_name)

        # Оптимизация
        self.study.optimize(objective, n_trials=n_trials)

        # Получение результатов
        end_time = datetime.now()
        optimization_time = end_time - start_time

        # Проверка конвергенции
        completed_trials = [t for t in self.study.trials if t.state == optuna.TrialState.COMPLETE]
        convergence = len(completed_trials) >= 10

        result = OptimizationResult(
            strategy_name=strategy_name,
            best_params=self.study.best_params,
            best_value=self.study.best_value,
            trials_count=len(self.study.trials),
            optimization_time=optimization_time,
            convergence=convergence
        )

        # Сохранение лучших параметров
        self.best_params[strategy_name] = result

        logger.info(f"Optuna optimization completed for {strategy_name}")
        logger.info(f"Best value: {result.best_value:.4f}")
        logger.info(f"Best params: {result.best_params}")

        return result

    def _evaluate_params(self, strategy_name: str, params: dict,
                        data: pd.DataFrame, metric: str) -> float:
        """
        Оценка параметров стратегии

        Args:
            strategy_name: Имя стратегии
            params: Параметры для тестирования
            data: Данные для тестирования
            metric: Метрика оценки

        Returns:
            Значение метрики
        """
        try:
            # Импорт стратегии
            from strategy_manager import get_strategy_manager
            strategy_mgr = get_strategy_manager()

            # Создание экземпляра стратегии с новыми параметрами
            strategy = strategy_mgr.get_strategy(strategy_name)
            if not strategy:
                # Динамическое создание с параметрами
                if strategy_name.startswith('trendbot'):
                    from strategies.trendbot import SMACrossRSIStrategy
                    strategy = SMACrossRSIStrategy(config=params)
                else:
                    # Создание с базовыми параметрами
                    strategy = strategy_mgr.get_strategy('ma_crossover')

            if not strategy:
                return 0.0

            # Запуск бэктесинга с данными параметрами
            # Здесь должна быть реализация тестирования стратегии
            # Для демо возвращаем случайное значение
            score = self._mock_strategy_test(strategy_name, params, data, metric)

            return score

        except Exception as e:
            logger.error(f"Error evaluating params: {e}")
            return 0.0

    def _mock_strategy_test(self, strategy_name: str, params: dict,
                           data: pd.DataFrame, metric: str) -> float:
        """
        Мок-тестирование стратегии (для демонстрации)
        В реальной реализации здесь должен быть полноценный бэктест
        """
        # Простая оценка на основе параметров
        base_score = 0.0

        if strategy_name.startswith('trendbot') or strategy_name == 'sma_rsi':
            # Для SMA RSI стратегии
            fast_sma = params.get('fast_sma', 50)
            slow_sma = params.get('slow_sma', 200)

            # Логическая оценка: лучше когда быстрая MA не слишком близка к медленной
            spread_ratio = abs(fast_sma - slow_sma) / max(fast_sma, slow_sma)
            base_score = spread_ratio * 2.0

            # RSI параметры
            rsi = params.get('rsi_oversold', 30)
            if 25 <= rsi <= 35:
                base_score += 0.5

        elif strategy_name == 'ma_crossover':
            # Для простого пересечения MA
            base_score = 0.8

        # Добавляем случайный шума для имитации реальных результатов
        noise = np.random.normal(0, 0.2)
        score = base_score + noise

        # Ограничение в разумных пределах
        score = np.clip(score, 0.1, 2.0)

        return score

    def get_optimization_history(self, strategy_name: str = None) -> List[OptimizationResult]:
        """Получение истории оптимизаций"""
        if strategy_name:
            return [self.best_params.get(strategy_name)]

        return list(self.best_params.values())

    def save_best_params(self, filename: str = 'config/best_params.json'):
        """Сохранение лучших параметров"""
        try:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)

            data_to_save = {
                strategy: {
                    'best_params': result.best_params,
                    'best_value': result.best_value,
                    'optimization_time': result.optimization_time.total_seconds(),
                    'timestamp': datetime.now().isoformat()
                }
                for strategy, result in self.best_params.items()
            }

            with open(filename, 'w') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)

            logger.info(f"Best parameters saved to {filename}")

        except Exception as e:
            logger.error(f"Failed to save best params: {e}")

    def load_best_params(self, filename: str = 'config/best_params.json'):
        """Загрузка лучших параметров"""
        try:
            if Path(filename).exists():
                with open(filename, 'r') as f:
                    data = json.load(f)

                for strategy, params in data.items():
                    if 'best_params' in params:
                        self.best_params[strategy] = OptimizationResult(
                            strategy_name=strategy,
                            best_params=params['best_params'],
                            best_value=params.get('best_value', 0.0),
                            trials_count=params.get('trials_count', 0),
                            optimization_time=timedelta(seconds=params.get('optimization_time', 0)),
                            convergence=True  # Assuming loaded params are good
                        )

                logger.info(f"Best parameters loaded from {filename}")

        except Exception as e:
            logger.error(f"Failed to load best params: {e}")


# Глобальные экземпляры
ab_testing = ABTestingFramework()
if OPTUNA_AVAILABLE:
    optuna_optimizer = OptunaOptimizer()


def start_ab_test(strategy_a: str, strategy_b: str, duration_hours: int = 24) -> str:
    """Запуск A/B тестирования"""
    return ab_testing.start_ab_test(strategy_a, strategy_b, duration_hours)


def record_trade_for_ab_test(test_id: str, strategy: str, trade_result: dict):
    """Запись сделки для A/B тестирования"""
    ab_testing.record_trade(test_id, strategy, trade_result)


def get_ab_test_results(test_id: str = None) -> List[ABTestResult]:
    """Получение результатов A/B тестирования"""
    return ab_testing.get_test_results(test_id)


def optimize_strategy_params(strategy_name: str, data: pd.DataFrame,
                           param_space: Dict[str, Any], metric: str = 'sharpe_ratio') -> OptimizationResult:
    """Оптимизация параметров стратегии через Optuna"""
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna не установлен. Установите: pip install optuna")

    return optuna_optimizer.optimize_strategy(strategy_name, data, param_space, metric)


def get_optimized_params(strategy_name: str) -> dict:
    """Получение оптимизированных параметров стратегии"""
    if strategy_name in optuna_optimizer.best_params:
        return optuna_optimizer.best_params[strategy_name].best_params
    return {}