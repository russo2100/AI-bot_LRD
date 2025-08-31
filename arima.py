# src/models/arima.py
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class ARIMAModel:
    """
    ARIMA модель для прогнозирования временных рядов
    Реализует подходы из книги "ИИ в трейдинге"
    """

    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        """
        Args:
            order: Порядок ARIMA модели (p, d, q)
                   p - авторегрессия, d - интегрирование, q - скользящее среднее
        """
        self.order = order
        self.model = None
        self.fitted_model = None
        self.is_fitted = False
        self.fit_results = {}

    def check_stationarity(self, series: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Проверка стационарности ряда с помощью теста Дики-Фуллера
        Важный этап предобработки согласно книге
        """
        try:
            result = adfuller(series.dropna())

            is_stationary = result[1] <= alpha

            return {
                'is_stationary': is_stationary,
                'adf_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'recommendation': 'Ряд стационарен' if is_stationary else f'Рекомендуется дифференцирование (d={self.order[1]+1})'
            }
        except Exception as e:
            logger.error(f"❌ Ошибка проверки стационарности: {e}")
            return {'is_stationary': False, 'error': str(e)}

    def make_stationary(self, series: pd.Series, max_diff: int = 2) -> Tuple[pd.Series, int]:
        """
        Приведение ряда к стационарному виду через дифференцирование
        """
        original_series = series.copy()
        diff_order = 0

        for d in range(max_diff + 1):
            stationarity = self.check_stationarity(series)

            if stationarity.get('is_stationary', False):
                logger.info(f"✅ Ряд стационарен после {d} дифференцирований")
                return series, d

            if d < max_diff:
                series = series.diff().dropna()
                diff_order = d + 1

        logger.warning(f"⚠️ Ряд не стал стационарным после {max_diff} дифференцирований")
        return series, diff_order

    def auto_arima(self, series: pd.Series, max_p: int = 5, max_d: int = 2, max_q: int = 5) -> Tuple[int, int, int]:
        """
        Автоматический выбор оптимальных параметров ARIMA на основе AIC
        """
        logger.info("🔍 Поиск оптимальных параметров ARIMA...")

        best_aic = np.inf
        best_order = (1, 1, 1)
        best_model = None

        # Проверка стационарности для определения d
        stationarity_result = self.check_stationarity(series)
        if not stationarity_result.get('is_stationary', False):
            _, optimal_d = self.make_stationary(series, max_d)
        else:
            optimal_d = 0

        # Ограничиваем поиск вокруг найденного d
        d_range = [optimal_d] if optimal_d > 0 else [0, 1]

        total_combinations = len(d_range) * (max_p + 1) * (max_q + 1)
        tested = 0

        for d in d_range:
            for p in range(max_p + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted = model.fit(method_kwargs={"warn_convergence": False})

                        aic = fitted.aic
                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                            best_model = fitted

                        tested += 1
                        if tested % 10 == 0:
                            logger.info(f"🔄 Протестировано {tested}/{total_combinations} комбинаций")

                    except Exception as e:
                        continue

        logger.info(f"✅ Лучшие параметры ARIMA{best_order} с AIC={best_aic:.2f}")
        return best_order

    def fit(self, series: pd.Series, auto_order: bool = False) -> 'ARIMAModel':
        """
        Обучение ARIMA модели

        Args:
            series: Временной ряд для обучения
            auto_order: Автоматически подбирать параметры модели
        """
        try:
            logger.info(f"🎯 Обучение ARIMA модели на {len(series)} наблюдениях")

            # Очистка данных
            clean_series = series.dropna()
            if len(clean_series) < 10:
                raise ValueError(f"Недостаточно данных: {len(clean_series)} < 10")

            # Автоматический подбор параметров
            if auto_order:
                self.order = self.auto_arima(clean_series)

            # Проверка стационарности
            stationarity = self.check_stationarity(clean_series)
            logger.info(f"📊 Стационарность: {stationarity.get('recommendation', 'Неизвестно')}")

            # Обучение модели
            self.model = ARIMA(clean_series, order=self.order)
            self.fitted_model = self.model.fit(method_kwargs={"warn_convergence": False})

            # Сохранение результатов
            self.fit_results = {
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'log_likelihood': self.fitted_model.llf,
                'params': self.fitted_model.params.to_dict(),
                'training_samples': len(clean_series),
                'stationarity': stationarity
            }

            self.is_fitted = True
            logger.info(f"✅ ARIMA{self.order} обучена (AIC={self.fitted_model.aic:.2f})")

            return self

        except Exception as e:
            logger.error(f"❌ Ошибка обучения ARIMA: {e}")
            raise

    def predict(self, steps: int = 1, return_conf_int: bool = False, alpha: float = 0.05) -> pd.Series:
        """
        Прогнозирование на заданное количество шагов

        Args:
            steps: Количество шагов для прогноза
            return_conf_int: Возвращать доверительные интервалы
            alpha: Уровень значимости для доверительных интервалов
        """
        if not self.is_fitted:
            raise ValueError("❌ Модель не обучена. Вызовите fit() перед predict()")

        try:
            forecast = self.fitted_model.forecast(steps=steps, alpha=alpha)

            if return_conf_int:
                conf_int = self.fitted_model.get_forecast(steps=steps, alpha=alpha).conf_int()
                return forecast, conf_int

            logger.info(f"📈 Прогноз на {steps} шагов: {forecast.iloc[0]:.4f}")
            return forecast

        except Exception as e:
            logger.error(f"❌ Ошибка прогнозирования: {e}")
            raise

    def residual_analysis(self) -> Dict[str, Any]:
        """
        Анализ остатков модели для проверки качества
        """
        if not self.is_fitted:
            raise ValueError("❌ Модель не обучена")

        try:
            residuals = self.fitted_model.resid

            # Статистики остатков
            residual_stats = {
                'mean': residuals.mean(),
                'std': residuals.std(),
                'skewness': residuals.skew(),
                'kurtosis': residuals.kurtosis(),
                'jarque_bera_pvalue': None,  # Тест нормальности
                'ljung_box_pvalue': None,    # Тест автокорреляции
            }

            # Тест Льюнга-Бокса на автокорреляцию остатков
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                lb_result = acorr_ljungbox(residuals, lags=min(10, len(residuals)//4), return_df=True)
                residual_stats['ljung_box_pvalue'] = lb_result['lb_pvalue'].iloc[-1]
            except:
                pass

            # Тест Харке-Бера на нормальность
            try:
                from scipy.stats import jarque_bera
                jb_stat, jb_pvalue = jarque_bera(residuals)
                residual_stats['jarque_bera_pvalue'] = jb_pvalue
            except:
                pass

            return residual_stats

        except Exception as e:
            logger.error(f"❌ Ошибка анализа остатков: {e}")
            return {}

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Получить полную сводку по модели
        """
        if not self.is_fitted:
            return {"error": "Модель не обучена"}

        summary = {
            'order': self.order,
            'fit_results': self.fit_results,
            'residual_analysis': self.residual_analysis(),
            'model_summary': str(self.fitted_model.summary()) if self.fitted_model else None
        }

        return summary

    def save_model(self, filepath: str) -> bool:
        """
        Сохранение обученной модели
        """
        if not self.is_fitted:
            logger.error("❌ Нет обученной модели для сохранения")
            return False

        try:
            import pickle

            model_data = {
                'order': self.order,
                'fitted_model': self.fitted_model,
                'fit_results': self.fit_results,
                'is_fitted': self.is_fitted
            }

            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"💾 Модель сохранена в {filepath}")
            return True

        except Exception as e:
            logger.error(f"❌ Ошибка сохранения модели: {e}")
            return False

    @classmethod
    def load_model(cls, filepath: str) -> 'ARIMAModel':
        """
        Загрузка сохраненной модели
        """
        try:
            import pickle

            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            model = cls(order=model_data['order'])
            model.fitted_model = model_data['fitted_model']
            model.fit_results = model_data['fit_results']
            model.is_fitted = model_data['is_fitted']

            logger.info(f"📂 Модель загружена из {filepath}")
            return model

        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            raise


def decompose_time_series(series: pd.Series, model: str = 'additive', period: int = None) -> Dict[str, pd.Series]:
    """
    Декомпозиция временного ряда на тренд, сезонность и остатки
    """
    try:
        if period is None:
            # Попытка автоматически определить период
            period = min(len(series) // 4, 365) if len(series) > 8 else 2

        decomposition = seasonal_decompose(series, model=model, period=period)

        return {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal, 
            'residual': decomposition.resid,
            'observed': decomposition.observed
        }

    except Exception as e:
        logger.error(f"❌ Ошибка декомпозиции: {e}")
        return {}


if __name__ == "__main__":
    # Тестирование ARIMA модели
    print("🧪 Тестирование ARIMA модели")

    # Создаем тестовый временной ряд с трендом и шумом
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='D')

    # Имитируем цену с трендом и случайными колебаниями
    trend = np.linspace(100, 120, 200)
    noise = np.random.randn(200) * 2
    seasonal = 5 * np.sin(2 * np.pi * np.arange(200) / 50)  # сезонность

    prices = trend + seasonal + noise
    price_series = pd.Series(prices, index=dates, name='price')

    print(f"📊 Тестовый ряд: {len(price_series)} наблюдений")

    try:
        # Создание и обучение модели
        model = ARIMAModel()

        # Проверка стационарности
        stationarity = model.check_stationarity(price_series)
        print(f"📈 Стационарность: {stationarity['recommendation']}")

        # Обучение с автоподбором параметров
        print("🎯 Обучение модели с автоподбором параметров...")
        model.fit(price_series, auto_order=True)

        # Прогнозирование
        forecast = model.predict(steps=5)
        print(f"📮 Прогноз на 5 дней: {forecast.iloc[0]:.2f}")

        # Анализ остатков
        residual_stats = model.residual_analysis()
        print(f"📊 Среднее остатков: {residual_stats.get('mean', 0):.4f}")
        print(f"📊 Стд. отклонение остатков: {residual_stats.get('std', 0):.4f}")

        # Получение сводки модели
        summary = model.get_model_summary()
        print(f"📋 AIC: {summary['fit_results']['aic']:.2f}")
        print(f"📋 Параметры: {summary['order']}")

        print("✅ Тестирование ARIMA завершено успешно!")

    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
