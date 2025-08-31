# src/models/classifiers.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MarketDirectionClassifier:
    """
    Классификатор для предсказания направления движения рынка
    Использует подходы из книги "ИИ в трейдинге"
    """

    def __init__(self, model_type: str = 'random_forest'):
        """
        Args:
            model_type: Тип классификатора ('random_forest', 'gradient_boost', 'logistic', 'svm')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_columns = []
        self.target_mapping = {-1: 'DOWN', 0: 'FLAT', 1: 'UP'}

        # Инициализация модели
        self._init_model()

    def _init_model(self):
        """Инициализация модели в зависимости от типа"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boost':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'logistic':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                probability=True,  # Для получения вероятностей
                random_state=42
            )
        else:
            raise ValueError(f"Неизвестный тип модели: {self.model_type}")

    def _create_target_variable(self, df: pd.DataFrame, target_column: str = 'close', 
                               threshold: float = 0.005, horizon: int = 1) -> pd.Series:
        """
        Создание целевой переменной для классификации направления движения

        Args:
            df: Данные
            target_column: Колонка для анализа
            threshold: Пороговое значение для определения значимого движения
            horizon: Горизонт прогноза (количество периодов вперед)

        Returns:
            Серия с метками: -1 (падение), 0 (боковик), 1 (рост)
        """
        if target_column not in df.columns:
            raise ValueError(f"Колонка {target_column} не найдена")

        # Вычисляем изменение цены через horizon периодов
        future_price = df[target_column].shift(-horizon)
        current_price = df[target_column]

        # Относительное изменение
        price_change = (future_price - current_price) / current_price

        # Классификация
        target = pd.Series(0, index=df.index)  # По умолчанию FLAT
        target[price_change > threshold] = 1   # UP
        target[price_change < -threshold] = -1 # DOWN

        # Убираем последние строки где нет будущих данных
        target = target[:-horizon] if horizon > 0 else target

        logger.info(f"Создана целевая переменная: UP={sum(target==1)}, FLAT={sum(target==0)}, DOWN={sum(target==-1)}")

        return target

    def fit(self, df: pd.DataFrame, 
            feature_columns: List[str] = None,
            target_column: str = 'close',
            threshold: float = 0.005,
            horizon: int = 1,
            optimize_params: bool = False) -> 'MarketDirectionClassifier':
        """
        Обучение классификатора

        Args:
            df: Данные для обучения
            feature_columns: Колонки признаков
            target_column: Целевая колонка
            threshold: Пороговое значение для классификации
            horizon: Горизонт прогноза
            optimize_params: Оптимизировать гиперпараметры
        """
        try:
            logger.info(f"🎯 Обучение {self.model_type} классификатора на {len(df)} записях")

            # Подготовка признаков
            if feature_columns is None:
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                # Исключаем целевую переменную
                feature_columns = [col for col in numeric_columns if col != target_column]

            self.feature_columns = feature_columns

            # Создание целевой переменной
            y = self._create_target_variable(df, target_column, threshold, horizon)

            # Подготовка признаков (убираем последние строки для выравнивания с target)
            X = df[feature_columns].iloc[:len(y)]

            # Проверка на достаточность данных
            if len(X) < 50:
                raise ValueError(f"Недостаточно данных для обучения: {len(X)} < 50")

            # Проверка баланса классов
            class_counts = pd.Series(y).value_counts()
            logger.info(f"📊 Распределение классов: {dict(class_counts)}")

            # Обработка пропущенных значений
            X = X.fillna(X.mean())

            # Нормализация признаков
            X_scaled = self.scaler.fit_transform(X)

            # Оптимизация гиперпараметров
            if optimize_params:
                self._optimize_hyperparameters(X_scaled, y)

            # Обучение модели
            self.model.fit(X_scaled, y)

            # Оценка на обучающих данных
            train_score = self.model.score(X_scaled, y)

            # Кросс-валидация
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='accuracy')

            self.is_fitted = True

            logger.info(f"✅ Обучение завершено: Train Acc={train_score:.3f}, CV Acc={cv_scores.mean():.3f}±{cv_scores.std():.3f}")

            return self

        except Exception as e:
            logger.error(f"❌ Ошибка обучения классификатора: {e}")
            raise

    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray):
        """Оптимизация гиперпараметров модели"""
        logger.info("🔧 Оптимизация гиперпараметров...")

        if self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        elif self.model_type == 'gradient_boost':
            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        elif self.model_type == 'logistic':
            param_grid = {
                'C': [0.01, 0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2']
            }
        elif self.model_type == 'svm':
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto', 0.01, 0.1]
            }
        else:
            return

        try:
            grid_search = GridSearchCV(
                self.model, param_grid, 
                cv=3, scoring='accuracy', 
                n_jobs=-1, verbose=0
            )
            grid_search.fit(X, y)

            self.model = grid_search.best_estimator_
            logger.info(f"✅ Оптимальные параметры: {grid_search.best_params_}")

        except Exception as e:
            logger.warning(f"⚠️ Ошибка оптимизации параметров: {e}")

    def predict(self, df: pd.DataFrame, return_probabilities: bool = False) -> np.ndarray:
        """
        Предсказание направления движения

        Args:
            df: Данные для предсказания
            return_probabilities: Возвращать вероятности классов

        Returns:
            Предсказания или вероятности
        """
        if not self.is_fitted:
            raise ValueError("❌ Модель не обучена. Вызовите fit() перед predict()")

        try:
            # Подготовка признаков
            X = df[self.feature_columns].fillna(df[self.feature_columns].mean())
            X_scaled = self.scaler.transform(X)

            if return_probabilities:
                probabilities = self.model.predict_proba(X_scaled)
                return probabilities
            else:
                predictions = self.model.predict(X_scaled)
                return predictions

        except Exception as e:
            logger.error(f"❌ Ошибка предсказания: {e}")
            raise

    def predict_single(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Предсказание для одной записи с интерпретацией
        """
        if not self.is_fitted:
            raise ValueError("❌ Модель не обучена")

        try:
            # Подготовка данных
            feature_vector = np.array([features.get(col, 0) for col in self.feature_columns]).reshape(1, -1)
            feature_vector_scaled = self.scaler.transform(feature_vector)

            # Предсказание
            prediction = self.model.predict(feature_vector_scaled)[0]
            probabilities = self.model.predict_proba(feature_vector_scaled)[0]

            # Формирование результата
            result = {
                'prediction': prediction,
                'prediction_label': self.target_mapping[prediction],
                'probabilities': {
                    'DOWN': probabilities[0] if len(probabilities) > 0 else 0,
                    'FLAT': probabilities[1] if len(probabilities) > 1 else 0, 
                    'UP': probabilities[2] if len(probabilities) > 2 else 0
                },
                'confidence': max(probabilities)
            }

            # Важность признаков (если доступна)
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
                result['feature_importance'] = feature_importance

            return result

        except Exception as e:
            logger.error(f"❌ Ошибка предсказания: {e}")
            raise

    def evaluate(self, df: pd.DataFrame, target_column: str = 'close',
                threshold: float = 0.005, horizon: int = 1) -> Dict[str, float]:
        """
        Оценка качества классификатора
        """
        if not self.is_fitted:
            raise ValueError("❌ Модель не обучена")

        try:
            # Создание целевой переменной
            y_true = self._create_target_variable(df, target_column, threshold, horizon)

            # Предсказания
            X = df[self.feature_columns].iloc[:len(y_true)]
            X = X.fillna(X.mean())
            X_scaled = self.scaler.transform(X)
            y_pred = self.model.predict(X_scaled)

            # Метрики
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
                'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
                'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
                'samples': len(y_true)
            }

            # Метрики по классам
            for class_label, class_name in self.target_mapping.items():
                if class_label in y_true.values:
                    class_precision = precision_score(y_true, y_pred, labels=[class_label], average='macro', zero_division=0)
                    class_recall = recall_score(y_true, y_pred, labels=[class_label], average='macro', zero_division=0)
                    metrics[f'{class_name.lower()}_precision'] = class_precision
                    metrics[f'{class_name.lower()}_recall'] = class_recall

            logger.info(f"📊 Метрики модели: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1_macro']:.3f}")

            return metrics

        except Exception as e:
            logger.error(f"❌ Ошибка оценки: {e}")
            return {}

    def get_feature_importance(self) -> Dict[str, float]:
        """Получить важность признаков"""
        if not self.is_fitted:
            raise ValueError("❌ Модель не обучена")

        if hasattr(self.model, 'feature_importances_'):
            importance = dict(zip(self.feature_columns, self.model.feature_importances_))
            # Сортируем по важности
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            return importance
        else:
            logger.warning("⚠️ Модель не поддерживает feature importance")
            return {}

    def save_model(self, filepath: str) -> bool:
        """Сохранение обученной модели"""
        if not self.is_fitted:
            logger.error("❌ Нет обученной модели для сохранения")
            return False

        try:
            import pickle

            model_data = {
                'model_type': self.model_type,
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'target_mapping': self.target_mapping,
                'is_fitted': self.is_fitted
            }

            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"💾 Классификатор сохранен: {filepath}")
            return True

        except Exception as e:
            logger.error(f"❌ Ошибка сохранения: {e}")
            return False

    @classmethod
    def load_model(cls, filepath: str) -> 'MarketDirectionClassifier':
        """Загрузка сохраненной модели"""
        try:
            import pickle

            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            classifier = cls(model_type=model_data['model_type'])
            classifier.model = model_data['model']
            classifier.scaler = model_data['scaler']
            classifier.feature_columns = model_data['feature_columns']
            classifier.target_mapping = model_data['target_mapping']
            classifier.is_fitted = model_data['is_fitted']

            logger.info(f"📂 Классификатор загружен: {filepath}")
            return classifier

        except Exception as e:
            logger.error(f"❌ Ошибка загрузки: {e}")
            raise


class EnsembleClassifier:
    """
    Ансамбль классификаторов для повышения точности
    """

    def __init__(self, models: List[str] = None):
        if models is None:
            models = ['random_forest', 'gradient_boost', 'logistic']

        self.models = []
        self.weights = []

        for model_type in models:
            self.models.append(MarketDirectionClassifier(model_type))

        self.is_fitted = False

    def fit(self, df: pd.DataFrame, **kwargs):
        """Обучение всех моделей в ансамбле"""
        logger.info(f"🎯 Обучение ансамбля из {len(self.models)} моделей")

        model_scores = []

        for i, model in enumerate(self.models):
            try:
                logger.info(f"🔄 Обучение модели {i+1}/{len(self.models)}: {model.model_type}")
                model.fit(df, **kwargs)

                # Оценка модели для весов
                metrics = model.evaluate(df, **kwargs)
                score = metrics.get('f1_macro', 0.5)
                model_scores.append(score)

                logger.info(f"✅ Модель {model.model_type}: F1={score:.3f}")

            except Exception as e:
                logger.error(f"❌ Ошибка обучения {model.model_type}: {e}")
                model_scores.append(0.1)  # Минимальный вес для неработающих моделей

        # Вычисление весов (нормализованные оценки)
        total_score = sum(model_scores)
        if total_score > 0:
            self.weights = [score / total_score for score in model_scores]
        else:
            self.weights = [1.0 / len(self.models)] * len(self.models)

        self.is_fitted = True

        logger.info(f"✅ Ансамбль обучен. Веса моделей: {dict(zip([m.model_type for m in self.models], self.weights))}")

        return self

    def predict(self, df: pd.DataFrame, return_probabilities: bool = False):
        """Предсказание ансамбля"""
        if not self.is_fitted:
            raise ValueError("❌ Ансамбль не обучен")

        if return_probabilities:
            # Взвешенное усреднение вероятностей
            ensemble_probs = np.zeros((len(df), 3))  # 3 класса: DOWN, FLAT, UP

            for model, weight in zip(self.models, self.weights):
                try:
                    probs = model.predict(df, return_probabilities=True)
                    ensemble_probs += weight * probs
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка модели {model.model_type}: {e}")

            return ensemble_probs
        else:
            # Голосование по предсказаниям
            predictions = []

            for model, weight in zip(self.models, self.weights):
                try:
                    pred = model.predict(df)
                    predictions.append((pred, weight))
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка модели {model.model_type}: {e}")

            # Взвешенное голосование
            if predictions:
                weighted_votes = np.zeros((len(df), 3))  # Для 3 классов

                for pred, weight in predictions:
                    for i, class_pred in enumerate(pred):
                        class_idx = {-1: 0, 0: 1, 1: 2}[class_pred]
                        weighted_votes[i, class_idx] += weight

                # Выбираем класс с максимальным весом
                final_predictions = []
                for votes in weighted_votes:
                    class_idx = np.argmax(votes)
                    class_label = {0: -1, 1: 0, 2: 1}[class_idx]
                    final_predictions.append(class_label)

                return np.array(final_predictions)
            else:
                return np.zeros(len(df))


if __name__ == "__main__":
    # Тестирование классификатора
    print("🧪 Тестирование классификатора направления рынка")

    # Создаем тестовые данные с техническими индикаторами
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=300, freq='D')

    # Имитируем цены с трендом
    base_price = 100
    trend = np.linspace(0, 20, 300)
    noise = np.random.randn(300) * 2
    prices = base_price + trend + noise.cumsum() * 0.5

    test_data = pd.DataFrame({
        'close': prices,
        'volume': np.random.randint(1000, 10000, 300),
        'sma_10': pd.Series(prices).rolling(10).mean().fillna(prices),
        'sma_20': pd.Series(prices).rolling(20).mean().fillna(prices),
        'rsi': np.random.uniform(20, 80, 300),
        'macd': np.random.randn(300) * 0.5,
        'volatility': pd.Series(prices).pct_change().rolling(20).std().fillna(0.02) * 100
    }, index=dates)

    print(f"📊 Тестовые данные: {len(test_data)} записей, {len(test_data.columns)} признаков")

    try:
        # Тестирование Random Forest
        rf_classifier = MarketDirectionClassifier('random_forest')

        print("🎯 Обучение Random Forest классификатора...")
        rf_classifier.fit(
            test_data,
            feature_columns=['volume', 'sma_10', 'sma_20', 'rsi', 'macd', 'volatility'],
            threshold=0.01,
            horizon=1
        )

        # Предсказания
        predictions = rf_classifier.predict(test_data.tail(10))
        print(f"📈 Предсказания на последние 10 дней: {predictions}")

        # Предсказание для одной записи
        single_pred = rf_classifier.predict_single({
            'volume': 5000,
            'sma_10': 120,
            'sma_20': 118,
            'rsi': 65,
            'macd': 0.5,
            'volatility': 2.0
        })
        print(f"🎯 Единичное предсказание: {single_pred['prediction_label']} (уверенность: {single_pred['confidence']:.2f})")

        # Оценка модели
        metrics = rf_classifier.evaluate(test_data)
        print(f"📊 Метрики: Accuracy={metrics.get('accuracy', 0):.3f}, F1={metrics.get('f1_macro', 0):.3f}")

        # Важность признаков
        importance = rf_classifier.get_feature_importance()
        print(f"📋 Топ-3 важных признака: {list(importance.items())[:3]}")

        # Тестирование ансамбля
        print("\n🎯 Тестирование ансамблевого классификатора...")
        ensemble = EnsembleClassifier(['random_forest', 'gradient_boost'])
        ensemble.fit(
            test_data,
            feature_columns=['volume', 'sma_10', 'sma_20', 'rsi', 'macd', 'volatility'],
            threshold=0.01,
            horizon=1
        )

        ensemble_predictions = ensemble.predict(test_data.tail(10))
        print(f"🏆 Ансамблевые предсказания: {ensemble_predictions}")

        print("✅ Тестирование классификатора завершено успешно!")

    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
        import traceback
        traceback.print_exc()
