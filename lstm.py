# src/models/lstm.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List
import logging
import warnings

# Подавляем предупреждения TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l1_l2
    TF_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow не установлен. Установите: pip install tensorflow")
    TF_AVAILABLE = False

logger = logging.getLogger(__name__)

class LSTMModel:
    """
    LSTM модель для прогнозирования временных рядов
    Реализует подходы глубокого обучения из книги "ИИ в трейдинге"
    """

    def __init__(self, 
                 window_size: int = 30,
                 units: int = 64,
                 dropout: float = 0.2,
                 layers: int = 2,
                 learning_rate: float = 0.001):
        """
        Args:
            window_size: Размер входного окна (рекомендуется 30 дней как в книге)
            units: Количество LSTM нейронов
            dropout: Коэффициент dropout для регуляризации
            layers: Количество LSTM слоев
            learning_rate: Скорость обучения
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow не установлен")

        self.window_size = window_size
        self.units = units
        self.dropout = dropout
        self.layers = layers
        self.learning_rate = learning_rate

        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.feature_columns = []
        self.history = {}

        # Настройка TensorFlow для стабильности
        tf.random.set_seed(42)

    def _prepare_sequences(self, data: np.ndarray, target: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Создание последовательностей для LSTM согласно методике из книги
        """
        if len(data) < self.window_size + 1:
            raise ValueError(f"Недостаточно данных: {len(data)} < {self.window_size + 1}")

        X, y = [], []

        for i in range(len(data) - self.window_size):
            # Входная последовательность
            X.append(data[i:i + self.window_size])

            # Целевое значение (следующий после последовательности)
            if target is not None:
                y.append(target[i + self.window_size])
            else:
                # Если target не указан, используем последний столбец data
                y.append(data[i + self.window_size, -1])

        X = np.array(X)
        y = np.array(y)

        logger.info(f"📊 Подготовлено последовательностей: {X.shape[0]}, размер окна: {X.shape[1]}, признаков: {X.shape[2]}")

        return X, y

    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Построение архитектуры LSTM модели
        """
        model = Sequential()

        # Входной слой
        model.add(Input(shape=input_shape))

        # LSTM слои
        for i in range(self.layers):
            return_sequences = (i < self.layers - 1)  # Все кроме последнего слоя

            model.add(LSTM(
                units=self.units,
                return_sequences=return_sequences,
                dropout=self.dropout,
                recurrent_dropout=self.dropout,
                kernel_regularizer=l1_l2(l1=0.01, l2=0.01)
            ))

            # Batch normalization для стабилизации обучения
            if i < self.layers - 1:
                model.add(BatchNormalization())

        # Выходные слои
        model.add(Dropout(self.dropout))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(self.dropout / 2))
        model.add(Dense(1))  # Один выход для цены

        # Компиляция модели
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        return model

    def fit(self, 
            df: pd.DataFrame,
            target_column: str = 'close',
            feature_columns: List[str] = None,
            validation_split: float = 0.2,
            epochs: int = 100,
            batch_size: int = 32,
            verbose: int = 1) -> 'LSTMModel':
        """
        Обучение LSTM модели

        Args:
            df: Данные для обучения
            target_column: Целевая колонка для прогноза
            feature_columns: Колонки признаков (None = все числовые)
            validation_split: Доля данных для валидации
            epochs: Количество эпох обучения
            batch_size: Размер батча
            verbose: Уровень детализации вывода
        """
        try:
            logger.info(f"🎯 Обучение LSTM модели на {len(df)} записях")

            # Подготовка признаков
            if feature_columns is None:
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                # Исключаем целевую колонку из признаков если она там есть
                feature_columns = [col for col in numeric_columns if col != target_column]

            self.feature_columns = feature_columns

            if target_column not in df.columns:
                raise ValueError(f"Целевая колонка '{target_column}' не найдена")

            # Нормализация данных
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()

            # Подготовка данных
            features_data = df[feature_columns].values
            target_data = df[target_column].values

            # Нормализация
            features_scaled = self.scaler.fit_transform(features_data)
            target_scaled = (target_data - target_data.min()) / (target_data.max() - target_data.min())

            # Создание последовательностей
            X, y = self._prepare_sequences(features_scaled, target_scaled)

            # Разделение на обучающую и валидационную выборки
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            logger.info(f"📊 Train: {X_train.shape[0]} образцов, Val: {X_val.shape[0]} образцов")

            # Построение модели
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.model = self._build_model(input_shape)

            logger.info(f"🏗️ Архитектура LSTM: {self.layers} слоев по {self.units} нейронов")

            # Callbacks для улучшения обучения
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=verbose
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.7,
                    patience=8,
                    min_lr=1e-6,
                    verbose=verbose
                )
            ]

            # Обучение модели
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose,
                shuffle=True
            )

            self.history = {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss'],
                'mae': history.history['mae'],
                'val_mae': history.history['val_mae']
            }

            self.is_fitted = True

            # Финальные метрики
            final_loss = history.history['val_loss'][-1]
            final_mae = history.history['val_mae'][-1]

            logger.info(f"✅ LSTM обучена: Val Loss={final_loss:.6f}, Val MAE={final_mae:.6f}")

            return self

        except Exception as e:
            logger.error(f"❌ Ошибка обучения LSTM: {e}")
            raise

    def predict(self, 
                df: pd.DataFrame, 
                steps: int = 1,
                return_sequences: bool = False) -> np.ndarray:
        """
        Прогнозирование с помощью обученной LSTM модели

        Args:
            df: Данные для прогноза
            steps: Количество шагов прогноза
            return_sequences: Возвращать все промежуточные прогнозы
        """
        if not self.is_fitted:
            raise ValueError("❌ Модель не обучена. Вызовите fit() перед predict()")

        try:
            # Подготовка последней последовательности
            features_data = df[self.feature_columns].tail(self.window_size).values
            features_scaled = self.scaler.transform(features_data)

            # Преобразование в формат для LSTM
            X = features_scaled.reshape(1, self.window_size, len(self.feature_columns))

            predictions = []
            current_sequence = X.copy()

            # Многошаговое прогнозирование
            for step in range(steps):
                # Прогноз следующего значения
                pred = self.model.predict(current_sequence, verbose=0)
                predictions.append(pred[0, 0])

                if step < steps - 1:
                    # Обновление последовательности для следующего прогноза
                    # Сдвигаем окно и добавляем прогноз как последний элемент
                    new_sequence = np.roll(current_sequence, -1, axis=1)
                    # Для простоты используем прогноз как все признаки (можно улучшить)
                    new_sequence[0, -1, :] = pred[0, 0]  
                    current_sequence = new_sequence

            predictions = np.array(predictions)

            if return_sequences:
                logger.info(f"📈 Многошаговый прогноз на {steps} периодов")
                return predictions
            else:
                logger.info(f"📈 Прогноз на следующий период: {predictions[0]:.6f}")
                return predictions[0]

        except Exception as e:
            logger.error(f"❌ Ошибка прогнозирования: {e}")
            raise

    def evaluate(self, df: pd.DataFrame, target_column: str = 'close') -> Dict[str, float]:
        """
        Оценка качества модели на тестовых данных
        """
        if not self.is_fitted:
            raise ValueError("❌ Модель не обучена")

        try:
            # Подготовка данных
            features_data = df[self.feature_columns].values
            target_data = df[target_column].values

            # Нормализация (используем уже обученный scaler)
            features_scaled = self.scaler.transform(features_data)
            target_scaled = (target_data - target_data.min()) / (target_data.max() - target_data.min())

            # Создание последовательностей
            X, y = self._prepare_sequences(features_scaled, target_scaled)

            # Оценка модели
            loss, mae = self.model.evaluate(X, y, verbose=0)

            # Дополнительные метрики
            predictions = self.model.predict(X, verbose=0).flatten()

            # RMSE
            rmse = np.sqrt(np.mean((y - predictions) ** 2))

            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((y - predictions) / (y + 1e-10))) * 100

            # Направленная точность (процент правильно предсказанных направлений)
            actual_direction = np.sign(np.diff(y))
            predicted_direction = np.sign(np.diff(predictions))
            directional_accuracy = np.mean(actual_direction == predicted_direction) * 100

            metrics = {
                'loss': loss,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'directional_accuracy': directional_accuracy,
                'samples': len(y)
            }

            logger.info(f"📊 Оценка модели: MAE={mae:.6f}, RMSE={rmse:.6f}, DA={directional_accuracy:.1f}%")

            return metrics

        except Exception as e:
            logger.error(f"❌ Ошибка оценки: {e}")
            return {}

    def plot_training_history(self) -> None:
        """
        Визуализация процесса обучения
        """
        if not self.history:
            logger.warning("⚠️ История обучения отсутствует")
            return

        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # График потерь
            ax1.plot(self.history['loss'], label='Train Loss')
            ax1.plot(self.history['val_loss'], label='Val Loss')
            ax1.set_title('Model Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)

            # График MAE
            ax2.plot(self.history['mae'], label='Train MAE')
            ax2.plot(self.history['val_mae'], label='Val MAE')
            ax2.set_title('Model MAE')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            plt.show()

        except ImportError:
            logger.warning("⚠️ Matplotlib не установлен для визуализации")
        except Exception as e:
            logger.error(f"❌ Ошибка построения графика: {e}")

    def save_model(self, filepath: str) -> bool:
        """
        Сохранение обученной модели
        """
        if not self.is_fitted:
            logger.error("❌ Нет обученной модели для сохранения")
            return False

        try:
            # Сохраняем модель TensorFlow
            model_path = filepath.replace('.pkl', '_model.h5')
            self.model.save(model_path)

            # Сохраняем дополнительные данные
            import pickle
            additional_data = {
                'window_size': self.window_size,
                'units': self.units,
                'dropout': self.dropout,
                'layers': self.layers,
                'learning_rate': self.learning_rate,
                'feature_columns': self.feature_columns,
                'scaler': self.scaler,
                'history': self.history,
                'is_fitted': self.is_fitted
            }

            with open(filepath, 'wb') as f:
                pickle.dump(additional_data, f)

            logger.info(f"💾 LSTM модель сохранена: {filepath}, {model_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Ошибка сохранения: {e}")
            return False

    @classmethod
    def load_model(cls, filepath: str) -> 'LSTMModel':
        """
        Загрузка сохраненной модели
        """
        try:
            import pickle

            # Загружаем дополнительные данные
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            # Создаем экземпляр модели
            model = cls(
                window_size=data['window_size'],
                units=data['units'],
                dropout=data['dropout'],
                layers=data['layers'],
                learning_rate=data['learning_rate']
            )

            # Загружаем TensorFlow модель
            model_path = filepath.replace('.pkl', '_model.h5')
            model.model = tf.keras.models.load_model(model_path)

            # Восстанавливаем состояние
            model.feature_columns = data['feature_columns']
            model.scaler = data['scaler']
            model.history = data['history']
            model.is_fitted = data['is_fitted']

            logger.info(f"📂 LSTM модель загружена из {filepath}")
            return model

        except Exception as e:
            logger.error(f"❌ Ошибка загрузки: {e}")
            raise


class GRUModel(LSTMModel):
    """
    GRU модель - альтернатива LSTM с меньшим количеством параметров
    """

    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Построение архитектуры GRU модели
        """
        model = Sequential()

        # Входной слой
        model.add(Input(shape=input_shape))

        # GRU слои (быстрее LSTM)
        for i in range(self.layers):
            return_sequences = (i < self.layers - 1)

            model.add(GRU(
                units=self.units,
                return_sequences=return_sequences,
                dropout=self.dropout,
                recurrent_dropout=self.dropout,
                kernel_regularizer=l1_l2(l1=0.01, l2=0.01)
            ))

            if i < self.layers - 1:
                model.add(BatchNormalization())

        # Выходные слои
        model.add(Dropout(self.dropout))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(self.dropout / 2))
        model.add(Dense(1))

        # Компиляция
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        return model


if __name__ == "__main__":
    # Тестирование LSTM модели
    print("🧪 Тестирование LSTM модели")

    if not TF_AVAILABLE:
        print("❌ TensorFlow не установлен")
        exit(1)

    # Создаем тестовые данные
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='D')

    # Имитируем цены с трендом и техническими индикаторами
    trend = np.linspace(100, 120, 200)
    noise = np.random.randn(200) * 2

    test_data = pd.DataFrame({
        'close': trend + noise,
        'volume': np.random.randint(1000, 10000, 200),
        'sma_10': (trend + noise) * 0.98,  # имитация SMA
        'rsi': np.random.uniform(30, 70, 200),  # имитация RSI
        'macd': np.random.randn(200) * 0.5
    }, index=dates)

    print(f"📊 Тестовые данные: {len(test_data)} записей, {len(test_data.columns)} признаков")

    try:
        # Создание и обучение LSTM
        lstm_model = LSTMModel(window_size=20, units=32, layers=1)

        print("🎯 Обучение LSTM модели...")
        lstm_model.fit(
            test_data,
            target_column='close',
            feature_columns=['close', 'volume', 'sma_10', 'rsi', 'macd'],
            epochs=20,  # Мало эпох для теста
            batch_size=16,
            verbose=0
        )

        # Прогнозирование
        prediction = lstm_model.predict(test_data.tail(30))
        print(f"📈 Прогноз: {prediction:.4f}")

        # Многошаговый прогноз
        multi_pred = lstm_model.predict(test_data.tail(30), steps=5, return_sequences=True)
        print(f"📊 Прогноз на 5 шагов: {multi_pred}")

        # Оценка модели
        metrics = lstm_model.evaluate(test_data.tail(50))
        print(f"📊 MAE: {metrics.get('mae', 0):.6f}")
        print(f"📊 Направленная точность: {metrics.get('directional_accuracy', 0):.1f}%")

        # Тестирование GRU
        print("\n🎯 Тестирование GRU модели...")
        gru_model = GRUModel(window_size=20, units=32, layers=1)

        gru_model.fit(
            test_data,
            target_column='close',
            feature_columns=['close', 'volume', 'sma_10', 'rsi', 'macd'],
            epochs=20,
            batch_size=16,
            verbose=0
        )

        gru_prediction = gru_model.predict(test_data.tail(30))
        print(f"📈 GRU прогноз: {gru_prediction:.4f}")

        print("✅ Тестирование LSTM/GRU завершено успешно!")

    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
        import traceback
        traceback.print_exc()
