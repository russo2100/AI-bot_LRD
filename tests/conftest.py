"""Common fixtures for tests"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def sample_ohlcv_data():
    """Sample OHLCV data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    np.random.seed(42)  # For reproducibility

    # Generate realistic price data
    base_price = 100.0
    prices = []
    current_price = base_price

    for _ in dates:
        # Random walk with some volatility
        change = np.random.normal(0, 2.0)  # Mean=0, Std=2
        current_price += change
        current_price = max(current_price, 10)  # Floor price
        prices.append(current_price)

    prices = np.array(prices)

    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.01, len(dates))),
        'high': prices * (1 + np.random.normal(0, 0.02, len(dates))),
        'low': prices * (1 - np.random.normal(0, 0.02, len(dates))),
        'close': prices,
        'volume': np.random.uniform(100000, 1000000, len(dates))
    }, index=dates)

    # Ensure high >= open, close and low <= min(open, close)
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)

    return data


@pytest.fixture
def sample_data_with_indicators(sample_ohlcv_data):
    """Sample data with technical indicators"""
    data = sample_ohlcv_data.copy()

    # Add some basic indicators for testing
    data['rsi'] = np.random.uniform(30, 70, len(data))
    data['sma_20'] = data['close'].rolling(20).mean()
    data['sma_50'] = data['close'].rolling(50).mean()
    data['macd'] = data['close'].ewm(span=12).mean() - data['close'].ewm(span=26).mean()
    data['volume_sma'] = data['volume'].rolling(20).mean()

    return data


@pytest.fixture
def mock_data_fetcher(sample_ohlcv_data):
    """Mock data fetcher for testing"""
    class MockDataFetcher:
        def __init__(self):
            self.data = sample_ohlcv_data

        def get_historical_data(self, symbol, start_date, end_date):
            return self.data

        def fetch_latest_data(self, symbol):
            return self.data.iloc[-1:]

    return MockDataFetcher()


@pytest.fixture
def mock_ai_service():
    """Mock AI service for testing"""
    class MockAIService:
        def predict(self, data):
            return {
                'prediction': np.random.choice(['BUY', 'SELL', 'HOLD']),
                'confidence': np.random.uniform(0.5, 0.95),
                'prediction_value': np.random.uniform(-0.1, 0.1)
            }

        def train_model(self, data, target_column):
            return {'accuracy': 0.85, 'loss': 0.15}

    return MockAIService()


@pytest.fixture
def temp_model_path(tmp_path):
    """Temporary path for model tests"""
    return str(tmp_path / "test_model.pkl")


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        'general': {
            'symbols': ['AAPL', 'MSFT', 'GOOGL'],
            'api_token': 'test_token',
            'log_level': 'INFO'
        },
        'trading': {
            'initial_capital': 100000,
            'commission': 0.001,
            'max_position_size': 0.1
        },
        'risk': {
            'max_drawdown': 0.1,
            'stop_loss': 0.05,
            'take_profit': 0.1
        },
        'strategies': {
            'default': 'ma_crossover',
            'enabled_strategies': ['ma_crossover', 'rsi', 'bollinger']
        }
    }