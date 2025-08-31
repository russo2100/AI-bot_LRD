"""Tests for preprocess module"""
import pytest
import pandas as pd
import numpy as np
from preprocess import (
    clean_data,
    normalize_data,
    denormalize_data,
    create_returns_features,
    create_time_windows,
    split_data,
    add_lagged_features,
    detect_outliers
)


class TestCleanData:
    """Test cases for clean_data function"""

    @pytest.mark.unit
    def test_clean_data_empty_dataframe(self):
        """Test cleaning empty dataframe"""
        df_empty = pd.DataFrame()
        result = clean_data(df_empty)
        assert result.empty

    @pytest.mark.unit
    def test_clean_data_normal_data(self, sample_ohlcv_data):
        """Test cleaning normal data"""
        original_length = len(sample_ohlcv_data)
        result = clean_data(sample_ohlcv_data)

        assert len(result) == original_length
        assert all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        # Test that prices are positive
        assert (result[['open', 'high', 'low', 'close']] > 0).all().all()

    @pytest.mark.unit
    def test_clean_data_negative_prices(self, sample_ohlcv_data):
        """Test handling negative prices"""
        data_with_negatives = sample_ohlcv_data.copy()
        data_with_negatives.loc[data_with_negatives.index[0], 'close'] = -10

        result = clean_data(data_with_negatives)

        # Negative price should be corrected
        assert result.loc[result.index[0], 'close'] > 0

    @pytest.mark.unit
    def test_clean_data_outliers(self, sample_ohlcv_data):
        """Test handling outliers"""
        data_with_outliers = sample_ohlcv_data.copy()
        # Create extreme outlier (> 10 std from mean)
        mean_price = sample_ohlcv_data['close'].mean()
        std_price = sample_ohlcv_data['close'].std()
        outlier_value = mean_price + 15 * std_price
        data_with_outliers.loc[data_with_outliers.index[0], 'close'] = outlier_value

        result = clean_data(data_with_outliers)

        # Outlier should be replaced
        assert abs(result.loc[result.index[0], 'close'] - sample_ohlcv_data['close'].median()) < 0.01

    @pytest.mark.unit
    def test_clean_data_ohlc_correction(self, sample_ohlcv_data):
        """Test OHLC validation and correction"""
        data_with_invalid_ohlc = sample_ohlcv_data.copy()
        # Make high lower than close (invalid)
        data_with_invalid_ohlc.loc[data_with_invalid_ohlc.index[0], 'high'] = 50

        result = clean_data(data_with_invalid_ohlc)

        # High should be corrected to be >= max(open, close)
        assert result.loc[result.index[0], 'high'] >= result.loc[result.index[0], 'close']

    @pytest.mark.unit
    def test_clean_data_missing_values(self, sample_ohlcv_data):
        """Test filling missing values"""
        data_with_nan = sample_ohlcv_data.copy()
        data_with_nan.loc[data_with_nan.index[5:8], ['open', 'high', 'low', 'close']] = np.nan

        result = clean_data(data_with_nan)

        # Missing values should be filled
        assert not result[['open', 'high', 'low', 'close']].isnull().any().any()


class TestNormalizeData:
    """Test cases for normalize_data function"""

    @pytest.mark.unit
    def test_normalize_data_minmax(self, sample_ohlcv_data):
        """Test MinMax normalization"""
        result, scalers = normalize_data(sample_ohlcv_data, method='minmax', columns=['close'])

        assert len(result) == len(sample_ohlcv_data)
        assert 'close' in result.columns
        assert 'close' in scalers
        # Values should be in [0, 1] range
        assert result['close'].min() >= 0
        assert result['close'].max() <= 1

    @pytest.mark.unit
    def test_normalize_data_standard(self, sample_ohlcv_data):
        """Test Standard normalization"""
        result, scalers = normalize_data(sample_ohlcv_data, method='standard', columns=['close'])

        assert len(result) == len(sample_ohlcv_data)
        # For standardized data, mean should be close to 0 and std close to 1 (approximately)
        assert abs(result['close'].mean()) < 0.1
        assert abs(result['close'].std() - 1) < 0.5  # Allow some tolerance for small sample

    @pytest.mark.unit
    def test_normalize_data_robust(self, sample_ohlcv_data):
        """Test Robust normalization"""
        result, scalers = normalize_data(sample_ohlcv_data, method='robust', columns=['close'])

        assert len(result) == len(sample_ohlcv_data)
        assert 'close' in scalers

    @pytest.mark.unit
    def test_normalize_data_invalid_method(self, sample_ohlcv_data):
        """Test invalid normalization method"""
        with pytest.raises(ValueError):
            normalize_data(sample_ohlcv_data, method='invalid_method')

    @pytest.mark.unit
    def test_dornormalize_data(self, sample_ohlcv_data):
        """Test denormalization"""
        # Normalize data
        normalized, scalers = normalize_data(sample_ohlcv_data, method='minmax', columns=['close'])

        # Denormalize back
        denormalized = denormalize_data(normalized, scalers)

        # Values should be close to original (within reasonable tolerance due to floating point precision)
        np.testing.assert_array_almost_equal(
            denormalized['close'].values,
            sample_ohlcv_data['close'].values,
            decimal=5
        )


class TestCreateReturnsFeatures:
    """Test cases for create_returns_features function"""

    @pytest.mark.unit
    def test_create_returns_features_normal(self, sample_ohlcv_data):
        """Test creation of return features"""
        result = create_returns_features(sample_ohlcv_data)

        expected_features = [
            'returns_1d', 'returns_5d', 'returns_20d',
            'log_returns_1d', 'log_returns_5d',
            'volatility_10d', 'volatility_20d',
            'price_zscore_20d'
        ]

        for feature in expected_features:
            assert feature in result.columns

    @pytest.mark.unit
    def test_create_returns_features_empty_data(self):
        """Test with empty dataframe"""
        df_empty = pd.DataFrame()
        result = create_returns_features(df_empty)
        assert result.empty

    @pytest.mark.unit
    def test_create_returns_features_no_close(self):
        """Test with no close column"""
        df_no_close = pd.DataFrame({'open': [100, 101], 'volume': [1000, 1100]})
        result = create_returns_features(df_no_close)
        assert len(result) == len(df_no_close)
        # Should not have return features
        assert 'returns_1d' not in result.columns


class TestCreateTimeWindows:
    """Test cases for create_time_windows function"""

    @pytest.mark.unit
    def test_create_time_windows_normal(self, sample_data_with_indicators):
        """Test normal time window creation"""
        X, y = create_time_windows(sample_data_with_indicators, window_size=10)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(X) > 0
        assert len(y) > 0
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 10  # window_size

    @pytest.mark.unit
    def test_create_time_windows_insufficient_data(self):
        """Test with insufficient data"""
        small_data = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        })
        X, y = create_time_windows(small_data, window_size=10)

        assert len(X) == 0
        assert len(y) == 0

    @pytest.mark.unit
    def test_create_time_windows_no_numeric_columns(self):
        """Test with no numeric columns"""
        no_numeric_data = pd.DataFrame({'text': ['a', 'b', 'c'], 'close': ['100', '101', '102']})
        X, y = create_time_windows(no_numeric_data, window_size=2)

        # Should still work if target column is present
        assert len(X) > 0
        assert len(y) > 0


class TestSplitData:
    """Test cases for split_data function"""

    @pytest.mark.unit
    def test_split_data_default(self, sample_ohlcv_data):
        """Test default 80/10/10 split"""
        train, val, test = split_data(sample_ohlcv_data)

        assert len(train) + len(val) + len(test) == len(sample_ohlcv_data)
        assert len(train) >= len(val)
        assert len(val) == len(test)

    @pytest.mark.unit
    def test_split_data_custom_ratios(self, sample_ohlcv_data):
        """Test custom train/val ratios"""
        train_ratio = 0.7
        val_ratio = 0.2
        train, val, test = split_data(sample_ohlcv_data, train_ratio=train_ratio, val_ratio=val_ratio)

        expected_train_size = int(len(sample_ohlcv_data) * train_ratio)
        expected_val_size = int(len(sample_ohlcv_data) * val_ratio)

        assert len(train) == expected_train_size
        assert len(val) == expected_val_size


class TestAddLaggedFeatures:
    """Test cases for add_lagged_features function"""

    @pytest.mark.unit
    def test_add_lagged_features_normal(self, sample_ohlcv_data):
        """Test adding lagged features"""
        columns = ['close', 'volume']
        lags = [1, 2, 3]
        result = add_lagged_features(sample_ohlcv_data, columns, lags)

        expected_new_columns = [
            'close_lag_1', 'close_lag_2', 'close_lag_3',
            'volume_lag_1', 'volume_lag_2', 'volume_lag_3'
        ]

        for col in expected_new_columns:
            assert col in result.columns

        # First few rows of lagged features should be 0 (filled NaN)
        assert result['close_lag_1'].iloc[0] == 0
        assert result['close_lag_2'].iloc[0:2].tolist() == [0, 0]

    @pytest.mark.unit
    def test_add_lagged_features_missing_columns(self, sample_ohlcv_data):
        """Test handling missing columns"""
        missing_columns = ['close', 'missing_column']
        result = add_lagged_features(sample_ohlcv_data, missing_columns, [1])

        # Should create features only for existing columns
        assert 'close_lag_1' in result.columns
        assert 'missing_column_lag_1' not in result.columns


class TestDetectOutliers:
    """Test cases for detect_outliers function"""

    @pytest.mark.unit
    def test_detect_outliers_iqr(self, sample_ohlcv_data):
        """Test IQR outlier detection"""
        # Create artificial outlier
        outlier_data = sample_ohlcv_data.copy()
        outlier_data.loc[outlier_data.index[-1], 'close'] *= 10  # Create huge outlier

        outliers = detect_outliers(outlier_data, 'close', method='iqr')
        assert isinstance(outliers, pd.Series)
        assert outliers.sum() >= 1  # Should detect at least one outlier

    @pytest.mark.unit
    def test_detect_outliers_zscore(self, sample_ohlcv_data):
        """Test z-score outlier detection"""
        outliers = detect_outliers(sample_ohlcv_data, 'close', method='zscore')
        assert isinstance(outliers, pd.Series)
        assert outliers.sum() <= len(sample_ohlcv_data) * 0.05  # Should not flag too many normal points

    @pytest.mark.unit
    def test_detect_outliers_missing_column(self, sample_ohlcv_data):
        """Test detection on non-existing column"""
        outliers = detect_outliers(sample_ohlcv_data, 'non_existing_column')
        assert isinstance(outliers, pd.Series)
        assert outliers.sum() == 0  # No outliers should be detected

    @pytest.mark.unit
    def test_detect_outliers_invalid_method(self, sample_ohlcv_data):
        """Test invalid detection method"""
        with pytest.raises(ValueError):
            detect_outliers(sample_ohlcv_data, 'close', method='invalid_method')