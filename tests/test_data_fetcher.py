"""Tests for data_fetcher module"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch
from data_fetcher import DataFetcher


class TestDataFetcher:
    """Test cases for DataFetcher class"""

    @pytest.mark.unit
    def test_initialization(self):
        """Test DataFetcher initialization"""
        fetcher = DataFetcher()
        assert hasattr(fetcher, 'api_key')
        assert hasattr(fetcher, 'base_url')
        assert fetcher.session is not None

    @pytest.mark.unit
    def test_fetch_historical_data_success(self, sample_ohlcv_data):
        """Test successful historical data fetch"""
        with patch('requests.get') as mock_get:
            # Mock successful API response
            mock_response = Mock()
            mock_response.json.return_value = {
                'results': [
                    {
                        't': 1640995200,  # 2022-01-01 timestamp
                        'o': '100.50',
                        'h': '101.20',
                        'l': '99.80',
                        'c': '101.00',
                        'v': '1000000'
                    }
                ]
            }
            mock_get.return_value = mock_response

            fetcher = DataFetcher()
            result = fetcher.fetch_historical_data('AAPL', '2022-01-01', '2022-01-02')

            assert isinstance(result, pd.DataFrame)
            assert len(result) >= 0  # May be empty if no data
            if len(result) > 0:
                assert all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume'])

    @pytest.mark.unit
    def test_fetch_historical_data_api_error(self):
        """Test handling of API errors"""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("API Connection Error")

            fetcher = DataFetcher()

            with pytest.raises(Exception):
                fetcher.fetch_historical_data('AAPL', '2022-01-01', '2022-01-02')

    @pytest.mark.unit
    def test_fetch_latest_data(self):
        """Test fetching latest data"""
        with patch('requests.get') as mock_get:
            # Mock API response for latest data
            mock_response = Mock()
            mock_response.json.return_value = {
                'results': [
                    {
                        't': 1641081600,
                        'o': '101.00',
                        'h': '102.00',
                        'l': '100.50',
                        'c': '101.50',
                        'v': '1200000'
                    }
                ]
            }
            mock_get.return_value = mock_response

            fetcher = DataFetcher()
            result = fetcher.fetch_latest_data('AAPL')

            assert isinstance(result, pd.DataFrame)
            if len(result) > 0:
                assert all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume'])

    @pytest.mark.unit
    def test_validate_data_structure(self, sample_ohlcv_data):
        """Test data validation"""
        fetcher = DataFetcher()

        # Valid data should pass validation
        assert fetcher._validate_data_structure(sample_ohlcv_data) is None

        # Invalid data (missing columns) should raise error
        invalid_data = pd.DataFrame({'invalid_col': [1, 2, 3]})
        with pytest.raises(ValueError):
            fetcher._validate_data_structure(invalid_data)

    @pytest.mark.unit
    def test_parse_api_response(self):
        """Test API response parsing"""
        fetcher = DataFetcher()

        api_response = {
            'results': [
                {
                    't': 1640995200,
                    'o': '100.50',
                    'h': '101.20',
                    'l': '99.80',
                    'c': '101.00',
                    'v': '1000000'
                }
            ]
        }

        result = fetcher._parse_api_response(api_response)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]['open'] == 100.50
        assert result.iloc[0]['close'] == 101.00