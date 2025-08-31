"""
Configuration management using Pydantic models
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseSettings, Field
from pydantic.env import SettingsConfigDict


class TradingSettings(BaseSettings):
    """Trading specific settings"""
    symbols: List[str] = Field(default=['SBER', 'MOEX'], description="Default trading symbols")
    risk_per_trade: float = Field(default=0.02, description="Risk per trade percentage")
    max_positions: int = Field(default=5, description="Maximum open positions")
    commission: float = Field(default=0.001, description="Trading commission")
    initial_capital: float = Field(default=100000, description="Initial trading capital")


class APISettings(BaseSettings):
    """API related settings"""
    tinkoff_token: Optional[str] = Field(default=None, description="Tinkoff Invest API token")
    tinkoff_account_id: Optional[str] = Field(default=None, description="Tinkoff account ID")
    sandbox: bool = Field(default=True, description="Use sandbox mode")


class ModelSettings(BaseSettings):
    """ML models related settings"""
    model_dir: str = Field(default="models", description="Directory for saved models")
    retrain_interval: int = Field(default=7, description="Retraining interval in days")
    use_ai_models: bool = Field(default=True, description="Enable AI model predictions")


class BacktestSettings(BaseSettings):
    """Backtesting related settings"""
    default_strategy: str = Field(default="ma_crossover", description="Default backtest strategy")
    start_date: str = Field(default="2023-01-01", description="Default backtest start date")
    end_date: str = Field(default="2023-12-31", description="Default backtest end date")
    commission: float = Field(default=0.001, description="Backtest commission")


class WebSettings(BaseSettings):
    """Web interface settings"""
    host: str = Field(default="0.0.0.0", description="Web server host")
    port: int = Field(default=5000, description="Web server port")
    debug: bool = Field(default=True, description="Enable debug mode")


class LoggingSettings(BaseSettings):
    """Logging configuration"""
    level: str = Field(default="INFO", description="Logging level")
    file_path: str = Field(default="trading_bot.log", description="Log file path")
    max_file_size: int = Field(default=10*1024*1024, description="Max log file size")
    backup_count: int = Field(default=5, description="Number of backup log files")


class DataSettings(BaseSettings):
    """Data management settings"""
    data_dir: str = Field(default="data", description="Main data directory")
    raw_data_dir: str = Field(default="data/raw", description="Raw data storage")
    processed_data_dir: str = Field(default="data/processed", description="Processed data storage")
    cache_enabled: bool = Field(default=True, description="Enable data caching")


class RiskSettings(BaseSettings):
    """Risk management settings"""
    max_drawdown: float = Field(default=0.1, description="Maximum allowed drawdown")
    stop_loss_percent: float = Field(default=0.05, description="Default stop loss percentage")
    take_profit_percent: float = Field(default=0.1, description="Default take profit percentage")
    var_confidence: float = Field(default=0.95, description="VaR confidence level")


class NotificationSettings(BaseSettings):
    """Notification settings"""
    email_enabled: bool = Field(default=False, description="Enable email notifications")
    email_smtp_server: Optional[str] = Field(default=None, description="SMTP server")
    email_port: int = Field(default=587, description="SMTP port")
    email_username: Optional[str] = Field(default=None, description="Email username")
    email_password: Optional[str] = Field(default=None, description="Email password")
    email_recipients: List[str] = Field(default=[], description="Email recipients")

    telegram_enabled: bool = Field(default=False, description="Enable Telegram notifications")
    telegram_bot_token: Optional[str] = Field(default=None, description="Telegram bot token")
    telegram_chat_id: Optional[str] = Field(default=None, description="Telegram chat ID")


class Settings(BaseSettings):
    """Main application settings"""
    trading: TradingSettings = TradingSettings()
    api: APISettings = APISettings()
    models: ModelSettings = ModelSettings()
    backtest: BacktestSettings = BacktestSettings()
    web: WebSettings = WebSettings()
    logging: LoggingSettings = LoggingSettings()
    data: DataSettings = DataSettings()
    risk: RiskSettings = RiskSettings()
    notifications: NotificationSettings = NotificationSettings()

    class Config(SettingsConfigDict):
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_prefix = "TRADING_"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Create necessary directories
def create_directories():
    """Create necessary directories from settings"""
    dirs_to_create = [
        settings.models.model_dir,
        settings.data.data_dir,
        settings.data.raw_data_dir,
        settings.data.processed_data_dir,
        "logs",
        "backtest_results",
        "templates"
    ]

    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def validate_settings():
    """Validate critical settings"""
    if not settings.api.tinkoff_token:
        print("⚠️ Warning: TINKOFF_TOKEN not set, running in demo mode")

    # Validate directories
    create_directories()

    return True


# Initialize on import
validate_settings()