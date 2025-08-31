# src/config.py
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    # API настройки
    token: str
    account_id: Optional[str]
    sandbox: bool

    # Логирование
    log_level: str
    log_file: str

    # Торговые настройки
    default_risk_per_trade: float
    max_positions: int

    # Пути
    data_dir: str
    models_dir: str
    logs_dir: str

def load_settings() -> Settings:
    return Settings(
        # API
        token=os.getenv("TINKOFF_TOKEN", ""),
        account_id=os.getenv("TINKOFF_ACCOUNT_ID"),
        sandbox=os.getenv("SANDBOX", "true").lower() in ("1", "true", "yes"),

        # Логирование
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE", "logs/trading.log"),

        # Торговля
        default_risk_per_trade=float(os.getenv("DEFAULT_RISK_PER_TRADE", "0.02")),
        max_positions=int(os.getenv("MAX_POSITIONS", "5")),

        # Пути
        data_dir=os.getenv("DATA_DIR", "data"),
        models_dir=os.getenv("MODELS_DIR", "data/models"),
        logs_dir=os.getenv("LOGS_DIR", "logs"),
    )

# Создать необходимые директории при загрузке
def create_directories():
    settings = load_settings()
    for directory in [settings.data_dir, settings.models_dir, settings.logs_dir]:
        os.makedirs(directory, exist_ok=True)
        os.makedirs(f"{directory}/raw", exist_ok=True)
        os.makedirs(f"{directory}/processed", exist_ok=True)

if __name__ == "__main__":
    create_directories()
    print("✅ Директории созданы")
