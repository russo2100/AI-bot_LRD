# src/monitoring/metrics_tracker.py
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
import json
import sqlite3
import os

logger = logging.getLogger(__name__)

class MetricsTracker:
    """
    Отслеживание и мониторинг торговых метрик в реальном времени
    """

    def __init__(self, db_path: str = 'data/metrics.db'):
        self.db_path = db_path
        self.current_metrics = {}
        self.alerts = []

        # Создаем папку для БД если не существует
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Инициализация базы данных
        self._init_database()

    def _init_database(self):
        """Инициализация SQLite базы данных"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Таблица сделок
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        action TEXT NOT NULL,
                        quantity REAL NOT NULL,
                        price REAL NOT NULL,
                        pnl REAL,
                        strategy TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Таблица дневных метрик
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS daily_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL UNIQUE,
                        total_pnl REAL NOT NULL,
                        num_trades INTEGER NOT NULL,
                        win_rate REAL,
                        max_drawdown REAL,
                        sharpe_ratio REAL,
                        portfolio_value REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Таблица алертов
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        alert_type TEXT NOT NULL,
                        message TEXT NOT NULL,
                        severity TEXT DEFAULT 'INFO',
                        acknowledged INTEGER DEFAULT 0,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                conn.commit()

            logger.info("✅ База данных метрик инициализирована")

        except Exception as e:
            logger.error(f"❌ Ошибка инициализации БД: {e}")

    def log_trade(self, trade_info: Dict[str, Any]):
        """Логирование сделки"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO trades (timestamp, symbol, action, quantity, price, pnl, strategy)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_info.get('timestamp', datetime.now().isoformat()),
                    trade_info.get('symbol', ''),
                    trade_info.get('action', ''),
                    trade_info.get('quantity', 0),
                    trade_info.get('price', 0),
                    trade_info.get('pnl', 0),
                    trade_info.get('strategy', '')
                ))
                conn.commit()

            logger.info(f"📝 Сделка зафиксирована: {trade_info['action']} {trade_info.get('symbol', '')}")

        except Exception as e:
            logger.error(f"❌ Ошибка логирования сделки: {e}")

    def create_alert(self, alert_type: str, message: str, severity: str = 'INFO'):
        """Создание алерта"""
        try:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'alert_type': alert_type,
                'message': message,
                'severity': severity
            }

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO alerts (timestamp, alert_type, message, severity)
                    VALUES (?, ?, ?, ?)
                """, (alert['timestamp'], alert_type, message, severity))
                conn.commit()

            self.alerts.append(alert)

            # Логирование в зависимости от серьезности
            if severity == 'CRITICAL':
                logger.critical(f"🚨 {alert_type}: {message}")
            elif severity == 'WARNING':
                logger.warning(f"⚠️ {alert_type}: {message}")
            else:
                logger.info(f"ℹ️ {alert_type}: {message}")

        except Exception as e:
            logger.error(f"❌ Ошибка создания алерта: {e}")

    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """Получение сводки производительности за последние дни"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

            with sqlite3.connect(self.db_path) as conn:
                # Загружаем метрики
                metrics_df = pd.read_sql_query("""
                    SELECT * FROM daily_metrics 
                    WHERE date >= ? 
                    ORDER BY date DESC
                """, conn, params=(cutoff_date,))

                # Загружаем сделки
                trades_df = pd.read_sql_query("""
                    SELECT * FROM trades 
                    WHERE timestamp >= ? 
                    ORDER BY timestamp DESC
                """, conn, params=(cutoff_date,))

            # Расчет сводной статистики
            summary = {
                'period_days': days,
                'total_trades': len(trades_df),
            }

            if not metrics_df.empty:
                summary.update({
                    'current_portfolio_value': metrics_df['portfolio_value'].iloc[0] if len(metrics_df) > 0 else 0,
                    'total_pnl': metrics_df['total_pnl'].sum(),
                    'avg_daily_pnl': metrics_df['total_pnl'].mean(),
                })

            # Статистика по сделкам
            if not trades_df.empty:
                profitable_trades = trades_df[trades_df['pnl'] > 0]
                summary.update({
                    'profitable_trades': len(profitable_trades),
                    'win_rate_trades': len(profitable_trades) / len(trades_df) * 100,
                    'avg_trade_pnl': trades_df['pnl'].mean(),
                })

            return summary

        except Exception as e:
            logger.error(f"❌ Ошибка получения сводки: {e}")
            return {}


if __name__ == "__main__":
    # Тестирование системы мониторинга
    print("🧪 Тестирование системы мониторинга")

    # Инициализация трекера
    tracker = MetricsTracker('test_metrics.db')

    # Тестирование логирования сделок
    print("📝 Тестирование логирования сделок...")

    test_trade = {
        'timestamp': datetime.now().isoformat(),
        'symbol': 'SBER',
        'action': 'BUY',
        'quantity': 100,
        'price': 250.5,
        'pnl': 0,
        'strategy': 'AI_Strategy'
    }

    tracker.log_trade(test_trade)

    # Получение сводки
    print("📋 Получение сводки производительности...")
    summary = tracker.get_performance_summary(days=1)
    print(f"Всего сделок: {summary.get('total_trades', 0)}")

    # Очистка тестовых файлов
    import os
    try:
        os.remove('test_metrics.db')
        print("🧹 Тестовые файлы удалены")
    except:
        pass

    print("✅ Тестирование мониторинга завершено!")
