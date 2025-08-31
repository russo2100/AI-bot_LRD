#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Мониторинг и алерты для AI-торгового бота T-Bank
Интеграция с Prometheus, алерты через email и Telegram
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import threading
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pandas as pd
import psutil
from prometheus_client import (
    CollectorRegistry, Gauge, Counter, Histogram,
    push_to_gateway, generate_latest, REGISTRY
)

from config.settings import settings

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Сборщик метрик для Prometheus"""

    def __init__(self):
        self.registry = CollectorRegistry()

        # Торговые метрики
        self.trade_count = Counter(
            'trading_bot_trades_total',
            'Total number of trades executed',
            ['symbol', 'strategy'],
            registry=self.registry
        )

        self.portfolio_value = Gauge(
            'trading_bot_portfolio_value',
            'Current portfolio value',
            registry=self.registry
        )

        self.strategy_performance = Gauge(
            'trading_bot_strategy_performance',
            'Strategy performance metrics',
            ['strategy', 'metric'],
            registry=self.registry
        )

        self.drawdown_current = Gauge(
            'trading_bot_drawdown_current',
            'Current drawdown percentage',
            registry=self.registry
        )

        self.risk_exposure = Gauge(
            'trading_bot_risk_exposure',
            'Current risk exposure',
            registry=self.registry
        )

        # Системные метрики
        self.cpu_usage = Gauge(
            'trading_bot_cpu_usage',
            'CPU usage percentage',
            registry=self.registry
        )

        self.memory_usage = Gauge(
            'trading_bot_memory_usage',
            'Memory usage percentage',
            registry=self.registry
        )

        self.disk_usage = Gauge(
            'trading_bot_disk_usage',
            'Disk usage percentage',
            registry=self.registry
        )

        # Ошибки и события
        self.errors_total = Counter(
            'trading_bot_errors_total',
            'Total number of errors',
            ['type'],
            registry=self.registry
        )

        self.warnings_total = Counter(
            'trading_bot_warnings_total',
            'Total number of warnings',
            ['type'],
            registry=self.registry
        )

    def update_trading_metrics(self, portfolio_value: float,
                              drawdown: float, risk_exposure: float):
        """Обновление торговых метрик"""
        self.portfolio_value.set(portfolio_value)
        self.drawdown_current.set(drawdown)
        self.risk_exposure.set(risk_exposure)

    def update_system_metrics(self):
        """Обновление системных метрик"""
        # CPU и память
        self.cpu_usage.set(psutil.cpu_percent(interval=1))
        self.memory_usage.set(psutil.virtual_memory().percent)

        # Диск
        disk = psutil.disk_usage('/')
        self.disk_usage.set(disk.percent)

    def record_trade(self, symbol: str, strategy: str):
        """Запись выполненной сделки"""
        self.trade_count.labels(symbol=symbol, strategy=strategy).inc()

    def record_strategy_metric(self, strategy: str, metric: str, value: float):
        """Запись метрики стратегии"""
        self.strategy_performance.labels(
            strategy=strategy, metric=metric
        ).set(value)

    def record_error(self, error_type: str):
        """Запись ошибки"""
        self.errors_total.labels(type=error_type).inc()

    def record_warning(self, warning_type: str):
        """Запись предупреждения"""
        self.warnings_total.labels(type=warning_type).inc()

    def get_metrics_json(self) -> str:
        """Получение метрик в JSON формате"""
        return generate_latest(self.registry).decode('utf-8')


class AlertManager:
    """Менеджер алертов"""

    def __init__(self):
        self.alert_history = []
        self.notification_settings = settings.notifications

    def send_alert(self, alert_type: str, message: str, data: dict = None):
        """Отправка алерта"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'data': data or {},
            'severity': self._get_alert_severity(alert_type)
        }

        self.alert_history.append(alert)
        logger.warning(f"ALERT [{alert_type}]: {message}")

        # Отправка уведомления
        self._send_notification(alert)

    def check_trading_conditions(self, bot_metrics: dict):
        """Проверка торговых условий для алертов"""
        conditions = [
            # Критическое снижение капитала
            lambda: bot_metrics.get('portfolio_value', 0) <
                   settings.trading.initial_capital * 0.5,
            "Критическое снижение капитала",

            # Высокий drawdown
            lambda: bot_metrics.get('drawdown', 0) > settings.risk.max_drawdown,
            f"Drawdown превысил {settings.risk.max_drawdown:.1%}",

            # Превышение максимального риска
            lambda: bot_metrics.get('risk_exposure', 0) > 1.0,
            "Превышение максимального риска",

            # Отсутствие активности
            lambda: (datetime.now() - bot_metrics.get('last_trade', datetime.now())).total_seconds() > 3600,
            "Отсутствие торговой активности более 1 часа",
        ]

        for i in range(0, len(conditions), 2):
            condition_func = conditions[i]
            message = conditions[i+1]

            try:
                if condition_func():
                    self.send_alert('trading_alert', message, bot_metrics)
            except Exception as e:
                logger.error(f"Error checking condition: {e}")

    def _send_notification(self, alert: dict):
        """Отправка уведомления через email и Telegram"""
        message = self._format_alert_message(alert)

        # Email алерты
        if self.notification_settings.email_enabled:
            self._send_email_alert(alert, message)

        # Telegram алерты
        if self.notification_settings.telegram_enabled:
            self._send_telegram_alert(alert, message)

    def _send_email_alert(self, alert: dict, message: str):
        """Отправка email алерта"""
        try:
            if not all([
                self.notification_settings.email_smtp_server,
                self.notification_settings.email_username,
                self.notification_settings.email_password
            ]):
                logger.warning("Email settings incomplete, skipping email alert")
                return

            # Создание сообщения
            msg = MIMEMultipart()
            msg['From'] = self.notification_settings.email_username
            msg['To'] = ', '.join(self.notification_settings.email_recipients)
            msg['Subject'] = f'AI Trading Bot Alert: {alert["type"]}'

            # Добавление тела сообщения
            body = f"""
AI Trading Bot Alert
===================

Type: {alert['type']}
Severity: {alert['severity']}
Timestamp: {alert['timestamp']}
Message: {alert['message']}

{alert['data'] and json.dumps(alert['data'], indent=2, ensure_ascii=False) or ''}

===================
This is an automated message from AI Trading Bot monitoring system.
            """

            msg.attach(MIMEText(body, 'plain', 'utf-8'))

            # Отправка
            server = smtplib.SMTP(
                self.notification_settings.email_smtp_server,
                self.notification_settings.email_port
            )
            server.starttls()
            server.login(
                self.notification_settings.email_username,
                self.notification_settings.email_password
            )
            server.send_message(msg)
            server.quit()

            logger.info("Email alert sent successfully")

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    def _send_telegram_alert(self, alert: dict, message: str):
        """Отправка Telegram алерта"""
        try:
            if not all([
                self.notification_settings.telegram_bot_token,
                self.notification_settings.telegram_chat_id
            ]):
                logger.warning("Telegram settings incomplete, skipping Telegram alert")
                return

            # Здесь можно добавить код для отправки Telegram сообщение
            # В продакшене нужно использовать python-telegram-bot

            logger.info("Telegram alert sent successfully (stub)")

        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")

    def _format_alert_message(self, alert: dict) -> str:
        """Форматирование сообщения алерта"""
        return f"""🚨 AI Trading Bot Alert

📊 Type: {alert['type']}
⚠️ Severity: {alert['severity']}
🕐 Time: {alert['timestamp']}
📝 Message: {alert['message']}

{alert['data'] and f'📈 Data: {json.dumps(alert["data"], ensure_ascii=False)}' or ''}
        """

    def _get_alert_severity(self, alert_type: str) -> str:
        """Определение уровня серьезности алерта"""
        severity_map = {
            'trading_alert': 'HIGH',
            'system_alert': 'MEDIUM',
            'monitoring_alert': 'LOW',
            'info': 'INFO'
        }
        return severity_map.get(alert_type, 'UNKNOWN')

    def get_alert_history(self, limit: int = 50) -> List[dict]:
        """Получение истории алертов"""
        return self.alert_history[-limit:] if len(self.alert_history) > limit else self.alert_history

    def get_alert_stats(self) -> dict:
        """Получение статистики по алертам"""
        if not self.alert_history:
            return {'total': 0, 'by_type': {}, 'by_severity': {}}

        stats = {
            'total': len(self.alert_history),
            'by_type': {},
            'by_severity': {}
        }

        for alert in self.alert_history:
            alert_type = alert['type']
            severity = alert['severity']

            stats['by_type'][alert_type] = stats['by_type'].get(alert_type, 0) + 1
            stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1

        return stats


class MonitoringSystem:
    """Интегрированная система мониторинга"""

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.is_monitoring = False
        self.monitor_thread = None

        # Интеграция с Prometheus pushgateway
        self.prometheus_gateway = os.getenv('PROMETHEUS_PUSHGATEWAY_URL')

    def start_monitoring(self):
        """Запуск мониторинга"""
        if self.is_monitoring:
            logger.info("Monitoring already running")
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Monitoring system started")

    def stop_monitoring(self):
        """Остановка мониторинга"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Monitoring system stopped")

    def _monitoring_loop(self):
        """Основной цикл мониторинга"""
        while self.is_monitoring:
            try:
                # Обновление системных метрик
                self.metrics_collector.update_system_metrics()

                # Проверка системы и отправка алертов
                self._check_system_health()

                # Отправка метрик в Prometheus
                if self.prometheus_gateway:
                    self._push_metrics_to_prometheus()

                # Ожидание перед следующим циклом
                time.sleep(60)  # Каждую минуту

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self.metrics_collector.record_error('monitoring_loop_error')
                time.sleep(60)

    def _check_system_health(self):
        """Проверка состояния системы"""
        # Проверка CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 90:
            self.alert_manager.send_alert(
                'system_alert',
                f"High CPU usage: {cpu_percent:.1f}%",
                {'cpu_percent': cpu_percent}
            )

        # Проверка памяти
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            self.alert_manager.send_alert(
                'system_alert',
                f"High memory usage: {memory.percent:.1f}%",
                {'memory_percent': memory.percent}
            )

        # Проверка диска
        disk = psutil.disk_usage('/')
        if disk.percent > 95:
            self.alert_manager.send_alert(
                'system_alert',
                f"Critical disk usage: {disk.percent:.1f}%",
                {'disk_percent': disk.percent}
            )

    def _push_metrics_to_prometheus(self):
        """Отправка метрик в Prometheus Pushgateway"""
        try:
            push_to_gateway(
                self.prometheus_gateway,
                job='ai_trading_bot',
                registry=self.metrics_collector.registry
            )
        except Exception as e:
            logger.error(f"Failed to push metrics to Prometheus: {e}")

    def update_trading_metrics(self, bot_metrics: dict):
        """Обновление торговых метрик"""
        try:
            portfolio_value = bot_metrics.get('portfolio_value', 0)
            drawdown = bot_metrics.get('drawdown', 0)
            risk_exposure = bot_metrics.get('risk_exposure', 0)

            self.metrics_collector.update_trading_metrics(
                portfolio_value, drawdown, risk_exposure
            )

            # Проверка условий для алертов
            self.alert_manager.check_trading_conditions(bot_metrics)

        except Exception as e:
            logger.error(f"Error updating trading metrics: {e}")

    def record_trade(self, symbol: str, strategy: str):
        """Запись выполнения сделки"""
        self.metrics_collector.record_trade(symbol, strategy)
        logger.info(f"Trade recorded: {symbol} via {strategy}")

    def record_stratgy_metric(self, strategy: str, metric: str, value: float):
        """Запись метрики стратегии"""
        self.metrics_collector.record_strategy_metric(strategy, metric, value)

    def get_status(self) -> dict:
        """Получение статуса мониторинга"""
        return {
            'is_monitoring': self.is_monitoring,
            'alert_stats': self.alert_manager.get_alert_stats(),
            'system_metrics': {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            },
            'prometheus_enabled': bool(self.prometheus_gateway)
        }


# Глобальный экземпляр системы мониторинга
monitoring_system = MonitoringSystem()


def start_monitoring():
    """Запуск системы мониторинга"""
    monitoring_system.start_monitoring()


def stop_monitoring():
    """Остановка системы мониторинга"""
    monitoring_system.stop_monitoring()


def get_monitor_status():
    """Получение статуса мониторинга"""
    return monitoring_system.get_status()