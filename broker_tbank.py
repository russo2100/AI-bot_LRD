# src/execution/broker_tbank.py
from __future__ import annotations

import logging
from uuid import uuid4
from datetime import datetime, timedelta, timezone
from typing import List, Optional
import pandas as pd
from tinkoff.invest import (
    Client,
    CandleInterval,
    OrderDirection,
    OrderType,
    Quotation,
    GetOrdersRequest,
)
from tinkoff.invest.constants import INVEST_GRPC_API, INVEST_GRPC_API_SANDBOX
from src.config import load_settings

logger = logging.getLogger(__name__)

class TBankBroker:
    """
    Обёртка над Tinkoff Invest API v2 с расширенной функциональностью
    Поддерживает песочницу и боевой режим
    """

    def __init__(self):
        self.settings = load_settings()
        self.target = INVEST_GRPC_API_SANDBOX if self.settings.sandbox else INVEST_GRPC_API
        if not self.settings.token:
            raise ValueError("❌ TINKOFF_TOKEN не найден в переменных окружения")

        # Настройка логирования
        logging.basicConfig(level=getattr(logging, self.settings.log_level.upper(), logging.INFO))

        mode = "🧪 ПЕСОЧНИЦА" if self.settings.sandbox else "⚠️ БОЕВОЙ РЕЖИМ"
        logger.info(f"🤖 TBankBroker инициализирован в режиме: {mode}")

    def _client(self) -> Client:
        """Создание клиента API"""
        return Client(self.settings.token, target=self.target)

    def get_accounts(self) -> List:
        """Получить список счетов"""
        try:
            with self._client() as client:
                response = client.users.get_accounts()
                logger.info(f"📊 Найдено счетов: {len(response.accounts)}")
                return response.accounts
        except Exception as e:
            logger.error(f"❌ Ошибка получения счетов: {e}")
            raise

    def get_portfolio(self, account_id: Optional[str] = None) -> dict:
        """Получить информацию о портфеле"""
        try:
            account_id = account_id or self.settings.account_id
            if not account_id:
                raise ValueError("account_id не указан")

            with self._client() as client:
                portfolio = client.operations.get_portfolio(account_id=account_id)

                portfolio_info = {
                    "total_amount": self._quotation_to_float(portfolio.total_amount_shares),
                    "total_amount_bonds": self._quotation_to_float(portfolio.total_amount_bonds),
                    "total_amount_etf": self._quotation_to_float(portfolio.total_amount_etf),
                    "total_amount_currencies": self._quotation_to_float(portfolio.total_amount_currencies),
                    "positions": []
                }

                for position in portfolio.positions:
                    pos_info = {
                        "figi": position.figi,
                        "quantity": self._quotation_to_float(position.quantity),
                        "current_price": self._quotation_to_float(position.current_price),
                        "average_position_price": self._quotation_to_float(position.average_position_price),
                    }
                    portfolio_info["positions"].append(pos_info)

                return portfolio_info

        except Exception as e:
            logger.error(f"❌ Ошибка получения портфеля: {e}")
            raise

    def get_orderbook(self, figi: str, depth: int = 20) -> dict:
        """Получить стакан цен"""
        try:
            with self._client() as client:
                orderbook = client.market_data.get_order_book(figi=figi, depth=depth)

                result = {
                    "figi": figi,
                    "depth": depth,
                    "bids": [(self._quotation_to_float(bid.price), bid.quantity) for bid in orderbook.bids],
                    "asks": [(self._quotation_to_float(ask.price), ask.quantity) for ask in orderbook.asks],
                    "last_price": self._quotation_to_float(orderbook.last_price) if orderbook.last_price else None,
                    "close_price": self._quotation_to_float(orderbook.close_price) if orderbook.close_price else None,
                }

                logger.info(f"📖 Стакан получен для {figi}: bids={len(result['bids'])}, asks={len(result['asks'])}")
                return result

        except Exception as e:
            logger.error(f"❌ Ошибка получения стакана {figi}: {e}")
            raise

    def get_candles(self, figi: str, days: int = 30, 
                   interval: CandleInterval = CandleInterval.CANDLE_INTERVAL_HOUR) -> List:
        """Получить исторические свечи"""
        try:
            to_dt = datetime.now(timezone.utc)
            from_dt = to_dt - timedelta(days=days)

            with self._client() as client:
                iterator = client.market_data.get_candles(
                    figi=figi,
                    from_=from_dt,
                    to=to_dt,
                    interval=interval,
                )
                candles = list(iterator.candles)

            logger.info(f"📈 Получено {len(candles)} свечей для {figi} за {days} дней")
            return candles

        except Exception as e:
            logger.error(f"❌ Ошибка получения свечей {figi}: {e}")
            raise

    def place_market_order(self, figi: str, quantity: int, buy: bool = True, 
                          account_id: Optional[str] = None) -> dict:
        """Разместить рыночную заявку"""
        try:
            account_id = account_id or self.settings.account_id
            if not account_id:
                raise ValueError("account_id не указан")

            with self._client() as client:
                order_id = str(uuid4())
                direction = OrderDirection.ORDER_DIRECTION_BUY if buy else OrderDirection.ORDER_DIRECTION_SELL

                response = client.orders.post_order(
                    figi=figi,
                    quantity=quantity,
                    direction=direction,
                    account_id=account_id,
                    order_type=OrderType.ORDER_TYPE_MARKET,
                    order_id=order_id,
                )

                action = "🟢 Покупка" if buy else "🔴 Продажа"
                logger.info(f"{action} {quantity} шт. {figi} (рыночная заявка)")

                return {
                    "order_id": response.order_id,
                    "execution_report_status": response.execution_report_status,
                    "lots_requested": response.lots_requested,
                    "lots_executed": response.lots_executed,
                    "initial_order_price": self._quotation_to_float(response.initial_order_price),
                    "executed_order_price": self._quotation_to_float(response.executed_order_price),
                    "total_order_amount": self._quotation_to_float(response.total_order_amount),
                    "direction": "BUY" if buy else "SELL",
                    "figi": figi,
                    "quantity": quantity
                }

        except Exception as e:
            logger.error(f"❌ Ошибка размещения рыночной заявки {figi}: {e}")
            raise

    def place_limit_order(self, figi: str, quantity: int, price: float, 
                         buy: bool = True, account_id: Optional[str] = None) -> dict:
        """Разместить лимитную заявку"""
        try:
            account_id = account_id or self.settings.account_id
            if not account_id:
                raise ValueError("account_id не указан")

            with self._client() as client:
                order_id = str(uuid4())
                direction = OrderDirection.ORDER_DIRECTION_BUY if buy else OrderDirection.ORDER_DIRECTION_SELL

                # Преобразование цены в Quotation
                units = int(price)
                nano = int((price - units) * 1_000_000_000)
                price_quotation = Quotation(units=units, nano=nano)

                response = client.orders.post_order(
                    figi=figi,
                    quantity=quantity,
                    direction=direction,
                    account_id=account_id,
                    order_type=OrderType.ORDER_TYPE_LIMIT,
                    price=price_quotation,
                    order_id=order_id,
                )

                action = "🟢 Покупка" if buy else "🔴 Продажа"
                logger.info(f"{action} {quantity} шт. {figi} по цене {price:.2f} (лимитная заявка)")

                return {
                    "order_id": response.order_id,
                    "execution_report_status": response.execution_report_status,
                    "lots_requested": response.lots_requested,
                    "price": price,
                    "direction": "BUY" if buy else "SELL",
                    "figi": figi,
                    "quantity": quantity
                }

        except Exception as e:
            logger.error(f"❌ Ошибка размещения лимитной заявки {figi}: {e}")
            raise

    def cancel_order(self, order_id: str, account_id: Optional[str] = None) -> bool:
        """Отменить заявку"""
        try:
            account_id = account_id or self.settings.account_id
            if not account_id:
                raise ValueError("account_id не указан")

            with self._client() as client:
                client.orders.cancel_order(account_id=account_id, order_id=order_id)
                logger.info(f"❌ Заявка {order_id} отменена")
                return True

        except Exception as e:
            logger.error(f"❌ Ошибка отмены заявки {order_id}: {e}")
            return False

    def cancel_all_orders(self, account_id: Optional[str] = None) -> int:
        """Отменить все активные заявки"""
        try:
            account_id = account_id or self.settings.account_id
            if not account_id:
                raise ValueError("account_id не указан")

            with self._client() as client:
                orders = client.orders.get_orders(account_id=account_id).orders
                cancelled_count = 0

                for order in orders:
                    try:
                        client.orders.cancel_order(account_id=account_id, order_id=order.order_id)
                        cancelled_count += 1
                    except Exception as e:
                        logger.warning(f"⚠️ Не удалось отменить заявку {order.order_id}: {e}")

                logger.info(f"🧹 Отменено заявок: {cancelled_count} из {len(orders)}")
                return cancelled_count

        except Exception as e:
            logger.error(f"❌ Ошибка отмены всех заявок: {e}")
            return 0

    def get_active_orders(self, account_id: Optional[str] = None) -> List[dict]:
        """Получить активные заявки"""
        try:
            account_id = account_id or self.settings.account_id
            if not account_id:
                raise ValueError("account_id не указан")

            with self._client() as client:
                orders = client.orders.get_orders(account_id=account_id).orders

                active_orders = []
                for order in orders:
                    order_info = {
                        "order_id": order.order_id,
                        "figi": order.figi,
                        "direction": "BUY" if order.direction == OrderDirection.ORDER_DIRECTION_BUY else "SELL",
                        "quantity": order.lots_requested,
                        "price": self._quotation_to_float(order.initial_order_price),
                        "order_type": "MARKET" if order.order_type == OrderType.ORDER_TYPE_MARKET else "LIMIT",
                        "execution_report_status": order.execution_report_status
                    }
                    active_orders.append(order_info)

                logger.info(f"📋 Активных заявок: {len(active_orders)}")
                return active_orders

        except Exception as e:
            logger.error(f"❌ Ошибка получения активных заявок: {e}")
            return []

    def _quotation_to_float(self, quotation) -> float:
        """Преобразование Quotation в float"""
        if quotation is None:
            return 0.0
        return quotation.units + quotation.nano / 1_000_000_000

    def get_instrument_info(self, figi: str) -> dict:
        """Получить информацию об инструменте"""
        try:
            with self._client() as client:
                response = client.instruments.get_instrument_by(figi=figi)
                instrument = response.instrument

                info = {
                    "figi": instrument.figi,
                    "ticker": instrument.ticker,
                    "name": instrument.name,
                    "currency": instrument.currency,
                    "lot": instrument.lot,
                    "min_price_increment": self._quotation_to_float(instrument.min_price_increment),
                    "trading_status": instrument.trading_status,
                    "api_trade_available_flag": instrument.api_trade_available_flag,
                    "for_iis_flag": instrument.for_iis_flag,
                    "sector": instrument.sector,
                    "country_of_risk": instrument.country_of_risk,
                    "country_of_risk_name": instrument.country_of_risk_name,
                }

                return info

        except Exception as e:
            logger.error(f"❌ Ошибка получения информации об инструменте {figi}: {e}")
            raise

    def health_check(self) -> bool:
        """Проверка работоспособности API"""
        try:
            accounts = self.get_accounts()
            if accounts:
                logger.info("✅ API работает корректно")
                return True
            else:
                logger.warning("⚠️ API работает, но счета не найдены")
                return False
        except Exception as e:
            logger.error(f"❌ Проверка API провалена: {e}")
            return False


# Функции для тестирования
def test_connection():
    """Тестирование подключения к API"""
    print("🧪 Тестирование подключения к T-Bank API...")

    try:
        broker = TBankBroker()
        if broker.health_check():
            print("✅ Подключение успешно!")

            # Получить информацию о счетах
            accounts = broker.get_accounts()
            for i, account in enumerate(accounts):
                print(f"  📊 Счёт {i+1}: {account.id} ({account.name})")

            return True
        else:
            print("❌ Подключение не работает")
            return False

    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
        return False


if __name__ == "__main__":
    test_connection()
