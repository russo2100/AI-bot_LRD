# src/ai_integration/external_ai.py
from __future__ import annotations

import requests
import json
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import os
import time

logger = logging.getLogger(__name__)

class AIAnalysisService:
    """
    Сервис для интеграции с внешними ИИ API для анализа рынка
    Поддерживает OpenAI, Anthropic Claude, Google Gemini
    """

    def __init__(self):
        self.services = {
            'openai': OpenAIAnalyzer(),
            'claude': ClaudeAnalyzer(),
            'gemini': GeminiAnalyzer(),
            'local_llm': LocalLLMAnalyzer()  # Для локальных моделей
        }

    def analyze_market_sentiment(self, news_texts: List[str], 
                                service: str = 'openai') -> Dict[str, Any]:
        """
        Анализ настроений рынка на основе новостей

        Args:
            news_texts: Список новостных текстов
            service: Используемый ИИ сервис

        Returns:
            Результат анализа настроений
        """
        if service not in self.services:
            raise ValueError(f"Неподдерживаемый сервис: {service}")

        return self.services[service].analyze_sentiment(news_texts)

    def generate_trading_insights(self, market_data: Dict[str, Any],
                                 service: str = 'openai') -> Dict[str, Any]:
        """
        Генерация торговых инсайтов на основе рыночных данных
        """
        if service not in self.services:
            raise ValueError(f"Неподдерживаемый сервис: {service}")

        return self.services[service].generate_insights(market_data)

    def explain_prediction(self, prediction_data: Dict[str, Any],
                          service: str = 'openai') -> str:
        """
        Объяснение прогноза в понятной форме
        """
        if service not in self.services:
            raise ValueError(f"Неподдерживаемый сервис: {service}")

        return self.services[service].explain_prediction(prediction_data)


class OpenAIAnalyzer:
    """Интеграция с OpenAI GPT"""

    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.base_url = "https://api.openai.com/v1"
        self.model = "gpt-4-turbo-preview"

    def analyze_sentiment(self, news_texts: List[str]) -> Dict[str, Any]:
        """Анализ настроений через OpenAI"""
        if not self.api_key:
            logger.error("❌ OPENAI_API_KEY не найден")
            return {'error': 'API key отсутствует'}

        try:
            # Объединяем новости для анализа
            combined_text = "\n".join(news_texts[:10])  # Ограничиваем количество

            prompt = f"""
            Проанализируй настроения в следующих финансовых новостях и дай оценку влияния на рынок:

            НОВОСТИ:
            {combined_text}

            Оцени по шкале от -100 до +100:
            - Общее настроение (-100 = очень негативное, +100 = очень позитивное)
            - Влияние на акции (-100 = сильно негативное, +100 = сильно позитивное)
            - Влияние на валютный рынок
            - Краткосрочное влияние (1-7 дней)
            - Долгосрочное влияние (1-3 месяца)

            Ответь в JSON формате:
            {{
                "overall_sentiment": число,
                "stock_impact": число,
                "currency_impact": число,
                "short_term_impact": число,
                "long_term_impact": число,
                "key_themes": ["тема1", "тема2"],
                "summary": "краткое резюме"
            }}
            """

            response = self._make_request(prompt)

            # Парсим JSON ответ
            try:
                result = json.loads(response)
                result['timestamp'] = datetime.now().isoformat()
                result['source'] = 'openai'
                return result
            except json.JSONDecodeError:
                logger.warning("⚠️ Не удалось распарсить JSON ответ")
                return {
                    'overall_sentiment': 0,
                    'summary': response[:500],
                    'source': 'openai',
                    'timestamp': datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"❌ Ошибка анализа настроений OpenAI: {e}")
            return {'error': str(e)}

    def generate_insights(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация торговых инсайтов"""
        try:
            prompt = f"""
            Проанализируй следующие рыночные данные и дай торговые рекомендации:

            ДАННЫЕ:
            {json.dumps(market_data, indent=2, ensure_ascii=False)}

            Дай рекомендации в JSON формате:
            {{
                "recommendation": "BUY/SELL/HOLD",
                "confidence": 0.0-1.0,
                "reasoning": "обоснование",
                "risk_level": "LOW/MEDIUM/HIGH",
                "time_horizon": "SHORT/MEDIUM/LONG",
                "key_factors": ["фактор1", "фактор2"],
                "price_target": число или null,
                "stop_loss": число или null
            }}
            """

            response = self._make_request(prompt)

            try:
                result = json.loads(response)
                result['timestamp'] = datetime.now().isoformat()
                result['source'] = 'openai'
                return result
            except json.JSONDecodeError:
                return {
                    'recommendation': 'HOLD',
                    'reasoning': response[:500],
                    'source': 'openai',
                    'timestamp': datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"❌ Ошибка генерации инсайтов: {e}")
            return {'error': str(e)}

    def explain_prediction(self, prediction_data: Dict[str, Any]) -> str:
        """Объяснение прогноза модели"""
        try:
            prompt = f"""
            Объясни простыми словами результат прогноза торговой модели:

            ДАННЫЕ ПРОГНОЗА:
            {json.dumps(prediction_data, indent=2, ensure_ascii=False)}

            Дай понятное объяснение:
            1. Что означает этот прогноз
            2. Насколько ему можно доверять
            3. Какие факторы повлияли на результат
            4. Что это означает для торговли

            Ответ должен быть понятен даже начинающему трейдеру.
            """

            response = self._make_request(prompt)
            return response

        except Exception as e:
            logger.error(f"❌ Ошибка объяснения прогноза: {e}")
            return f"Ошибка объяснения: {e}"

    def _make_request(self, prompt: str) -> str:
        """Выполнение запроса к OpenAI API"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': 'Ты эксперт по финансовым рынкам и техническому анализу. Отвечай точно и профессионально.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'temperature': 0.3,
            'max_tokens': 2000
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")


class ClaudeAnalyzer:
    """Интеграция с Anthropic Claude"""

    def __init__(self):
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        self.base_url = "https://api.anthropic.com/v1"
        self.model = "claude-3-sonnet-20240229"

    def analyze_sentiment(self, news_texts: List[str]) -> Dict[str, Any]:
        """Анализ настроений через Claude"""
        if not self.api_key:
            logger.warning("⚠️ ANTHROPIC_API_KEY не найден, используется заглушка")
            return self._mock_sentiment_response()

        try:
            combined_text = "\n".join(news_texts[:10])

            prompt = f"""
            Проанализируй финансовые новости и оцени рыночные настроения:

            {combined_text}

            Дай оценки от -100 до +100 и верни результат в JSON:
            {{
                "overall_sentiment": число,
                "stock_impact": число, 
                "currency_impact": число,
                "volatility_expectation": число,
                "key_themes": ["тема1", "тема2"],
                "summary": "краткое резюме"
            }}
            """

            response = self._make_request(prompt)

            try:
                result = json.loads(response)
                result['timestamp'] = datetime.now().isoformat()
                result['source'] = 'claude'
                return result
            except json.JSONDecodeError:
                return self._mock_sentiment_response()

        except Exception as e:
            logger.error(f"❌ Ошибка Claude API: {e}")
            return self._mock_sentiment_response()

    def generate_insights(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация торговых инсайтов через Claude"""
        # Заглушка для Claude (реальная реализация аналогична OpenAI)
        return {
            'recommendation': 'HOLD',
            'confidence': 0.6,
            'reasoning': 'Анализ через Claude временно недоступен',
            'source': 'claude',
            'timestamp': datetime.now().isoformat()
        }

    def explain_prediction(self, prediction_data: Dict[str, Any]) -> str:
        """Объяснение прогноза через Claude"""
        return "Объяснение через Claude API временно недоступно"

    def _make_request(self, prompt: str) -> str:
        """Выполнение запроса к Claude API"""
        # Упрощенная реализация (требует доработки под реальный API)
        raise NotImplementedError("Claude API интеграция требует доработки")

    def _mock_sentiment_response(self) -> Dict[str, Any]:
        """Заглушка для ответа анализа настроений"""
        return {
            'overall_sentiment': 0,
            'stock_impact': 0,
            'currency_impact': 0,
            'volatility_expectation': 0,
            'key_themes': ['mock_analysis'],
            'summary': 'Заглушка анализа настроений',
            'source': 'claude_mock',
            'timestamp': datetime.now().isoformat()
        }


class GeminiAnalyzer:
    """Интеграция с Google Gemini"""

    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        self.base_url = "https://generativelanguage.googleapis.com/v1"
        self.model = "gemini-pro"

    def analyze_sentiment(self, news_texts: List[str]) -> Dict[str, Any]:
        """Заглушка для Gemini"""
        return {
            'overall_sentiment': 0,
            'summary': 'Gemini интеграция в разработке',
            'source': 'gemini_mock',
            'timestamp': datetime.now().isoformat()
        }

    def generate_insights(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Заглушка для Gemini"""
        return {
            'recommendation': 'HOLD',
            'reasoning': 'Gemini интеграция в разработке',
            'source': 'gemini_mock',
            'timestamp': datetime.now().isoformat()
        }

    def explain_prediction(self, prediction_data: Dict[str, Any]) -> str:
        """Заглушка для Gemini"""
        return "Gemini интеграция в разработке"


class LocalLLMAnalyzer:
    """Интеграция с локальными LLM моделями"""

    def __init__(self):
        self.ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
        self.model = os.getenv('LOCAL_LLM_MODEL', 'llama2')

    def analyze_sentiment(self, news_texts: List[str]) -> Dict[str, Any]:
        """Анализ настроений через локальную модель"""
        try:
            # Проверяем доступность Ollama
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise Exception("Ollama недоступен")

            combined_text = "\n".join(news_texts[:5])  # Меньше текста для локальных моделей

            prompt = f"""
            Проанализируй финансовые новости и оцени настроения рынка от -100 до +100:

            {combined_text}

            Ответь кратко в формате:
            Настроение: [число]
            Резюме: [краткое описание]
            """

            response = self._make_ollama_request(prompt)

            # Парсим простой ответ
            sentiment = 0
            summary = response

            # Пытаемся извлечь числовое значение
            import re
            sentiment_match = re.search(r'Настроение:\s*([+-]?\d+)', response)
            if sentiment_match:
                sentiment = int(sentiment_match.group(1))

            return {
                'overall_sentiment': sentiment,
                'summary': summary[:200],
                'source': 'local_llm',
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.warning(f"⚠️ Локальная LLM недоступна: {e}")
            return {
                'overall_sentiment': 0,
                'summary': 'Локальная модель недоступна',
                'source': 'local_llm_mock',
                'timestamp': datetime.now().isoformat()
            }

    def generate_insights(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Торговые инсайты через локальную модель"""
        return {
            'recommendation': 'HOLD',
            'reasoning': 'Локальная LLM анализ в разработке',
            'source': 'local_llm',
            'timestamp': datetime.now().isoformat()
        }

    def explain_prediction(self, prediction_data: Dict[str, Any]) -> str:
        """Объяснение через локальную модель"""
        return "Объяснение через локальную LLM в разработке"

    def _make_ollama_request(self, prompt: str) -> str:
        """Запрос к Ollama API"""
        data = {
            'model': self.model,
            'prompt': prompt,
            'stream': False
        }

        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json=data,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            return result.get('response', '')
        else:
            raise Exception(f"Ollama API error: {response.status_code}")


# Модуль для сбора новостей
class NewsCollector:
    """Сборщик финансовых новостей для анализа настроений"""

    def __init__(self):
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')

    def collect_financial_news(self, query: str = 'financial markets', 
                              language: str = 'ru',
                              max_articles: int = 20) -> List[Dict[str, str]]:
        """
        Сбор финансовых новостей из различных источников

        Returns:
            Список словарей с новостями: [{'title': '', 'content': '', 'source': '', 'timestamp': ''}]
        """
        news = []

        # NewsAPI
        if self.news_api_key:
            try:
                news.extend(self._collect_from_newsapi(query, language, max_articles // 2))
            except Exception as e:
                logger.warning(f"⚠️ Ошибка NewsAPI: {e}")

        # Alpha Vantage News
        if self.alpha_vantage_key:
            try:
                news.extend(self._collect_from_alpha_vantage(max_articles // 2))
            except Exception as e:
                logger.warning(f"⚠️ Ошибка Alpha Vantage: {e}")

        # Заглушка с примерами новостей
        if not news:
            news = self._get_mock_news()

        return news[:max_articles]

    def _collect_from_newsapi(self, query: str, language: str, max_articles: int) -> List[Dict[str, str]]:
        """Сбор новостей через NewsAPI"""
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'language': language,
            'sortBy': 'publishedAt',
            'pageSize': max_articles,
            'apiKey': self.news_api_key
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        news = []

        for article in data.get('articles', []):
            news.append({
                'title': article.get('title', ''),
                'content': article.get('description', '') + ' ' + article.get('content', ''),
                'source': article.get('source', {}).get('name', 'NewsAPI'),
                'timestamp': article.get('publishedAt', ''),
                'url': article.get('url', '')
            })

        return news

    def _collect_from_alpha_vantage(self, max_articles: int) -> List[Dict[str, str]]:
        """Сбор новостей через Alpha Vantage"""
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'NEWS_SENTIMENT',
            'apikey': self.alpha_vantage_key,
            'limit': max_articles
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        news = []

        for article in data.get('feed', []):
            news.append({
                'title': article.get('title', ''),
                'content': article.get('summary', ''),
                'source': article.get('source', 'Alpha Vantage'),
                'timestamp': article.get('time_published', ''),
                'sentiment': article.get('overall_sentiment_label', 'Neutral')
            })

        return news

    def _get_mock_news(self) -> List[Dict[str, str]]:
        """Заглушка с примерами новостей"""
        return [
            {
                'title': 'Рынки акций показывают смешанную динамику',
                'content': 'Российский фондовый рынок завершил торги в смешанном настроении на фоне колебаний нефтяных котировок.',
                'source': 'Mock News',
                'timestamp': datetime.now().isoformat()
            },
            {
                'title': 'ЦБ РФ сохранил ключевую ставку',
                'content': 'Центральный банк России сохранил ключевую ставку на текущем уровне, что соответствовало ожиданиям аналитиков.',
                'source': 'Mock News',
                'timestamp': datetime.now().isoformat()
            },
            {
                'title': 'Волатильность валютного рынка растет',
                'content': 'Курс доллара к рублю показывает повышенную волатильность на фоне геополитической напряженности.',
                'source': 'Mock News',
                'timestamp': datetime.now().isoformat()
            }
        ]


if __name__ == "__main__":
    # Тестирование интеграции с ИИ
    print("🧪 Тестирование интеграции с внешними ИИ сервисами")

    # Инициализация сервисов
    ai_service = AIAnalysisService()
    news_collector = NewsCollector()

    # Сбор новостей
    print("📰 Сбор финансовых новостей...")
    news = news_collector.collect_financial_news(max_articles=5)
    print(f"✅ Собрано новостей: {len(news)}")

    # Анализ настроений
    if news:
        news_texts = [article['title'] + ' ' + article['content'] for article in news]

        print("🧠 Анализ настроений через ИИ...")

        # Пробуем разные сервисы
        for service in ['openai', 'claude', 'local_llm']:
            try:
                sentiment = ai_service.analyze_market_sentiment(news_texts, service=service)
                print(f"📊 {service.upper()}: Настроение={sentiment.get('overall_sentiment', 0)}")
                print(f"   Резюме: {sentiment.get('summary', 'Нет резюме')[:100]}")
            except Exception as e:
                print(f"❌ Ошибка {service}: {e}")

    # Тестирование генерации инсайтов
    print("\n💡 Генерация торговых инсайтов...")
    mock_data = {
        'symbol': 'SBER',
        'current_price': 250.5,
        'change_percent': 2.3,
        'volume': 1500000,
        'rsi': 65,
        'macd': 0.8
    }

    try:
        insights = ai_service.generate_trading_insights(mock_data, service='openai')
        print(f"🎯 Рекомендация: {insights.get('recommendation', 'HOLD')}")
        print(f"   Обоснование: {insights.get('reasoning', 'Нет обоснования')[:100]}")
    except Exception as e:
        print(f"❌ Ошибка генерации инсайтов: {e}")

    print("✅ Тестирование завершено!")
