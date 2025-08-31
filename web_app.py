#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Веб-интерфейс для AI-торгового бота T-Bank
"""

import sys
import os
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import json
import plotly
import plotly.graph_objs as go
from plotly.utils import PlotlyJSONEncoder

# Добавляем корневую папку в Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импорты из основного бота
try:
    from main import AITradingBot
except ImportError as e:
    print(f"Warning: Could not import AITradingBot: {e}")
    AITradingBot = None

app = Flask(__name__, template_folder='templates')
app.json_encoder = PlotlyJSONEncoder

# Глобальная переменная для бота
bot = None
analysis_results = {}

def init_bot():
    """Инициализация бота"""
    global bot
    if bot is None and AITradingBot:
        try:
            bot = AITradingBot()
            print("Bot initialized successfully")
        except Exception as e:
            print(f"Error initializing bot: {e}")
            bot = None

@app.route('/')
def index():
    """Главная страница"""
    global bot, analysis_results
    init_bot()

    # Получаем статус бота
    status = bot.get_status() if bot else {
        'is_running': False,
        'models_loaded': [],
        'components': {}
    }

    return render_template('index.html',
                         status=status,
                         analysis_results=analysis_results,
                         symbols=['SBER', 'GAZP', 'LKOH', 'MOEX'] if not analysis_results else list(analysis_results.keys()))

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Выполнить анализ рынка"""
    global bot, analysis_results
    init_bot()

    if not bot:
        return jsonify({'error': 'Bot not initialized'}), 500

    if request.method == 'POST':
        # Получаем символы из формы
        symbols_input = request.form.get('symbols', 'SBER,MOEX')
        symbols = [s.strip() for s in symbols_input.split(',') if s.strip()]

        try:
            # Выполняем анализ
            results = bot.analyze_market(symbols)
            analysis_results = results

            # Обновляем переменную для сквозного использование в графиках
            return render_template('results.html',
                                 analysis_results=results,
                                 timestamp=datetime.now().isoformat(),
                                 symbols=symbols)

        except Exception as e:
            return json.dumps({'error': str(e)}), 500

    # GET запрос - показываем форму
    return render_template('analyze.html')

@app.route('/charts/<symbol>')
def chart(symbol):
    """Показать график для символа"""
    global bot
    init_bot()

    try:
        # Демпинг данные для графика
        # Получаем исторические данные
        if bot and hasattr(bot, 'data_collector'):
            end_date = datetime.now()
            start_date = end_date.replace(year=end_date.year - 1)  # Последний год

            data = bot.data_collector.get_historical_data(
                symbol,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )

            # Создаем график с помощью plotly
            fig = go.Figure()

            # Добавляем основные линии
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['close'],
                mode='lines',
                name='Цена закрытия',
                line=dict(color='blue', width=2)
            ))

            # Добавляем технические индикаторы если есть
            if hasattr(bot, 'technical_indicators'):
                try:
                    tech_indicators = bot.technical_indicators
                    data_with_indicators = tech_indicators.add_all_indicators(data)

                    # RSI
                    if 'rsi' in data_with_indicators.columns:
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data_with_indicators['rsi'],
                            mode='lines',
                            name='RSI',
                            yaxis='y2',
                            line=dict(color='orange', width=1)
                        ))

                    # MA 20 и MA 50
                    if 'sma_20' in data_with_indicators.columns:
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data_with_indicators['sma_20'],
                            mode='lines',
                            name='SMA 20',
                            line=dict(color='red', width=1, dash='dash')
                        ))

                    if 'sma_50' in data_with_indicators.columns:
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data_with_indicators['sma_50'],
                            mode='lines',
                            name='SMA 50',
                            line=dict(color='green', width=1, dash='dash')
                        ))

                except Exception as e:
                    print(f"Error adding technical indicators: {e}")

            # Настройки графика
            fig.update_layout(
                title=f"График {symbol}",
                xaxis_title="Дата",
                yaxis_title="Цена",
                yaxis2=dict(
                    title="RSI",
                    overlaying='y',
                    side='right',
                    range=[0, 100]
                ),
                height=600,
                template="plotly_white"
            )

            # Конвертируем в JSON для передачи в шаблон
            graph_json = fig.to_json()

        else:
            graph_json = '{}'

        return render_template('chart.html',
                             symbol=symbol,
                             graph_json=graph_json)

    except Exception as e:
        return f"Ошибка загрузки графика для {symbol}: {str(e)}"

@app.route('/api/results')
def get_analysis_results():
    """API для получения результатов анализа"""
    global analysis_results
    return jsonify({
        'results': analysis_results,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/backtest', methods=['GET', 'POST'])
def backtest():
    """Страница бэктестинга"""
    if request.method == 'POST':
        # Получаем параметры из формы
        symbols = request.form.get('backtest_symbols', 'SBER').split(',')
        start_date = request.form.get('start_date', '2024-01-01')
        end_date = request.form.get('end_date', '2024-06-01')
        strategy = request.form.get('backtest_strategy', 'ma_crossover')

        try:
            # Запускаем бэктест
            if bot:
                bot.run_backtest(symbols, start_date, end_date, strategy)
                return render_template('backtest_results.html',
                                     symbols=symbols,
                                     start_date=start_date,
                                     end_date=end_date,
                                     strategy=strategy)
            else:
                return "Bot not available for backtesting"

        except Exception as e:
            return f"Backtest error: {str(e)}"

    # GET запрос - показываем форму
    return render_template('backtest.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)