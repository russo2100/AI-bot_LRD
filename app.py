#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main application entry point for AI Trading Bot
Unified interface combining CLI and web functionality
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from config.settings import settings
    from main import AITradingBot
    from web_app import app as web_app
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed")
    sys.exit(1)


class TradingBotApp:
    """Main trading bot application"""

    def __init__(self, config_override: dict = None):
        self.config = config_override or {}
        self.bot: Optional[AITradingBot] = None
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging based on settings"""
        log_config = settings.logging

        logging.basicConfig(
            level=getattr(logging, log_config.level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_config.file_path, encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def initialize_bot(self):
        """Initialize the trading bot"""
        try:
            self.bot = AITradingBot()
            self.logger.info("Trading bot initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize bot: {e}")
            return False

    def start_trading(self, symbols: List[str] = None, demo: bool = True):
        """Start trading session"""
        if not self.bot:
            if not self.initialize_bot():
                return False

        symbols = symbols or settings.trading.symbols
        self.logger.info(f"Starting trading with symbols: {symbols}")
        self.logger.info(f"Demo mode: {demo}")

        try:
            self.bot.start_trading(symbols)
            return True
        except KeyboardInterrupt:
            self.logger.info("Trading stopped by user")
            return True
        except Exception as e:
            self.logger.error(f"Trading error: {e}")
            return False

    def run_backtest(self, symbols: List[str] = None, start_date: str = None,
                     end_date: str = None, strategy: str = None):
        """Run backtesting"""
        if not self.bot:
            if not self.initialize_bot():
                return False

        symbols = symbols or settings.trading.symbols
        start_date = start_date or settings.backtest.start_date
        end_date = end_date or settings.backtest.end_date
        strategy = strategy or settings.backtest.default_strategy

        self.logger.info(f"Running backtest: {start_date} - {end_date}")
        self.logger.info(f"Strategy: {strategy}, Symbols: {symbols}")

        try:
            self.bot.run_backtest(symbols, start_date, end_date, strategy)
            self.logger.info("Backtest completed successfully")
            return True
        except Exception as e:
            self.logger.error(f"Backtest error: {e}")
            return False

    def train_models(self, symbols: List[str] = None, force: bool = False):
        """Train or retrain models"""
        if not self.bot:
            if not self.initialize_bot():
                return False

        symbols = symbols or settings.trading.symbols

        self.logger.info(f"Training models for symbols: {symbols}")
        if force:
            self.logger.info("Forced retraining enabled")

        try:
            self.bot.load_or_train_models(symbols, retrain=force)
            self.logger.info("Model training completed")
            return True
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            return False

    def analyze_market(self, symbols: List[str] = None):
        """Perform market analysis"""
        if not self.bot:
            if not self.initialize_bot():
                return None

        symbols = symbols or settings.trading.symbols

        self.logger.info(f"Analyzing market for symbols: {symbols}")

        try:
            results = self.bot.analyze_market(symbols)
            return results
        except Exception as e:
            self.logger.error(f"Analysis error: {e}")
            return None

    def get_status(self):
        """Get bot status"""
        if not self.bot:
            return {'error': 'Bot not initialized', 'status': 'error'}

        try:
            return self.bot.get_status()
        except Exception as e:
            self.logger.error(f"Status error: {e}")
            return {'error': str(e), 'status': 'error'}

    def start_web_interface(self):
        """Start web interface"""
        web_config = settings.web
        self.logger.info(f"Starting web interface on {web_config.host}:{web_config.port}")

        try:
            web_app.run(
                host=web_config.host,
                port=web_config.port,
                debug=web_config.debug
            )
            return True
        except Exception as e:
            self.logger.error(f"Web interface error: {e}")
            return False


def create_argument_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description='AI Trading Bot - Unified Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python app.py start --symbols AAPL MSFT
  python app.py backtest --start-date 2023-01-01 --end-date 2023-06-01
  python app.py train --force
  python app.py analyze --symbols SBER GAZP
  python app.py status
  python app.py web
        '''
    )

    parser.add_argument('--version', action='version', version='1.0.0')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Trading command
    trading_parser = subparsers.add_parser('start', help='Start trading')
    trading_parser.add_argument('--symbols', '-s', nargs='+',
                               default=settings.trading.symbols,
                               help='Trading symbols')
    trading_parser.add_argument('--demo', action='store_true',
                               default=True, help='Run in demo mode')

    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtesting')
    backtest_parser.add_argument('--symbols', '-s', nargs='+',
                                default=settings.trading.symbols,
                                help='Trading symbols')
    backtest_parser.add_argument('--start-date', default=settings.backtest.start_date,
                                help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end-date', default=settings.backtest.end_date,
                                help='End date (YYYY-MM-DD)')
    backtest_parser.add_argument('--strategy', default=settings.backtest.default_strategy,
                                help='Trading strategy')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--symbols', '-s', nargs='+',
                             default=settings.trading.symbols,
                             help='Trading symbols')
    train_parser.add_argument('--force', '-f', action='store_true',
                             help='Force retraining')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze market')
    analyze_parser.add_argument('--symbols', '-s', nargs='+',
                               default=settings.trading.symbols,
                               help='Trading symbols')

    # Status command
    subparsers.add_parser('status', help='Show bot status')

    # Web command
    subparsers.add_parser('web', help='Start web interface')

    # Setup command
    subparsers.add_parser('setup', help='Initial project setup')

    return parser


def main():
    """Main entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize application
    app = TradingBotApp()

    # Execute command
    try:
        if args.command == 'start':
            success = app.start_trading(args.symbols, args.demo)

        elif args.command == 'backtest':
            success = app.run_backtest(args.symbols, args.start_date,
                                     args.end_date, args.strategy)

        elif args.command == 'train':
            success = app.train_models(args.symbols, args.force)

        elif args.command == 'analyze':
            results = app.analyze_market(args.symbols)
            if results:
                print("Market Analysis Results:")
                for symbol, analysis in results.items():
                    print(f"\n{symbol}:")
                    print(f"  Price: {analysis['current_price']:.2f}")
                    print(f"  Signal: {analysis['signal']}")
                    print(f"  RSI: {analysis['technical_indicators']['rsi']:.1f}")
                success = True
            else:
                success = False

        elif args.command == 'status':
            status = app.get_status()
            if 'error' not in status:
                print("Bot Status:")
                print(f"  Running: {status.get('is_running', False)}")
                print(f"  Trading: {status.get('is_trading', False)}")
                print(f"  Models loaded: {status.get('models_loaded', [])}")
                print(f"  Last analysis: {status.get('last_analysis', 'Never')}")
                success = True
            else:
                print(f"Error: {status['error']}")
                success = False

        elif args.command == 'web':
            success = app.start_web_interface()

        elif args.command == 'setup':
            print("🛠️ Setting up project...")
            settings.create_directories()
            print("✅ Setup completed!")
            success = True

        if success is not None:
            sys.exit(0 if success else 1)

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()