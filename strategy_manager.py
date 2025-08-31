#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Manager - отвечающий за создание и переключение стратегий
"""

import logging
from typing import Dict, Any, List, Optional, Type
from abc import ABC, abstractmethod

from config.settings import settings

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Base strategy class"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.is_active = True

    @abstractmethod
    def generate_signal(self, data, current_idx: int):
        """Generate trading signal"""
        pass


class StrategyManager:
    """
    Strategy Manager - отвечает только за создание и переключение стратегий
    """

    def __init__(self):
        self._strategies: Dict[str, Dict[str, Any]] = {}
        self._active_strategy: Optional[str] = None
        self._fallback_strategy = "ma_crossover"

    def register_strategy(self, name: str, strategy_class: Type[BaseStrategy],
                         config: Optional[Dict[str, Any]] = None,
                         enabled: bool = True) -> None:
        """
        Register a trading strategy

        Args:
            name: Strategy name
            strategy_class: Strategy class
            config: Strategy configuration
            enabled: Whether strategy is enabled
        """
        self._strategies[name] = {
            'class': strategy_class,
            'config': config or {},
            'enabled': enabled,
            'instance': None
        }
        logger.info(f"Strategy '{name}' registered")

    def unregister_strategy(self, name: str) -> None:
        """Unregister a strategy"""
        if name in self._strategies:
            del self._strategies[name]
            if self._active_strategy == name:
                self._active_strategy = None
            logger.info(f"Strategy '{name}' unregistered")

    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        """
        Get strategy instance

        Args:
            name: Strategy name

        Returns:
            Strategy instance or None if not found/disabled
        """
        if name not in self._strategies:
            logger.error(f"Strategy '{name}' not registered")
            return None

        strategy_info = self._strategies[name]
        if not strategy_info['enabled']:
            logger.warning(f"Strategy '{name}' is disabled")
            return None

        # Create instance if not exists
        if strategy_info['instance'] is None:
            try:
                strategy_info['instance'] = strategy_info['class'](
                    config=strategy_info['config']
                )
                logger.info(f"Strategy '{name}' instance created")
            except Exception as e:
                logger.error(f"Failed to create strategy '{name}': {e}")
                strategy_info['enabled'] = False
                return None

        return strategy_info['instance']

    def set_active_strategy(self, name: str) -> bool:
        """
        Set active strategy

        Args:
            name: Strategy name

        Returns:
            True if successful, False otherwise
        """
        strategy = self.get_strategy(name)
        if strategy:
            self._active_strategy = name
            logger.info(f"Active strategy set to '{name}'")
            return True
        else:
            logger.error(f"Cannot set active strategy to '{name}'")
            return False

    def get_active_strategy(self) -> Optional[BaseStrategy]:
        """Get active strategy instance"""
        if self._active_strategy:
            return self.get_strategy(self._active_strategy)
        return None

    def get_active_strategy_name(self) -> Optional[str]:
        """Get active strategy name"""
        return self._active_strategy

    def list_strategies(self) -> List[str]:
        """List all registered strategies"""
        return list(self._strategies.keys())

    def list_enabled_strategies(self) -> List[str]:
        """List enabled strategies"""
        return [name for name, info in self._strategies.items() if info['enabled']]

    def enable_strategy(self, name: str) -> bool:
        """Enable a strategy"""
        if name in self._strategies:
            self._strategies[name]['enabled'] = True
            logger.info(f"Strategy '{name}' enabled")
            return True
        return False

    def disable_strategy(self, name: str) -> bool:
        """Disable a strategy"""
        if name in self._strategies:
            self._strategies[name]['enabled'] = False
            # Clear instance to force recreation when re-enabled
            self._strategies[name]['instance'] = None
            if self._active_strategy == name:
                self.switch_to_fallback()
            logger.info(f"Strategy '{name}' disabled")
            return True
        return False

    def switch_to_fallback(self) -> bool:
        """Switch to fallback strategy"""
        if self._fallback_strategy and self._fallback_strategy != self._active_strategy:
            return self.set_active_strategy(self._fallback_strategy)
        return False

    def get_strategy_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Get strategy configuration"""
        if name in self._strategies:
            return self._strategies[name]['config']
        return None

    def update_strategy_config(self, name: str, config: Dict[str, Any]) -> bool:
        """
        Update strategy configuration

        Args:
            name: Strategy name
            config: New configuration

        Returns:
            True if successful, False otherwise
        """
        if name in self._strategies:
            self._strategies[name]['config'].update(config)
            # Reset instance to apply new config
            self._strategies[name]['instance'] = None
            logger.info(f"Strategy '{name}' configuration updated")
            return True
        return False

    def validate_strategy(self, name: str) -> bool:
        """
        Validate strategy

        Args:
            name: Strategy name

        Returns:
            True if strategy is valid, False otherwise
        """
        strategy = self.get_strategy(name)
        if not strategy:
            return False

        # Basic validation - check required methods
        required_methods = ['generate_signal']
        for method in required_methods:
            if not hasattr(strategy, method):
                logger.error(f"Strategy '{name}' missing required method: {method}")
                return False

        logger.info(f"Strategy '{name}' validation passed")
        return True

    def get_strategy_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed strategy information

        Args:
            name: Strategy name

        Returns:
            Strategy information dict or None
        """
        if name not in self._strategies:
            return None

        info = self._strategies[name].copy()
        info.pop('instance', None)  # Remove instance from public info
        info['is_active'] = (name == self._active_strategy)
        return info

    def reload_strategies(self) -> None:
        """Reload all strategies (clear instances)"""
        for name in self._strategies:
            self._strategies[name]['instance'] = None
        logger.info("All strategies reloaded")

    def get_status(self) -> Dict[str, Any]:
        """Get manager status"""
        return {
            'total_strategies': len(self._strategies),
            'enabled_strategies': len(self.list_enabled_strategies()),
            'active_strategy': self._active_strategy,
            'fallback_strategy': self._fallback_strategy,
            'strategies': {
                name: {
                    'enabled': info['enabled'],
                    'is_active': (name == self._active_strategy)
                }
                for name, info in self._strategies.items()
            }
        }


# Global strategy manager instance
strategy_manager = StrategyManager()


def get_strategy_manager() -> StrategyManager:
    """Get global strategy manager instance"""
    return strategy_manager


def create_strategy(name: str, strategy_class: Type[BaseStrategy] = None,
                   config: Dict[str, Any] = None) -> Optional[BaseStrategy]:
    """
    Helper function to create a strategy

    Args:
        name: Strategy name
        strategy_class: Strategy class (optional, if already registered)
        config: Strategy configuration

    Returns:
        Strategy instance
    """
    if strategy_class:
        strategy_manager.register_strategy(name, strategy_class, config)

    return strategy_manager.get_strategy(name)


def get_available_strategies() -> List[str]:
    """Get list of available strategies"""
    return strategy_manager.list_enabled_strategies()


def get_active_strategy() -> Optional[BaseStrategy]:
    """Get currently active strategy"""
    return strategy_manager.get_active_strategy()


# Initialize with default fallback
if not strategy_manager.list_strategies():
    logger.info("No strategies registered, using basic fallback logic")