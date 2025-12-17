"""
CLI command implementations.

This module provides command handlers for all CLI commands and a registry
for command dispatch.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    from config import SystemConfig
    from cli.system import LeapTradingSystem

from .train import execute_train
from .backtest import execute_backtest
from .walkforward import execute_walkforward
from .evaluate import execute_evaluate
from .autotrade import execute_autotrade

# Command registry mapping command names to handler functions
COMMANDS = {
    'train': execute_train,
    'backtest': execute_backtest,
    'walkforward': execute_walkforward,
    'evaluate': execute_evaluate,
    'autotrade': execute_autotrade,
}


def execute_command(
    command: str,
    system: 'LeapTradingSystem',
    args: 'argparse.Namespace',
    config: 'SystemConfig',
    resolved: dict
) -> None:
    """
    Execute a CLI command.

    Args:
        command: Command name (train, backtest, etc.)
        system: LeapTradingSystem instance
        args: Parsed command-line arguments
        config: System configuration
        resolved: Resolved configuration values from CLI/config
    """
    handler = COMMANDS.get(command)
    if handler is None:
        raise ValueError(f"Unknown command: {command}")
    handler(system, args, config, resolved)


__all__ = [
    'execute_command',
    'execute_train',
    'execute_backtest',
    'execute_walkforward',
    'execute_evaluate',
    'execute_autotrade',
    'COMMANDS',
]
