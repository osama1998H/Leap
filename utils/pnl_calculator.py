"""
Leap Trading System - PnL Calculator Utilities
Centralized profit and loss calculations to eliminate code duplication.
"""

from typing import Literal


def calculate_pnl(
    entry_price: float,
    exit_price: float,
    size: float,
    direction: Literal['long', 'short'],
    contract_size: float = 1.0
) -> float:
    """
    Calculate realized profit/loss for a closed position.

    This is the single source of truth for PnL calculations across the system.
    Use this instead of inline calculations to ensure consistency.

    Args:
        entry_price: Price at which position was opened
        exit_price: Price at which position was closed
        size: Position size (volume/lots)
        direction: Trade direction ('long' or 'short')
        contract_size: Contract size multiplier (default 1.0 for unit-based,
                      100000 for forex standard lots)

    Returns:
        Realized PnL (positive for profit, negative for loss)

    Examples:
        >>> calculate_pnl(1.1000, 1.1050, 1.0, 'long')
        0.005
        >>> calculate_pnl(1.1000, 1.1050, 1.0, 'short')
        -0.005
        >>> calculate_pnl(1.1000, 1.1050, 0.1, 'long', contract_size=100000)
        500.0
    """
    if direction == 'long':
        return (exit_price - entry_price) * size * contract_size
    else:
        return (entry_price - exit_price) * size * contract_size


def calculate_unrealized_pnl(
    entry_price: float,
    current_price: float,
    size: float,
    direction: Literal['long', 'short'],
    contract_size: float = 1.0
) -> float:
    """
    Calculate unrealized profit/loss for an open position.

    This is an alias for calculate_pnl with semantic clarity for open positions.

    Args:
        entry_price: Price at which position was opened
        current_price: Current market price
        size: Position size (volume/lots)
        direction: Trade direction ('long' or 'short')
        contract_size: Contract size multiplier (default 1.0)

    Returns:
        Unrealized PnL at current price
    """
    return calculate_pnl(entry_price, current_price, size, direction, contract_size)


def calculate_pnl_percent(
    entry_price: float,
    exit_price: float,
    direction: Literal['long', 'short']
) -> float:
    """
    Calculate PnL as a percentage of entry price.

    Args:
        entry_price: Price at which position was opened
        exit_price: Price at which position was closed
        direction: Trade direction ('long' or 'short')

    Returns:
        PnL as decimal (0.01 = 1%)
    """
    if direction == 'long':
        return (exit_price - entry_price) / entry_price
    else:
        return (entry_price - exit_price) / entry_price


__all__ = ['calculate_pnl', 'calculate_unrealized_pnl', 'calculate_pnl_percent']
