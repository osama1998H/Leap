"""
Leap Trading System - Position Sizing Utilities
Centralized position sizing calculations to eliminate code duplication.
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)


def calculate_risk_based_size(
    balance: float,
    risk_per_trade: float,
    stop_loss_pips: float,
    pip_value: float,
    entry_price: Optional[float] = None
) -> float:
    """
    Calculate position size based on risk amount and stop loss distance.

    This is the standard risk-based position sizing formula used by professional traders.
    Risk Amount = Balance * Risk Per Trade
    Position Size = Risk Amount / (Stop Loss Distance * Pip Value)

    Args:
        balance: Current account balance
        risk_per_trade: Fraction of balance to risk (e.g., 0.02 for 2%)
        stop_loss_pips: Stop loss distance in pips
        pip_value: Value per pip per unit (e.g., 0.0001 for EURUSD)
        entry_price: Entry price (optional, used for additional normalization)

    Returns:
        Calculated position size in units

    Examples:
        >>> calculate_risk_based_size(10000, 0.02, 50, 0.0001)
        4000.0
    """
    if stop_loss_pips <= 0 or pip_value <= 0:
        logger.warning(f"Invalid stop_loss_pips ({stop_loss_pips}) or pip_value ({pip_value})")
        return 0.0

    risk_amount = balance * risk_per_trade

    if entry_price is not None and entry_price > 0:
        # Normalized formula used by backtester
        size = risk_amount / (stop_loss_pips * pip_value * entry_price)
    else:
        # Simplified formula for when entry_price normalization isn't needed
        size = risk_amount / (stop_loss_pips * pip_value)

    return max(0.0, size)


def calculate_percentage_size(
    balance: float,
    position_fraction: float,
    entry_price: float
) -> float:
    """
    Calculate position size as a percentage of balance.

    This is a simpler sizing method that allocates a fixed fraction of balance
    to each position, regardless of stop loss distance.

    Args:
        balance: Current account balance
        position_fraction: Fraction of balance to allocate (e.g., 0.1 for 10%)
        entry_price: Entry price for the position

    Returns:
        Calculated position size in units

    Examples:
        >>> calculate_percentage_size(10000, 0.1, 1.1000)
        909.09...
    """
    if entry_price <= 0:
        logger.warning(f"Invalid entry_price: {entry_price}")
        return 0.0

    position_value = balance * position_fraction
    return position_value / entry_price


def apply_position_limits(
    size: float,
    balance: float,
    leverage: int,
    entry_price: float,
    max_position_size: Optional[float] = None,
    min_position_size: float = 0.0
) -> float:
    """
    Apply leverage and size limits to a calculated position size.

    This function applies common constraints:
    1. Leverage limit: max_size = (balance * leverage) / entry_price
    2. Maximum position size cap (if specified)
    3. Minimum position size floor (if specified)

    Args:
        size: Calculated position size before limits
        balance: Current account balance
        leverage: Account leverage (e.g., 100 for 100:1)
        entry_price: Entry price for the position
        max_position_size: Optional maximum position size cap
        min_position_size: Optional minimum position size floor (default 0)

    Returns:
        Position size after applying limits

    Examples:
        >>> apply_position_limits(1000000, 10000, 100, 1.1, max_position_size=500000)
        500000.0
    """
    if entry_price <= 0:
        return 0.0

    # Apply leverage limit
    max_leveraged_size = (balance * leverage) / entry_price
    size = min(size, max_leveraged_size)

    # Apply max position size cap
    if max_position_size is not None:
        size = min(size, max_position_size)

    # Apply minimum size floor
    size = max(size, min_position_size)

    return size


def calculate_position_size_with_limits(
    balance: float,
    risk_per_trade: float,
    stop_loss_pips: float,
    pip_value: float,
    entry_price: float,
    leverage: int = 100,
    max_position_size: Optional[float] = None
) -> float:
    """
    Calculate position size with risk-based sizing and apply limits.

    This is a convenience function that combines calculate_risk_based_size
    and apply_position_limits for the common use case.

    Args:
        balance: Current account balance
        risk_per_trade: Fraction of balance to risk (e.g., 0.02 for 2%)
        stop_loss_pips: Stop loss distance in pips
        pip_value: Value per pip per unit
        entry_price: Entry price for the position
        leverage: Account leverage (default 100)
        max_position_size: Optional maximum position size cap

    Returns:
        Final position size after risk calculation and limits
    """
    size = calculate_risk_based_size(
        balance, risk_per_trade, stop_loss_pips, pip_value, entry_price
    )
    return apply_position_limits(
        size, balance, leverage, entry_price, max_position_size
    )


__all__ = [
    'calculate_risk_based_size',
    'calculate_percentage_size',
    'apply_position_limits',
    'calculate_position_size_with_limits',
]
