"""
Leap Trading System - RiskManager Tests
Comprehensive tests for the risk management module.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch

# Import components to test
from core.risk_manager import (
    RiskManager, DynamicRiskManager,
    RiskLimits, PositionSizing, RiskState
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def default_risk_manager():
    """Create RiskManager with default settings."""
    return RiskManager(initial_balance=10000.0)


@pytest.fixture
def custom_risk_manager():
    """Create RiskManager with custom settings."""
    limits = RiskLimits(
        max_position_size_pct=0.03,
        max_total_exposure_pct=0.15,
        max_daily_loss_pct=0.03,
        max_drawdown_pct=0.10,
        max_consecutive_losses=3,
        max_open_positions=3,
        min_risk_reward=2.0
    )
    sizing = PositionSizing(
        method='percent',
        percent_risk=0.01,
        max_leverage=20
    )
    return RiskManager(
        initial_balance=50000.0,
        limits=limits,
        sizing=sizing
    )


@pytest.fixture
def dynamic_risk_manager():
    """Create DynamicRiskManager."""
    return DynamicRiskManager(initial_balance=10000.0)


# ============================================================================
# RiskLimits Tests
# ============================================================================

class TestRiskLimits:
    """Tests for RiskLimits dataclass."""

    def test_default_values(self):
        """Test default RiskLimits values."""
        limits = RiskLimits()

        assert limits.max_position_size_pct == 0.02
        assert limits.max_total_exposure_pct == 0.10
        assert limits.max_daily_loss_pct == 0.05
        assert limits.max_weekly_loss_pct == 0.10
        assert limits.max_drawdown_pct == 0.15
        assert limits.max_consecutive_losses == 5
        assert limits.max_open_positions == 5
        assert limits.max_correlation == 0.7
        assert limits.min_risk_reward == 1.5

    def test_custom_values(self):
        """Test RiskLimits with custom values."""
        limits = RiskLimits(
            max_position_size_pct=0.05,
            max_drawdown_pct=0.20,
            max_consecutive_losses=10
        )

        assert limits.max_position_size_pct == 0.05
        assert limits.max_drawdown_pct == 0.20
        assert limits.max_consecutive_losses == 10


# ============================================================================
# PositionSizing Tests
# ============================================================================

class TestPositionSizing:
    """Tests for PositionSizing dataclass."""

    def test_default_values(self):
        """Test default PositionSizing values."""
        sizing = PositionSizing()

        assert sizing.method == 'kelly'
        assert sizing.fixed_size == 0.01
        assert sizing.percent_risk == 0.02
        assert sizing.kelly_fraction == 0.5
        assert sizing.max_leverage == 10

    def test_custom_values(self):
        """Test PositionSizing with custom values."""
        sizing = PositionSizing(
            method='percent',
            percent_risk=0.03,
            max_leverage=50
        )

        assert sizing.method == 'percent'
        assert sizing.percent_risk == 0.03
        assert sizing.max_leverage == 50


# ============================================================================
# RiskManager Initialization Tests
# ============================================================================

class TestRiskManagerInitialization:
    """Tests for RiskManager initialization."""

    def test_initialization_default(self, default_risk_manager):
        """Test initialization with default values."""
        rm = default_risk_manager

        assert rm.initial_balance == 10000.0
        assert rm.current_balance == 10000.0
        assert rm.peak_equity == 10000.0
        assert rm.state.is_trading_allowed is True
        assert rm.wins == 0
        assert rm.losses == 0

    def test_initialization_custom_balance(self):
        """Test initialization with custom balance."""
        rm = RiskManager(initial_balance=50000.0)

        assert rm.initial_balance == 50000.0
        assert rm.current_balance == 50000.0

    def test_initialization_custom_limits(self, custom_risk_manager):
        """Test initialization with custom limits."""
        rm = custom_risk_manager

        assert rm.limits.max_position_size_pct == 0.03
        assert rm.limits.max_drawdown_pct == 0.10

    def test_initialization_custom_sizing(self, custom_risk_manager):
        """Test initialization with custom sizing."""
        rm = custom_risk_manager

        assert rm.sizing.method == 'percent'
        assert rm.sizing.percent_risk == 0.01


# ============================================================================
# Balance Update Tests
# ============================================================================

class TestBalanceUpdates:
    """Tests for balance update functionality."""

    def test_update_balance_profit(self, default_risk_manager):
        """Test balance update with profit."""
        rm = default_risk_manager
        rm.update_balance(10500.0)

        assert rm.current_balance == 10500.0
        assert rm.peak_equity == 10500.0
        assert rm.state.daily_pnl == 500.0

    def test_update_balance_loss(self, default_risk_manager):
        """Test balance update with loss."""
        rm = default_risk_manager
        rm.update_balance(9500.0)

        assert rm.current_balance == 9500.0
        assert rm.peak_equity == 10000.0  # Peak unchanged
        assert rm.state.daily_pnl == -500.0

    def test_update_balance_drawdown_calculation(self, default_risk_manager):
        """Test drawdown is calculated correctly."""
        rm = default_risk_manager

        # First profit to establish new peak
        rm.update_balance(11000.0)
        assert rm.peak_equity == 11000.0

        # Then loss
        rm.update_balance(9900.0)
        expected_drawdown = (11000.0 - 9900.0) / 11000.0
        assert rm.state.current_drawdown == pytest.approx(expected_drawdown)

    def test_update_balance_triggers_halt_on_max_drawdown(self, default_risk_manager):
        """Test trading halt on max drawdown."""
        rm = default_risk_manager

        # Create drawdown beyond limit (default 15%)
        rm.update_balance(8400.0)  # 16% drawdown

        assert rm.state.is_trading_allowed is False
        assert 'drawdown' in rm.state.halt_reason.lower()

    def test_update_balance_triggers_halt_on_daily_loss(self, default_risk_manager):
        """Test trading halt on daily loss limit."""
        rm = default_risk_manager

        # Lose more than 5% in a day
        rm.update_balance(9400.0)  # 6% daily loss

        assert rm.state.is_trading_allowed is False
        assert 'daily' in rm.state.halt_reason.lower()


# ============================================================================
# Trade Recording Tests
# ============================================================================

class TestTradeRecording:
    """Tests for trade recording functionality."""

    def test_record_winning_trade(self, default_risk_manager):
        """Test recording a winning trade."""
        rm = default_risk_manager
        rm.record_trade(100.0, True)

        assert rm.wins == 1
        assert rm.losses == 0
        assert rm.avg_win == 100.0
        assert rm.state.consecutive_losses == 0

    def test_record_losing_trade(self, default_risk_manager):
        """Test recording a losing trade."""
        rm = default_risk_manager
        rm.record_trade(-50.0, False)

        assert rm.wins == 0
        assert rm.losses == 1
        assert rm.avg_loss == 50.0
        assert rm.state.consecutive_losses == 1

    def test_record_trade_infers_win_from_pnl(self, default_risk_manager):
        """Test that win/loss is inferred from pnl when not provided."""
        rm = default_risk_manager

        # Positive pnl should be treated as win
        rm.record_trade(100.0)
        assert rm.wins == 1

        # Negative pnl should be treated as loss
        rm.record_trade(-50.0)
        assert rm.losses == 1

    def test_record_breakeven_trade(self, default_risk_manager):
        """Test recording a breakeven trade."""
        rm = default_risk_manager
        rm.record_trade(0.0)

        # Breakeven trades don't count as wins or losses
        assert rm.wins == 0
        assert rm.losses == 0

    def test_consecutive_losses_count(self, default_risk_manager):
        """Test consecutive losses counter."""
        rm = default_risk_manager

        rm.record_trade(-50.0)
        assert rm.state.consecutive_losses == 1

        rm.record_trade(-30.0)
        assert rm.state.consecutive_losses == 2

        rm.record_trade(20.0)  # Win resets counter
        assert rm.state.consecutive_losses == 0

    def test_consecutive_losses_triggers_halt(self, default_risk_manager):
        """Test trading halt on max consecutive losses."""
        rm = default_risk_manager

        # Default max is 5
        for i in range(5):
            rm.record_trade(-10.0)

        assert rm.state.is_trading_allowed is False
        assert 'consecutive' in rm.state.halt_reason.lower()

    def test_average_win_calculation(self, default_risk_manager):
        """Test average win calculation over multiple trades."""
        rm = default_risk_manager

        rm.record_trade(100.0, True)
        rm.record_trade(200.0, True)
        rm.record_trade(150.0, True)

        expected_avg = (100.0 + 200.0 + 150.0) / 3
        assert rm.avg_win == pytest.approx(expected_avg)

    def test_average_loss_calculation(self, default_risk_manager):
        """Test average loss calculation over multiple trades."""
        rm = default_risk_manager

        rm.record_trade(-50.0, False)
        rm.record_trade(-100.0, False)
        rm.record_trade(-75.0, False)

        expected_avg = (50.0 + 100.0 + 75.0) / 3
        assert rm.avg_loss == pytest.approx(expected_avg)


# ============================================================================
# Position Sizing Tests
# ============================================================================

class TestPositionSizing:
    """Tests for position sizing calculations."""

    def test_fixed_position_sizing(self):
        """Test fixed position sizing method."""
        sizing = PositionSizing(method='fixed', fixed_size=0.05)
        rm = RiskManager(initial_balance=10000.0, sizing=sizing)

        size = rm.calculate_position_size(
            entry_price=1.1000,
            stop_loss_price=1.0950
        )

        assert size == pytest.approx(0.05, rel=0.1)

    def test_percent_position_sizing(self):
        """Test percent risk position sizing."""
        sizing = PositionSizing(method='percent', percent_risk=0.02)
        rm = RiskManager(initial_balance=10000.0, sizing=sizing)

        entry_price = 1.1000
        stop_loss_price = 1.0900  # 100 pips risk

        size = rm.calculate_position_size(entry_price, stop_loss_price)

        # Risk = 2% of 10000 = 200
        # Risk per unit = 0.01
        # Size = 200 / 0.01 = 20000 (but capped by limits)
        assert size > 0

    def test_volatility_position_sizing(self):
        """Test volatility-based position sizing."""
        sizing = PositionSizing(method='volatility', percent_risk=0.02)
        rm = RiskManager(initial_balance=10000.0, sizing=sizing)

        size = rm.calculate_position_size(
            entry_price=1.1000,
            stop_loss_price=1.0950,
            volatility=0.02  # 2% volatility
        )

        assert size > 0

    def test_kelly_position_sizing_insufficient_data(self):
        """Test Kelly sizing with insufficient trade history."""
        sizing = PositionSizing(method='kelly')
        rm = RiskManager(initial_balance=10000.0, sizing=sizing)

        # Less than 20 trades - should use conservative sizing
        for i in range(10):
            rm.record_trade(50.0 if i % 2 == 0 else -30.0)

        size = rm.calculate_position_size(
            entry_price=1.1000,
            stop_loss_price=1.0950
        )

        assert size > 0

    def test_kelly_position_sizing_sufficient_data(self):
        """Test Kelly sizing with sufficient trade history."""
        sizing = PositionSizing(method='kelly')
        rm = RiskManager(initial_balance=10000.0, sizing=sizing)

        # Create 20+ trades with 60% win rate
        for i in range(25):
            if i % 5 < 3:  # 60% wins
                rm.record_trade(100.0, True)
            else:
                rm.record_trade(-80.0, False)

        size = rm.calculate_position_size(
            entry_price=1.1000,
            stop_loss_price=1.0950
        )

        assert size > 0

    def test_position_size_zero_when_trading_halted(self, default_risk_manager):
        """Test position size is zero when trading is halted."""
        rm = default_risk_manager
        rm.state.is_trading_allowed = False

        size = rm.calculate_position_size(
            entry_price=1.1000,
            stop_loss_price=1.0950
        )

        assert size == 0.0

    def test_position_size_respects_max_limits(self, default_risk_manager):
        """Test position size respects maximum limits."""
        rm = default_risk_manager

        # Try to get very large position
        size = rm.calculate_position_size(
            entry_price=1.1000,
            stop_loss_price=1.0999  # Very tight stop
        )

        # Should be limited by max position size percentage
        max_allowed = rm.current_balance * rm.limits.max_position_size_pct / 1.1000
        assert size <= max_allowed


# ============================================================================
# Stop Loss/Take Profit Tests
# ============================================================================

class TestStopLossTakeProfit:
    """Tests for stop loss and take profit calculations."""

    def test_calculate_stop_loss_long_atr(self, default_risk_manager):
        """Test stop loss calculation for long position using ATR."""
        rm = default_risk_manager

        sl = rm.calculate_stop_loss(
            entry_price=1.1000,
            direction='long',
            atr=0.0050,  # 50 pips ATR
            method='atr'
        )

        # 2 ATR stop = 100 pips below entry
        expected = 1.1000 - (0.0050 * 2)
        assert sl == pytest.approx(expected)

    def test_calculate_stop_loss_short_atr(self, default_risk_manager):
        """Test stop loss calculation for short position using ATR."""
        rm = default_risk_manager

        sl = rm.calculate_stop_loss(
            entry_price=1.1000,
            direction='short',
            atr=0.0050,
            method='atr'
        )

        expected = 1.1000 + (0.0050 * 2)
        assert sl == pytest.approx(expected)

    def test_calculate_stop_loss_percent(self, default_risk_manager):
        """Test stop loss calculation using percent method."""
        rm = default_risk_manager

        sl = rm.calculate_stop_loss(
            entry_price=1.1000,
            direction='long',
            method='percent'
        )

        # Default 2% stop
        expected = 1.1000 - (1.1000 * 0.02)
        assert sl == pytest.approx(expected)

    def test_calculate_take_profit_long(self, default_risk_manager):
        """Test take profit calculation for long position."""
        rm = default_risk_manager

        tp = rm.calculate_take_profit(
            entry_price=1.1000,
            stop_loss_price=1.0950,
            direction='long',
            risk_reward=2.0
        )

        # Risk = 50 pips, reward = 100 pips
        expected = 1.1000 + 0.0100
        assert tp == pytest.approx(expected)

    def test_calculate_take_profit_short(self, default_risk_manager):
        """Test take profit calculation for short position."""
        rm = default_risk_manager

        tp = rm.calculate_take_profit(
            entry_price=1.1000,
            stop_loss_price=1.1050,
            direction='short',
            risk_reward=2.0
        )

        # Risk = 50 pips, reward = 100 pips
        expected = 1.1000 - 0.0100
        assert tp == pytest.approx(expected)

    def test_calculate_take_profit_default_risk_reward(self, default_risk_manager):
        """Test take profit uses default risk/reward ratio."""
        rm = default_risk_manager

        tp = rm.calculate_take_profit(
            entry_price=1.1000,
            stop_loss_price=1.0950,
            direction='long'
        )

        # Default min_risk_reward = 1.5
        expected = 1.1000 + (0.0050 * 1.5)
        assert tp == pytest.approx(expected)


# ============================================================================
# Trade Approval Tests
# ============================================================================

class TestTradeApproval:
    """Tests for trade approval logic."""

    def test_should_take_trade_approved(self, default_risk_manager):
        """Test trade approval for valid trade."""
        rm = default_risk_manager

        approved, reason = rm.should_take_trade(
            entry_price=1.1000,
            stop_loss_price=1.0950,
            take_profit_price=1.1100,  # 2:1 R/R
            direction='long'
        )

        assert approved is True
        assert 'approved' in reason.lower()

    def test_should_take_trade_rejected_low_risk_reward(self, default_risk_manager):
        """Test trade rejection for low risk/reward."""
        rm = default_risk_manager

        approved, reason = rm.should_take_trade(
            entry_price=1.1000,
            stop_loss_price=1.0950,
            take_profit_price=1.1025,  # 0.5:1 R/R
            direction='long'
        )

        assert approved is False
        assert 'risk/reward' in reason.lower()

    def test_should_take_trade_rejected_trading_halted(self, default_risk_manager):
        """Test trade rejection when trading is halted."""
        rm = default_risk_manager
        rm.state.is_trading_allowed = False
        rm.state.halt_reason = "Test halt"

        approved, reason = rm.should_take_trade(
            entry_price=1.1000,
            stop_loss_price=1.0950,
            take_profit_price=1.1100,
            direction='long'
        )

        assert approved is False
        assert 'halted' in reason.lower()

    def test_should_take_trade_rejected_max_positions(self, default_risk_manager):
        """Test trade rejection when max positions reached."""
        rm = default_risk_manager
        rm.state.open_positions = 5  # At limit

        approved, reason = rm.should_take_trade(
            entry_price=1.1000,
            stop_loss_price=1.0950,
            take_profit_price=1.1100,
            direction='long'
        )

        assert approved is False
        assert 'position' in reason.lower()

    def test_should_take_trade_rejected_invalid_direction(self, default_risk_manager):
        """Test trade rejection for invalid direction."""
        rm = default_risk_manager

        approved, reason = rm.should_take_trade(
            entry_price=1.1000,
            stop_loss_price=1.0950,
            take_profit_price=1.1100,
            direction='invalid'
        )

        assert approved is False
        assert 'direction' in reason.lower()

    def test_should_take_trade_rejected_invalid_long_stops(self, default_risk_manager):
        """Test trade rejection for invalid long trade stop placement."""
        rm = default_risk_manager

        # Stop loss above entry for long
        approved, reason = rm.should_take_trade(
            entry_price=1.1000,
            stop_loss_price=1.1050,  # Invalid - above entry
            take_profit_price=1.1100,
            direction='long'
        )

        assert approved is False
        assert 'stop loss' in reason.lower()

    def test_should_take_trade_rejected_invalid_short_stops(self, default_risk_manager):
        """Test trade rejection for invalid short trade stop placement."""
        rm = default_risk_manager

        # Stop loss below entry for short
        approved, reason = rm.should_take_trade(
            entry_price=1.1000,
            stop_loss_price=1.0950,  # Invalid - below entry for short
            take_profit_price=1.0900,
            direction='short'
        )

        assert approved is False
        assert 'stop loss' in reason.lower()

    def test_should_take_trade_rejected_zero_risk(self, default_risk_manager):
        """Test trade rejection for zero risk."""
        rm = default_risk_manager

        approved, reason = rm.should_take_trade(
            entry_price=1.1000,
            stop_loss_price=1.1000,  # Same as entry
            take_profit_price=1.1100,
            direction='long'
        )

        assert approved is False


# ============================================================================
# Position Tracking Tests
# ============================================================================

class TestPositionTracking:
    """Tests for position open/close tracking."""

    def test_on_position_opened(self, default_risk_manager):
        """Test position opened tracking."""
        rm = default_risk_manager

        rm.on_position_opened(notional=10000.0)

        assert rm.state.open_positions == 1
        assert rm.state.total_exposure == 10000.0

    def test_on_position_closed(self, default_risk_manager):
        """Test position closed tracking."""
        rm = default_risk_manager
        rm.on_position_opened(notional=10000.0)
        rm.on_position_opened(notional=5000.0)

        rm.on_position_closed(notional=10000.0)

        assert rm.state.open_positions == 1
        assert rm.state.total_exposure == 5000.0

    def test_on_position_closed_prevents_negative(self, default_risk_manager):
        """Test that position closed doesn't go negative."""
        rm = default_risk_manager

        rm.on_position_closed(notional=10000.0)

        assert rm.state.open_positions == 0
        assert rm.state.total_exposure == 0.0


# ============================================================================
# Risk Report Tests
# ============================================================================

class TestRiskReport:
    """Tests for risk report generation."""

    def test_get_risk_report_initial(self, default_risk_manager):
        """Test risk report with initial state."""
        rm = default_risk_manager
        report = rm.get_risk_report()

        assert report['current_balance'] == 10000.0
        assert report['initial_balance'] == 10000.0
        assert report['peak_equity'] == 10000.0
        assert report['current_drawdown'] == 0.0
        assert report['is_trading_allowed'] is True
        assert report['total_trades'] == 0
        assert report['win_rate'] == 0.0

    def test_get_risk_report_after_trades(self, default_risk_manager):
        """Test risk report after some trades."""
        rm = default_risk_manager

        rm.record_trade(100.0, True)
        rm.record_trade(50.0, True)
        rm.record_trade(-30.0, False)
        rm.update_balance(10120.0)

        report = rm.get_risk_report()

        assert report['total_trades'] == 3
        assert report['win_rate'] == pytest.approx(2/3)
        assert report['daily_pnl'] == 120.0


# ============================================================================
# Reset Tests
# ============================================================================

class TestReset:
    """Tests for risk manager reset."""

    def test_reset_restores_initial_state(self, default_risk_manager):
        """Test that reset restores initial state."""
        rm = default_risk_manager

        # Make some changes
        rm.update_balance(9000.0)
        rm.record_trade(-100.0, False)
        rm.record_trade(-100.0, False)
        rm.on_position_opened(5000.0)

        # Reset
        rm.reset()

        assert rm.current_balance == 10000.0
        assert rm.peak_equity == 10000.0
        assert rm.wins == 0
        assert rm.losses == 0
        assert rm.state.open_positions == 0
        assert rm.state.is_trading_allowed is True
        assert len(rm.equity_history) == 0


# ============================================================================
# DynamicRiskManager Tests
# ============================================================================

class TestDynamicRiskManager:
    """Tests for DynamicRiskManager."""

    def test_initialization(self, dynamic_risk_manager):
        """Test DynamicRiskManager initialization."""
        drm = dynamic_risk_manager

        assert drm.regime == 'normal'
        assert len(drm.volatility_history) == 0

    def test_update_market_conditions_normal(self, dynamic_risk_manager):
        """Test market conditions update - normal regime."""
        drm = dynamic_risk_manager

        # Add some baseline volatility
        for _ in range(10):
            drm.update_market_conditions(0.01)

        # Add normal volatility
        drm.update_market_conditions(0.01)

        assert drm.regime == 'normal'

    def test_update_market_conditions_high_volatility(self, dynamic_risk_manager):
        """Test market conditions update - high volatility regime."""
        drm = dynamic_risk_manager

        # Add baseline volatility
        for _ in range(10):
            drm.update_market_conditions(0.01)

        # Add high volatility (> 1.5x average)
        drm.update_market_conditions(0.02)

        assert drm.regime == 'high_volatility'

    def test_update_market_conditions_low_volatility(self, dynamic_risk_manager):
        """Test market conditions update - low volatility regime."""
        drm = dynamic_risk_manager

        # Add baseline volatility
        for _ in range(10):
            drm.update_market_conditions(0.02)

        # Add low volatility (< 0.5x average)
        drm.update_market_conditions(0.005)

        assert drm.regime == 'low_volatility'

    def test_position_size_reduced_in_high_volatility(self, dynamic_risk_manager):
        """Test position size reduction in high volatility."""
        drm = dynamic_risk_manager

        # Get baseline position size
        baseline_size = drm.calculate_position_size(1.1000, 1.0950)

        # Set high volatility regime
        for _ in range(10):
            drm.update_market_conditions(0.01)
        drm.update_market_conditions(0.02)

        high_vol_size = drm.calculate_position_size(1.1000, 1.0950)

        # Size should be reduced
        assert high_vol_size < baseline_size

    def test_position_size_increased_in_low_volatility(self, dynamic_risk_manager):
        """Test position size increase in low volatility."""
        drm = dynamic_risk_manager

        # Set normal regime first
        for _ in range(10):
            drm.update_market_conditions(0.02)

        normal_size = drm.calculate_position_size(1.1000, 1.0950)

        # Set low volatility regime
        drm.update_market_conditions(0.005)

        low_vol_size = drm.calculate_position_size(1.1000, 1.0950)

        # Size should be increased (or equal if at limits)
        assert low_vol_size >= normal_size * 0.99  # Allow small tolerance

    def test_position_size_reduced_after_losses(self, dynamic_risk_manager):
        """Test position size reduction after consecutive losses."""
        drm = dynamic_risk_manager

        # Get baseline
        baseline_size = drm.calculate_position_size(1.1000, 1.0950)

        # Record 3 losses
        drm.record_trade(-50.0, False)
        drm.record_trade(-50.0, False)
        drm.record_trade(-50.0, False)

        reduced_size = drm.calculate_position_size(1.1000, 1.0950)

        # Size should be reduced
        assert reduced_size < baseline_size


# ============================================================================
# Edge Cases Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_balance(self):
        """Test with zero balance."""
        rm = RiskManager(initial_balance=0.0)

        size = rm.calculate_position_size(1.1000, 1.0950)
        assert size == 0.0

    def test_very_small_balance(self):
        """Test with very small balance."""
        rm = RiskManager(initial_balance=1.0)

        size = rm.calculate_position_size(1.1000, 1.0950)
        assert size >= 0.0

    def test_very_large_balance(self):
        """Test with very large balance."""
        rm = RiskManager(initial_balance=1000000000.0)

        size = rm.calculate_position_size(1.1000, 1.0950)
        assert size > 0

    def test_very_tight_stop_loss(self, default_risk_manager):
        """Test with very tight stop loss."""
        rm = default_risk_manager

        size = rm.calculate_position_size(
            entry_price=1.1000,
            stop_loss_price=1.0999  # 1 pip stop
        )

        # Should still be limited
        assert size > 0
        max_allowed = rm.current_balance * rm.limits.max_position_size_pct / 1.1000
        assert size <= max_allowed

    def test_very_wide_stop_loss(self, default_risk_manager):
        """Test with very wide stop loss."""
        rm = default_risk_manager

        size = rm.calculate_position_size(
            entry_price=1.1000,
            stop_loss_price=0.9000  # 2000 pip stop
        )

        assert size > 0

    def test_negative_pnl_trade(self, default_risk_manager):
        """Test recording large negative pnl."""
        rm = default_risk_manager

        rm.record_trade(-5000.0, False)

        assert rm.losses == 1
        assert rm.avg_loss == 5000.0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
