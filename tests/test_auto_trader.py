"""
Unit tests for the Auto-Trader components.
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

# Import components to test
from core.order_manager import OrderManager, TradingSignal, SignalType, OrderExecution
from core.position_sync import PositionSynchronizer, PositionTracker, PositionEvent, PositionChange
from core.mt5_broker import (
    MT5BrokerGateway, OrderType, AccountInfo, SymbolInfo, TickInfo,
    MT5Position, OrderResult
)


# ============================================================================
# Mock Classes for Testing
# ============================================================================

@dataclass
class MockAccountInfo:
    """Mock MT5 account info."""
    login: int = 12345
    balance: float = 10000.0
    equity: float = 10000.0
    margin: float = 0.0
    margin_free: float = 10000.0
    margin_level: float = 0.0
    leverage: int = 100
    currency: str = "USD"
    server: str = "Demo"
    company: str = "Test Broker"
    trade_allowed: bool = True


@dataclass
class MockSymbolInfo:
    """Mock MT5 symbol info."""
    name: str = "EURUSD"
    point: float = 0.00001
    digits: int = 5
    spread: int = 15
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    trade_contract_size: float = 100000.0
    trade_tick_value: float = 1.0
    trade_tick_size: float = 0.00001
    trade_stops_level: int = 0
    trade_freeze_level: int = 0
    filling_mode: int = 1


@dataclass
class MockTickInfo:
    """Mock MT5 tick info."""
    bid: float = 1.10000
    ask: float = 1.10015
    last: float = 1.10000
    volume: float = 1000.0
    time: int = 1700000000


@dataclass
class MockPosition:
    """Mock MT5 position."""
    ticket: int = 1001
    symbol: str = "EURUSD"
    type: int = 0  # BUY
    volume: float = 0.1
    price_open: float = 1.09950
    price_current: float = 1.10000
    sl: float = 1.09450
    tp: float = 1.10450
    profit: float = 50.0
    swap: float = 0.0
    magic: int = 234567
    comment: str = "Test"
    time: int = 1700000000


@dataclass
class MockOrderResult:
    """Mock MT5 order result."""
    retcode: int = 10009  # Success
    order: int = 1002
    volume: float = 0.1
    price: float = 1.10000
    bid: float = 1.10000
    ask: float = 1.10015
    comment: str = "Request executed"
    request_id: int = 1


# ============================================================================
# MT5 Broker Gateway Tests
# ============================================================================

class TestMT5BrokerGateway:
    """Tests for MT5BrokerGateway."""

    def test_account_info_from_mt5(self):
        """Test AccountInfo creation from MT5 data."""
        mock_info = MockAccountInfo()
        account = AccountInfo.from_mt5(mock_info)

        assert account.login == 12345
        assert account.balance == 10000.0
        assert account.equity == 10000.0
        assert account.leverage == 100
        assert account.trade_allowed is True

    def test_symbol_info_from_mt5(self):
        """Test SymbolInfo creation from MT5 data."""
        mock_info = MockSymbolInfo()
        symbol = SymbolInfo.from_mt5(mock_info)

        assert symbol.name == "EURUSD"
        assert symbol.point == 0.00001
        assert symbol.digits == 5
        assert symbol.volume_min == 0.01
        assert symbol.volume_max == 100.0

    def test_tick_info_from_mt5(self):
        """Test TickInfo creation from MT5 data."""
        mock_tick = MockTickInfo()
        tick = TickInfo.from_mt5(mock_tick, "EURUSD")

        assert tick.symbol == "EURUSD"
        assert tick.bid == 1.10000
        assert tick.ask == 1.10015
        assert tick.spread == pytest.approx(0.00015, rel=1e-5)

    def test_mt5_position_from_mt5(self):
        """Test MT5Position creation from MT5 data."""
        mock_pos = MockPosition()
        position = MT5Position.from_mt5(mock_pos)

        assert position.ticket == 1001
        assert position.symbol == "EURUSD"
        assert position.is_long is True
        assert position.is_short is False
        assert position.profit == 50.0

    def test_order_result_from_mt5(self):
        """Test OrderResult creation from MT5 data."""
        mock_result = MockOrderResult()
        result = OrderResult.from_mt5(mock_result)

        assert result.success is True
        assert result.ticket == 1002
        assert result.price == 1.10000

    def test_order_result_error(self):
        """Test OrderResult error creation."""
        result = OrderResult.error("Test error", retcode=10019)

        assert result.success is False
        assert result.comment == "Test error"
        assert result.retcode == 10019

    def test_gateway_initialization(self):
        """Test gateway initialization."""
        gateway = MT5BrokerGateway(
            magic_number=123456,
            max_retries=5
        )

        assert gateway.magic_number == 123456
        assert gateway.max_retries == 5
        assert gateway._is_connected is False


# ============================================================================
# Order Manager Tests
# ============================================================================

class TestOrderManager:
    """Tests for OrderManager."""

    @pytest.fixture
    def mock_broker(self):
        """Create mock broker."""
        broker = Mock(spec=MT5BrokerGateway)
        broker.is_connected = True
        broker.magic_number = 234567

        # Setup mock returns
        broker.get_account_info.return_value = AccountInfo.from_mt5(MockAccountInfo())
        broker.get_symbol_info.return_value = SymbolInfo.from_mt5(MockSymbolInfo())
        broker.get_current_tick.return_value = TickInfo.from_mt5(MockTickInfo(), "EURUSD")
        broker.get_positions.return_value = []

        return broker

    def test_trading_signal_creation(self):
        """Test TradingSignal creation."""
        signal = TradingSignal(
            signal_type=SignalType.BUY,
            symbol="EURUSD",
            confidence=0.8,
            stop_loss_pips=50.0,
            take_profit_pips=100.0
        )

        assert signal.is_entry is True
        assert signal.is_exit is False
        assert signal.is_hold is False

    def test_hold_signal(self):
        """Test HOLD signal properties."""
        signal = TradingSignal(
            signal_type=SignalType.HOLD,
            symbol="EURUSD"
        )

        assert signal.is_entry is False
        assert signal.is_exit is False
        assert signal.is_hold is True

    def test_close_signal(self):
        """Test CLOSE signal properties."""
        signal = TradingSignal(
            signal_type=SignalType.CLOSE,
            symbol="EURUSD"
        )

        assert signal.is_entry is False
        assert signal.is_exit is True
        assert signal.is_hold is False

    def test_order_manager_initialization(self, mock_broker):
        """Test OrderManager initialization."""
        manager = OrderManager(
            broker=mock_broker,
            default_sl_pips=50.0,
            default_tp_pips=100.0
        )

        assert manager.default_sl_pips == 50.0
        assert manager.default_tp_pips == 100.0
        assert manager.stats['total_signals'] == 0

    def test_execute_hold_signal(self, mock_broker):
        """Test executing HOLD signal."""
        manager = OrderManager(broker=mock_broker)

        signal = TradingSignal(
            signal_type=SignalType.HOLD,
            symbol="EURUSD"
        )

        execution = manager.execute_signal(signal)

        assert execution.executed is True
        assert "Hold signal" in execution.error_message
        assert manager.stats['total_signals'] == 1

    def test_execute_buy_signal_success(self, mock_broker):
        """Test executing BUY signal successfully."""
        mock_broker.send_market_order.return_value = OrderResult.from_mt5(MockOrderResult())

        manager = OrderManager(broker=mock_broker)

        signal = TradingSignal(
            signal_type=SignalType.BUY,
            symbol="EURUSD",
            confidence=0.8
        )

        execution = manager.execute_signal(signal)

        assert execution.executed is True
        assert execution.ticket == 1002
        mock_broker.send_market_order.assert_called_once()

    def test_reject_low_confidence(self, mock_broker):
        """Test rejection of low confidence signals."""
        manager = OrderManager(broker=mock_broker, min_confidence=0.7)

        signal = TradingSignal(
            signal_type=SignalType.BUY,
            symbol="EURUSD",
            confidence=0.5  # Below threshold
        )

        execution = manager.execute_signal(signal)

        assert execution.executed is False
        assert "Confidence" in execution.error_message
        assert manager.stats['rejected'] == 1

    def test_statistics_tracking(self, mock_broker):
        """Test that statistics are properly tracked."""
        mock_broker.send_market_order.return_value = OrderResult.from_mt5(MockOrderResult())

        manager = OrderManager(broker=mock_broker)

        # Execute a successful trade
        signal = TradingSignal(
            signal_type=SignalType.BUY,
            symbol="EURUSD",
            confidence=0.8
        )
        manager.execute_signal(signal)

        stats = manager.get_statistics()
        assert stats['total_signals'] == 1
        assert stats['executed'] == 1


# ============================================================================
# Position Synchronizer Tests
# ============================================================================

class TestPositionSynchronizer:
    """Tests for PositionSynchronizer."""

    @pytest.fixture
    def mock_broker(self):
        """Create mock broker."""
        broker = Mock(spec=MT5BrokerGateway)
        broker.is_connected = True
        broker.magic_number = 234567
        broker.get_positions.return_value = []
        return broker

    def test_initialization(self, mock_broker):
        """Test PositionSynchronizer initialization."""
        sync = PositionSynchronizer(broker=mock_broker)

        assert sync.broker == mock_broker
        assert len(sync.get_positions()) == 0

    def test_sync_detects_new_position(self, mock_broker):
        """Test that sync detects new positions."""
        sync = PositionSynchronizer(broker=mock_broker)

        # First sync - no positions
        changes = sync.sync(force=True)
        assert len(changes) == 0

        # Add a position
        mock_position = MT5Position.from_mt5(MockPosition())
        mock_broker.get_positions.return_value = [mock_position]

        # Second sync - should detect new position
        changes = sync.sync(force=True)
        assert len(changes) == 1
        assert changes[0].event == PositionEvent.OPENED

    def test_sync_detects_closed_position(self, mock_broker):
        """Test that sync detects closed positions."""
        mock_position = MT5Position.from_mt5(MockPosition())
        mock_broker.get_positions.return_value = [mock_position]

        sync = PositionSynchronizer(broker=mock_broker)
        sync.sync(force=True)  # Detect the position

        # Remove the position
        mock_broker.get_positions.return_value = []

        # Should detect closure
        changes = sync.sync(force=True)
        assert len(changes) == 1
        assert changes[0].event in [PositionEvent.CLOSED, PositionEvent.SL_HIT, PositionEvent.TP_HIT]

    def test_callback_registration(self, mock_broker):
        """Test callback registration and firing."""
        sync = PositionSynchronizer(broker=mock_broker)

        callback_fired = []

        def on_opened(change):
            callback_fired.append(change)

        sync.register_callback(PositionEvent.OPENED, on_opened)

        # Add a position
        mock_position = MT5Position.from_mt5(MockPosition())
        mock_broker.get_positions.return_value = [mock_position]

        sync.sync(force=True)

        assert len(callback_fired) == 1

    def test_has_position(self, mock_broker):
        """Test has_position check."""
        mock_position = MT5Position.from_mt5(MockPosition())  # type=0 (BUY/long)
        mock_broker.get_positions.return_value = [mock_position]

        sync = PositionSynchronizer(broker=mock_broker)
        sync.sync(force=True)

        assert sync.has_position("EURUSD") is True
        assert sync.has_position("EURUSD", direction="long") is True
        assert sync.has_position("EURUSD", direction="short") is False
        assert sync.has_position("GBPUSD") is False


class TestPositionTracker:
    """Tests for lightweight PositionTracker."""

    def test_open_position(self):
        """Test opening a position."""
        tracker = PositionTracker()

        ticket = tracker.open_position(
            symbol="EURUSD",
            position_type="long",
            volume=0.1,
            entry_price=1.10000,
            sl=1.09500,
            tp=1.10500
        )

        assert ticket == 1
        positions = tracker.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "EURUSD"

    def test_close_position(self):
        """Test closing a position."""
        tracker = PositionTracker()

        ticket = tracker.open_position(
            symbol="EURUSD",
            position_type="long",
            volume=0.1,
            entry_price=1.10000
        )

        closed = tracker.close_position(ticket)

        assert closed is not None
        assert closed.ticket == ticket
        assert len(tracker.get_positions()) == 0

    def test_update_position(self):
        """Test updating position with current price."""
        tracker = PositionTracker()

        ticket = tracker.open_position(
            symbol="EURUSD",
            position_type="long",
            volume=0.1,
            entry_price=1.10000
        )

        # Update with higher price
        tracker.update_position(ticket, current_price=1.10100)

        position = tracker.get_position(ticket)
        assert position.unrealized_pnl > 0

    def test_has_position(self):
        """Test has_position check."""
        tracker = PositionTracker()

        tracker.open_position(
            symbol="EURUSD",
            position_type="long",
            volume=0.1,
            entry_price=1.10000
        )

        assert tracker.has_position("EURUSD") is True
        assert tracker.has_position("EURUSD", direction="long") is True
        assert tracker.has_position("EURUSD", direction="short") is False
        assert tracker.has_position("GBPUSD") is False


# ============================================================================
# Configuration Tests
# ============================================================================

class TestAutoTraderConfig:
    """Tests for AutoTraderConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from config.settings import AutoTraderConfig

        config = AutoTraderConfig()

        assert config.symbols == ['EURUSD']
        assert config.timeframe == '1h'
        assert config.risk_per_trade == 0.01
        assert config.max_positions == 3
        assert config.paper_mode is True

    def test_custom_config(self):
        """Test custom configuration."""
        from config.settings import AutoTraderConfig

        config = AutoTraderConfig(
            symbols=['GBPUSD', 'USDJPY'],
            risk_per_trade=0.02,
            paper_mode=False
        )

        assert config.symbols == ['GBPUSD', 'USDJPY']
        assert config.risk_per_trade == 0.02
        assert config.paper_mode is False


# ============================================================================
# Integration-like Tests (without actual MT5)
# ============================================================================

class TestAutoTraderIntegration:
    """Integration-style tests for AutoTrader without MT5."""

    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing."""
        broker = Mock(spec=MT5BrokerGateway)
        broker.is_connected = True
        broker.magic_number = 234567
        broker.get_account_info.return_value = AccountInfo.from_mt5(MockAccountInfo())
        broker.get_symbol_info.return_value = SymbolInfo.from_mt5(MockSymbolInfo())
        broker.get_current_tick.return_value = TickInfo.from_mt5(MockTickInfo(), "EURUSD")
        broker.get_positions.return_value = []
        broker.connect.return_value = True

        predictor = Mock()
        predictor.predict.return_value = {
            'prediction': np.array([[0.002]]),
            'uncertainty': 0.3
        }

        agent = Mock()
        agent.select_action.return_value = (1, 0.9)  # BUY action

        return broker, predictor, agent

    def test_signal_generation(self, mock_components):
        """Test that signals are generated correctly."""
        from core.auto_trader import AutoTrader, AutoTraderConfig

        broker, predictor, agent = mock_components

        config = AutoTraderConfig(
            symbols=['EURUSD'],
            paper_mode=True
        )

        trader = AutoTrader(
            broker=broker,
            predictor=predictor,
            agent=agent,
            config=config
        )

        # Generate signal
        signal = trader._generate_signal('EURUSD')

        # Should generate a signal based on mock predictions
        assert signal is not None
        assert signal.symbol == 'EURUSD'


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
