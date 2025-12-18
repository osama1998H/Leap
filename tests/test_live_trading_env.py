"""
Integration tests for LiveTradingEnvironment with different broker implementations.

Tests that LiveTradingEnvironment works correctly with:
- PaperBrokerGateway (paper trading mode)
- Mocked BrokerGateway (for unit testing)
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from core.live_trading_env import LiveTradingEnvironment
from core.paper_broker import PaperBrokerGateway, DefaultPriceProvider
from core.broker_interface import (
    BrokerGateway,
    BrokerPosition,
    PaperBrokerConfig,
    OrderType,
    AccountInfo,
    SymbolInfo,
    TickInfo,
    OrderResult,
)
from core.trading_types import Action, EnvConfig, Position


class TestLiveTradingEnvWithPaperBroker:
    """Tests for LiveTradingEnvironment with PaperBrokerGateway."""

    @pytest.fixture
    def paper_broker(self):
        """Create a connected paper broker."""
        config = PaperBrokerConfig(
            initial_balance=10000.0,
            leverage=100,
            commission_per_lot=7.0,
            magic_number=123456
        )
        broker = PaperBrokerGateway(config=config)
        broker.connect()
        yield broker
        broker.disconnect()

    @pytest.fixture
    def live_env(self, paper_broker):
        """Create a LiveTradingEnvironment with paper broker."""
        env = LiveTradingEnvironment(
            broker=paper_broker,
            symbol='EURUSD',
            data_pipeline=None,
            initial_balance=10000.0,
            window_size=60,
            feature_dim=100,
            match_training_obs=True
        )
        yield env

    def test_initialization(self, live_env, paper_broker):
        """Test environment initializes correctly with paper broker."""
        assert live_env.broker is paper_broker
        assert live_env.symbol == 'EURUSD'
        assert live_env.initial_balance == 10000.0

    def test_reset(self, live_env):
        """Test environment reset."""
        obs, info = live_env.reset()

        assert obs is not None
        assert isinstance(obs, np.ndarray)
        assert info['balance'] == 10000.0
        assert 'open_positions' in info

    def test_step_hold(self, live_env):
        """Test HOLD action."""
        live_env.reset()
        obs, reward, terminated, truncated, info = live_env.step(Action.HOLD)

        assert obs is not None
        assert not terminated
        assert info['open_positions'] == 0

    def test_observation_shape(self, live_env):
        """Test that observation shape is correct."""
        obs, _ = live_env.reset()

        # Expected: window_size * (5 + n_additional_features) + n_account_features
        # 60 * (5 + 100) + 8 = 60 * 105 + 8 = 6300 + 8 = 6308
        expected_dim = 60 * (5 + 100) + 8
        assert obs.shape == (expected_dim,), f"Expected {expected_dim}, got {obs.shape}"

    def test_broker_gateway_protocol_compliance(self, paper_broker):
        """Verify PaperBrokerGateway satisfies BrokerGateway protocol."""
        broker: BrokerGateway = paper_broker

        # All protocol methods should be callable
        assert callable(broker.connect)
        assert callable(broker.disconnect)
        assert callable(broker.get_account_info)
        assert callable(broker.get_symbol_info)
        assert callable(broker.get_current_tick)
        assert callable(broker.get_positions)
        assert callable(broker.send_market_order)
        assert callable(broker.close_position)
        assert callable(broker.close_all_positions)


class TestLiveTradingEnvWithMockedBroker:
    """Tests for LiveTradingEnvironment with mocked BrokerGateway."""

    @pytest.fixture
    def mock_broker(self):
        """Create a mock broker satisfying BrokerGateway protocol."""
        broker = Mock()

        # Properties
        broker.is_connected = True
        broker.magic_number = 234567

        # Connection methods
        broker.connect.return_value = True
        broker.disconnect.return_value = None

        # Account info
        broker.get_account_info.return_value = AccountInfo(
            login=12345,
            balance=50000.0,
            equity=50000.0,
            margin=0.0,
            free_margin=50000.0,
            margin_level=0.0,
            leverage=100,
            currency='USD',
            server='TestServer',
            company='TestBroker',
            trade_allowed=True
        )

        # Symbol info
        broker.get_symbol_info.return_value = SymbolInfo.create_forex_default('EURUSD')

        # Tick info
        broker.get_current_tick.return_value = TickInfo(
            symbol='EURUSD',
            bid=1.0850,
            ask=1.0852,
            last=1.0851,
            volume=1000.0,
            time=datetime.now()
        )

        # Positions
        broker.get_positions.return_value = []
        broker.get_positions_count.return_value = 0
        broker.get_position_by_ticket.return_value = None

        # Orders
        broker.send_market_order.return_value = OrderResult.success_result(
            ticket=1001,
            volume=0.1,
            price=1.0852,
            comment="Test order"
        )
        broker.close_position.return_value = OrderResult.success_result(
            ticket=1001,
            volume=0.1,
            price=1.0850,
            comment="Position closed"
        )
        broker.close_all_positions.return_value = []

        # Other methods
        broker.check_order.return_value = (True, "", 1.0852, 1.0850)
        broker.calculate_profit.return_value = 0.0
        broker.calculate_margin.return_value = 1000.0
        broker.get_trade_history.return_value = []
        broker.modify_position.return_value = OrderResult.success_result(
            ticket=1001,
            volume=0.1,
            price=1.0852,
            comment="Position modified"
        )

        return broker

    def test_env_accepts_any_broker_gateway(self, mock_broker):
        """Test that LiveTradingEnvironment accepts any BrokerGateway."""
        env = LiveTradingEnvironment(
            broker=mock_broker,
            symbol='EURUSD',
            data_pipeline=None,
            initial_balance=50000.0,
            window_size=60,
            feature_dim=100,
            match_training_obs=True
        )

        obs, info = env.reset()
        assert obs is not None
        assert info['balance'] == 50000.0

    def test_sync_positions_from_broker(self, mock_broker):
        """Test that positions sync correctly from broker to state."""
        # Set up broker to return a position
        test_position = BrokerPosition(
            ticket=1001,
            symbol='EURUSD',
            type=0,  # BUY
            volume=0.1,
            price_open=1.0850,
            price_current=1.0855,
            sl=1.0800,
            tp=1.0900,
            profit=5.0,
            swap=0.0,
            commission=0.7,
            magic=234567,
            comment="Test position",
            time=datetime.now()
        )
        mock_broker.get_positions.return_value = [test_position]

        env = LiveTradingEnvironment(
            broker=mock_broker,
            symbol='EURUSD',
            data_pipeline=None,
            initial_balance=50000.0,
            window_size=60,
            feature_dim=100,
            match_training_obs=True
        )

        env.reset()

        # Force sync via position_sync
        env.position_sync.sync(force=True)
        env._sync_positions_to_state()

        # Verify position was synced to state
        assert len(env.state.positions) == 1
        assert env.state.positions[0].type == 'long'
        assert env.state.positions[0].entry_price == 1.0850
        assert env.state.positions[0].size == 0.1

    def test_broker_position_to_position_conversion(self, mock_broker):
        """Test BrokerPosition to Position conversion."""
        env = LiveTradingEnvironment(
            broker=mock_broker,
            symbol='EURUSD',
            data_pipeline=None,
            initial_balance=50000.0,
            window_size=60,
            feature_dim=100,
            match_training_obs=True
        )

        broker_pos = BrokerPosition(
            ticket=1001,
            symbol='EURUSD',
            type=1,  # SELL
            volume=0.2,
            price_open=1.0860,
            price_current=1.0850,
            sl=1.0900,
            tp=1.0800,
            profit=10.0,
            swap=0.0,
            commission=1.4,
            magic=234567,
            comment="Test short",
            time=datetime.now()
        )

        position = env._broker_position_to_position(broker_pos)

        assert position.type == 'short'
        assert position.entry_price == 1.0860
        assert position.size == 0.2
        assert position.stop_loss == 1.0900
        assert position.take_profit == 1.0800


class TestBrokerAbstraction:
    """Tests verifying broker abstraction works correctly."""

    def test_paper_broker_creates_valid_positions(self):
        """Test that PaperBrokerGateway creates valid positions."""
        config = PaperBrokerConfig(
            initial_balance=10000.0,
            leverage=100,
            magic_number=123456
        )
        broker = PaperBrokerGateway(config=config)
        broker.connect()

        try:
            # Open a position
            result = broker.send_market_order(
                symbol='EURUSD',
                order_type=OrderType.BUY,
                volume=0.1,
                sl=1.0800,
                tp=1.0900
            )

            assert result.success
            assert result.ticket > 0

            # Verify position exists
            positions = broker.get_positions('EURUSD')
            assert len(positions) == 1
            assert positions[0].type == 0  # BUY
            assert positions[0].volume == 0.1
        finally:
            broker.disconnect()

    def test_paper_broker_account_info_updates(self):
        """Test that PaperBrokerGateway account info is accurate."""
        config = PaperBrokerConfig(
            initial_balance=10000.0,
            leverage=100,
            magic_number=123456
        )
        broker = PaperBrokerGateway(config=config)
        broker.connect()

        try:
            account = broker.get_account_info()
            assert account is not None
            assert account.balance == 10000.0
            assert account.equity == 10000.0
            assert account.leverage == 100
        finally:
            broker.disconnect()

    def test_both_brokers_satisfy_protocol(self):
        """Test that both broker types satisfy BrokerGateway protocol."""
        # Paper broker
        paper_config = PaperBrokerConfig(initial_balance=10000.0)
        paper_broker = PaperBrokerGateway(config=paper_config)

        # Verify all required methods exist
        required_methods = [
            'connect', 'disconnect', 'get_account_info', 'get_symbol_info',
            'get_current_tick', 'get_positions', 'get_position_by_ticket',
            'get_positions_count', 'send_market_order', 'close_position',
            'close_all_positions', 'modify_position', 'check_order',
            'calculate_profit', 'calculate_margin', 'get_trade_history'
        ]

        for method in required_methods:
            assert hasattr(paper_broker, method), f"Missing method: {method}"
            assert callable(getattr(paper_broker, method)), f"Not callable: {method}"

        # Verify properties
        assert hasattr(paper_broker, 'is_connected')
        assert hasattr(paper_broker, 'magic_number')
