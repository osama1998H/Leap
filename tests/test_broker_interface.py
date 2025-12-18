"""
Tests for broker interface and paper broker implementation.

Tests the BrokerGateway Protocol, PaperBrokerGateway implementation,
and broker factory function.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import List

from core.broker_interface import (
    BrokerGateway,
    OrderType,
    TradeAction,
    AccountInfo,
    SymbolInfo,
    TickInfo,
    BrokerPosition,
    OrderResult,
    TradeHistory,
    PaperBrokerConfig,
    create_broker,
)
from core.paper_broker import (
    PaperBrokerGateway,
    DefaultPriceProvider,
)


class TestBrokerDataClasses:
    """Tests for broker data classes."""

    def test_account_info_create_paper(self):
        """Test AccountInfo paper trading factory."""
        account = AccountInfo.create_paper(
            balance=50000.0,
            leverage=100,
            currency="USD"
        )

        assert account.balance == 50000.0
        assert account.equity == 50000.0
        assert account.leverage == 100
        assert account.currency == "USD"
        assert account.trade_allowed is True
        assert account.server == "PaperTrading"

    def test_symbol_info_create_forex_default(self):
        """Test SymbolInfo forex default factory."""
        eurusd = SymbolInfo.create_forex_default("EURUSD")

        assert eurusd.name == "EURUSD"
        assert eurusd.digits == 5
        assert eurusd.point == 0.00001
        assert eurusd.volume_min == 0.01
        assert eurusd.trade_contract_size == 100000.0

    def test_symbol_info_jpy_pair(self):
        """Test SymbolInfo for JPY pairs."""
        usdjpy = SymbolInfo.create_forex_default("USDJPY")

        assert usdjpy.digits == 3
        assert usdjpy.point == 0.001

    def test_tick_info_spread(self):
        """Test TickInfo spread calculation."""
        tick = TickInfo(
            symbol="EURUSD",
            bid=1.0850,
            ask=1.0852,
            last=1.0851,
            volume=1000.0,
            time=datetime.now()
        )

        assert tick.spread == pytest.approx(0.0002, rel=1e-6)
        assert tick.mid == pytest.approx(1.0851, rel=1e-6)

    def test_broker_position_properties(self):
        """Test BrokerPosition properties."""
        long_position = BrokerPosition(
            ticket=1001,
            symbol="EURUSD",
            type=0,  # LONG
            volume=0.1,
            price_open=1.0850,
            price_current=1.0860,
            sl=1.0800,
            tp=1.0900,
            profit=10.0,
            swap=-0.5,
            commission=-3.5,
            magic=234567,
            comment="Test",
            time=datetime.now()
        )

        assert long_position.is_long is True
        assert long_position.is_short is False
        assert long_position.unrealized_pnl == 10.0 - 0.5 - 3.5

    def test_order_result_factory_methods(self):
        """Test OrderResult factory methods."""
        error = OrderResult.error("Test error", retcode=100)
        assert error.success is False
        assert error.comment == "Test error"
        assert error.retcode == 100

        success = OrderResult.success_result(
            ticket=1001,
            volume=0.1,
            price=1.0850,
            bid=1.0849,
            ask=1.0851
        )
        assert success.success is True
        assert success.ticket == 1001
        assert success.volume == 0.1


class TestDefaultPriceProvider:
    """Tests for DefaultPriceProvider."""

    def test_get_tick_known_symbol(self):
        """Test getting tick for known symbol."""
        provider = DefaultPriceProvider()
        tick = provider.get_tick("EURUSD")

        assert tick is not None
        assert tick.symbol == "EURUSD"
        assert tick.bid > 0
        assert tick.ask > tick.bid

    def test_get_tick_unknown_symbol(self):
        """Test getting tick for unknown symbol."""
        provider = DefaultPriceProvider()
        tick = provider.get_tick("UNKNOWN")

        assert tick is not None  # Should generate synthetic price

    def test_set_price(self):
        """Test setting custom price."""
        provider = DefaultPriceProvider()
        provider.set_price("CUSTOM", 1.5000, 1.5005)

        tick = provider.get_tick("CUSTOM")
        assert tick.bid == 1.5000
        assert tick.ask == 1.5005

    def test_get_symbol_info(self):
        """Test getting symbol info."""
        provider = DefaultPriceProvider()
        info = provider.get_symbol_info("GBPUSD")

        assert info is not None
        assert info.name == "GBPUSD"


class TestPaperBrokerGateway:
    """Tests for PaperBrokerGateway."""

    @pytest.fixture
    def broker(self):
        """Create a paper broker for testing."""
        config = PaperBrokerConfig(
            initial_balance=10000.0,
            leverage=100,
            commission_per_lot=7.0
        )
        broker = PaperBrokerGateway(config=config)
        broker.connect()
        yield broker
        broker.disconnect()

    def test_connect_disconnect(self):
        """Test connection management."""
        broker = PaperBrokerGateway()

        assert broker.is_connected is False
        assert broker.connect() is True
        assert broker.is_connected is True

        broker.disconnect()
        assert broker.is_connected is False

    def test_get_account_info(self, broker):
        """Test getting account information."""
        account = broker.get_account_info()

        assert account is not None
        assert account.balance == 10000.0
        assert account.equity == 10000.0
        assert account.leverage == 100
        assert account.trade_allowed is True

    def test_get_symbol_info(self, broker):
        """Test getting symbol information."""
        info = broker.get_symbol_info("EURUSD")

        assert info is not None
        assert info.name == "EURUSD"
        assert info.volume_min == 0.01

    def test_get_current_tick(self, broker):
        """Test getting current tick."""
        tick = broker.get_current_tick("EURUSD")

        assert tick is not None
        assert tick.symbol == "EURUSD"
        assert tick.bid > 0
        assert tick.ask > tick.bid

    def test_send_market_order_buy(self, broker):
        """Test sending buy market order."""
        result = broker.send_market_order(
            symbol="EURUSD",
            order_type=OrderType.BUY,
            volume=0.1,
            sl=1.0800,
            tp=1.0900
        )

        assert result.success is True
        assert result.ticket > 0
        assert result.volume == 0.1

        # Check position created
        positions = broker.get_positions()
        assert len(positions) == 1
        assert positions[0].is_long is True

    def test_send_market_order_sell(self, broker):
        """Test sending sell market order."""
        result = broker.send_market_order(
            symbol="EURUSD",
            order_type=OrderType.SELL,
            volume=0.1
        )

        assert result.success is True

        positions = broker.get_positions()
        assert len(positions) == 1
        assert positions[0].is_short is True

    def test_close_position(self, broker):
        """Test closing a position."""
        # Open a position
        open_result = broker.send_market_order(
            symbol="EURUSD",
            order_type=OrderType.BUY,
            volume=0.1
        )
        assert open_result.success is True

        # Close the position
        close_result = broker.close_position(open_result.ticket)
        assert close_result.success is True

        # Verify position is closed
        positions = broker.get_positions()
        assert len(positions) == 0

    def test_close_all_positions(self, broker):
        """Test closing all positions."""
        # Open multiple positions
        broker.send_market_order("EURUSD", OrderType.BUY, 0.1)
        broker.send_market_order("EURUSD", OrderType.SELL, 0.1)

        assert broker.get_positions_count() == 2

        # Close all
        results = broker.close_all_positions()
        assert len(results) == 2
        assert all(r.success for r in results)

        assert broker.get_positions_count() == 0

    def test_modify_position(self, broker):
        """Test modifying position SL/TP."""
        result = broker.send_market_order(
            symbol="EURUSD",
            order_type=OrderType.BUY,
            volume=0.1,
            sl=1.0800,
            tp=1.0900
        )

        # Modify SL/TP
        modify_result = broker.modify_position(
            ticket=result.ticket,
            sl=1.0750,
            tp=1.0950
        )

        assert modify_result.success is True

        # Verify changes
        position = broker.get_position_by_ticket(result.ticket)
        assert position.sl == 1.0750
        assert position.tp == 1.0950

    def test_check_order_valid(self, broker):
        """Test order validation for valid order."""
        is_valid, message, balance, equity = broker.check_order(
            symbol="EURUSD",
            order_type=OrderType.BUY,
            volume=0.1
        )

        assert is_valid is True
        assert balance > 0

    def test_check_order_excessive_volume(self, broker):
        """Test order validation for excessive volume (insufficient margin)."""
        # With 10000 balance and 100 leverage, max margin is ~1000 lots
        # Test with a very large volume to trigger margin check
        is_valid, message, _, _ = broker.check_order(
            symbol="EURUSD",
            order_type=OrderType.BUY,
            volume=10000.0  # Excessively large - should fail margin check
        )

        # Should fail due to insufficient margin or volume constraints
        assert is_valid is False

    def test_calculate_profit(self, broker):
        """Test profit calculation."""
        profit = broker.calculate_profit(
            symbol="EURUSD",
            order_type=OrderType.BUY,
            volume=1.0,
            open_price=1.0850,
            close_price=1.0860
        )

        # 10 pips * 1 lot * $10/pip = $100
        assert profit > 0

    def test_calculate_margin(self, broker):
        """Test margin calculation."""
        margin = broker.calculate_margin(
            symbol="EURUSD",
            order_type=OrderType.BUY,
            volume=1.0
        )

        assert margin is not None
        assert margin > 0

    def test_get_trade_history(self, broker):
        """Test trade history retrieval."""
        # Open and close a position
        result = broker.send_market_order("EURUSD", OrderType.BUY, 0.1)
        broker.close_position(result.ticket)

        # Get history
        history = broker.get_trade_history()
        assert len(history) == 1
        assert history[0].symbol == "EURUSD"

    def test_get_positions_by_symbol(self, broker):
        """Test getting positions filtered by symbol."""
        broker.send_market_order("EURUSD", OrderType.BUY, 0.1)
        broker.send_market_order("GBPUSD", OrderType.BUY, 0.1)

        eurusd_positions = broker.get_positions("EURUSD")
        assert len(eurusd_positions) == 1
        assert eurusd_positions[0].symbol == "EURUSD"

    def test_set_price_and_update(self, broker):
        """Test setting price and updating positions."""
        # Set initial price
        broker.set_price("EURUSD", 1.0850, 1.0852)

        # Open position
        result = broker.send_market_order("EURUSD", OrderType.BUY, 0.1)

        # Change price
        broker.set_price("EURUSD", 1.0860, 1.0862)

        # Get position and check profit
        position = broker.get_position_by_ticket(result.ticket)
        assert position.profit > 0

    def test_context_manager(self):
        """Test context manager usage."""
        with PaperBrokerGateway() as broker:
            assert broker.is_connected is True
            broker.send_market_order("EURUSD", OrderType.BUY, 0.1)

        # Should be disconnected after context


class TestBrokerFactory:
    """Tests for create_broker factory function."""

    def test_create_paper_broker(self):
        """Test creating paper broker."""
        broker = create_broker('paper')
        assert isinstance(broker, PaperBrokerGateway)

    def test_create_paper_broker_with_config(self):
        """Test creating paper broker with config."""
        config = PaperBrokerConfig(initial_balance=50000.0)
        broker = create_broker('paper', config=config)

        broker.connect()
        account = broker.get_account_info()
        assert account.balance == 50000.0
        broker.disconnect()

    def test_create_paper_broker_with_dict_config(self):
        """Test creating paper broker with dict config."""
        broker = create_broker('paper', config={'initial_balance': 25000.0})

        broker.connect()
        account = broker.get_account_info()
        assert account.balance == 25000.0
        broker.disconnect()

    def test_create_unknown_broker_raises(self):
        """Test that unknown broker type raises error."""
        with pytest.raises(ValueError, match="Unknown broker type"):
            create_broker('unknown')


class TestProtocolCompliance:
    """Tests to verify protocol compliance."""

    def test_paper_broker_satisfies_protocol(self):
        """Verify PaperBrokerGateway satisfies BrokerGateway protocol."""
        broker: BrokerGateway = PaperBrokerGateway()

        # Check all protocol methods exist and are callable
        assert callable(broker.connect)
        assert callable(broker.disconnect)
        assert callable(broker.get_account_info)
        assert callable(broker.get_symbol_info)
        assert callable(broker.get_current_tick)
        assert callable(broker.get_positions)
        assert callable(broker.get_position_by_ticket)
        assert callable(broker.get_positions_count)
        assert callable(broker.send_market_order)
        assert callable(broker.close_position)
        assert callable(broker.close_all_positions)
        assert callable(broker.modify_position)
        assert callable(broker.check_order)
        assert callable(broker.calculate_profit)
        assert callable(broker.calculate_margin)
        assert callable(broker.get_trade_history)

        # Check properties
        assert hasattr(broker, 'is_connected')
        assert hasattr(broker, 'magic_number')
