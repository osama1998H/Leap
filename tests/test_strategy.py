"""
Tests for trading strategy pattern implementation.

Tests the TradingStrategy ABC, CombinedPredictorAgentStrategy,
CallableStrategyAdapter, and strategy factory function.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock
from typing import List

from core.strategy import (
    TradingStrategy,
    StrategySignal,
    StrategyConfig,
    CombinedPredictorAgentStrategy,
    CallableStrategyAdapter,
    create_strategy,
)
from core.trading_types import Trade, Action
from core.order_manager import SignalType


class TestStrategySignal:
    """Tests for StrategySignal dataclass."""

    def test_create_signal(self):
        """Test creating a strategy signal."""
        signal = StrategySignal(
            action=SignalType.BUY,
            symbol="EURUSD",
            confidence=0.85,
            stop_loss_pips=50.0,
            take_profit_pips=100.0
        )

        assert signal.action == SignalType.BUY
        assert signal.symbol == "EURUSD"
        assert signal.confidence == 0.85
        assert signal.stop_loss_pips == 50.0
        assert signal.take_profit_pips == 100.0

    def test_signal_is_entry(self):
        """Test is_entry property."""
        buy_signal = StrategySignal(action=SignalType.BUY)
        sell_signal = StrategySignal(action=SignalType.SELL)
        hold_signal = StrategySignal(action=SignalType.HOLD)
        close_signal = StrategySignal(action=SignalType.CLOSE)

        assert buy_signal.is_entry is True
        assert sell_signal.is_entry is True
        assert hold_signal.is_entry is False
        assert close_signal.is_entry is False

    def test_signal_is_exit(self):
        """Test is_exit property."""
        close_signal = StrategySignal(action=SignalType.CLOSE)
        close_long_signal = StrategySignal(action=SignalType.CLOSE_LONG)
        close_short_signal = StrategySignal(action=SignalType.CLOSE_SHORT)
        buy_signal = StrategySignal(action=SignalType.BUY)

        assert close_signal.is_exit is True
        assert close_long_signal.is_exit is True
        assert close_short_signal.is_exit is True
        assert buy_signal.is_exit is False

    def test_signal_is_hold(self):
        """Test is_hold property."""
        hold_signal = StrategySignal(action=SignalType.HOLD)
        buy_signal = StrategySignal(action=SignalType.BUY)

        assert hold_signal.is_hold is True
        assert buy_signal.is_hold is False

    def test_to_backtest_dict(self):
        """Test conversion to backtest dictionary."""
        signal = StrategySignal(
            action=SignalType.BUY,
            stop_loss_pips=30.0,
            take_profit_pips=60.0
        )

        result = signal.to_backtest_dict()

        assert result['action'] == 'buy'
        assert result['stop_loss_pips'] == 30.0
        assert result['take_profit_pips'] == 60.0

    def test_to_backtest_dict_hold(self):
        """Test hold signal conversion."""
        signal = StrategySignal(action=SignalType.HOLD)

        result = signal.to_backtest_dict()

        assert result['action'] == 'hold'

    def test_to_backtest_dict_defaults(self):
        """Test default SL/TP values."""
        signal = StrategySignal(action=SignalType.BUY)

        result = signal.to_backtest_dict()

        assert result['stop_loss_pips'] == 50.0  # Default
        assert result['take_profit_pips'] == 100.0  # Default

    def test_to_trading_signal(self):
        """Test conversion to TradingSignal."""
        signal = StrategySignal(
            action=SignalType.BUY,
            symbol="EURUSD",
            confidence=0.9,
            stop_loss_pips=40.0,
            take_profit_pips=80.0,
            risk_percent=0.02,
            predicted_return=0.001,
            agent_action=Action.BUY
        )

        trading_signal = signal.to_trading_signal()

        assert trading_signal.signal_type == SignalType.BUY
        assert trading_signal.symbol == "EURUSD"
        assert trading_signal.confidence == 0.9
        assert trading_signal.stop_loss_pips == 40.0
        assert trading_signal.take_profit_pips == 80.0
        assert trading_signal.source == "strategy"
        assert trading_signal.metadata['predicted_return'] == 0.001
        assert trading_signal.metadata['agent_action'] == 'BUY'


class TestStrategyConfig:
    """Tests for StrategyConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StrategyConfig()

        assert config.min_confidence == 0.6
        assert config.prediction_threshold == 0.001
        assert config.default_sl_pips == 50.0
        assert config.default_tp_pips == 100.0
        assert config.risk_per_trade == 0.02
        assert config.lookback_window == 60

    def test_custom_config(self):
        """Test custom configuration values."""
        config = StrategyConfig(
            min_confidence=0.8,
            prediction_threshold=0.002,
            default_sl_pips=30.0,
            default_tp_pips=90.0,
            risk_per_trade=0.01,
            lookback_window=30
        )

        assert config.min_confidence == 0.8
        assert config.prediction_threshold == 0.002
        assert config.default_sl_pips == 30.0
        assert config.default_tp_pips == 90.0
        assert config.risk_per_trade == 0.01
        assert config.lookback_window == 30


class TestCombinedPredictorAgentStrategy:
    """Tests for CombinedPredictorAgentStrategy."""

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        n_bars = 100
        dates = pd.date_range('2024-01-01', periods=n_bars, freq='h')

        data = pd.DataFrame({
            'open': np.random.randn(n_bars).cumsum() + 100,
            'high': np.random.randn(n_bars).cumsum() + 101,
            'low': np.random.randn(n_bars).cumsum() + 99,
            'close': np.random.randn(n_bars).cumsum() + 100,
            'volume': np.random.randint(1000, 5000, n_bars).astype(float),
            'rsi_14': np.random.uniform(20, 80, n_bars),
            'macd': np.random.randn(n_bars),
            'sma_20': np.random.randn(n_bars).cumsum() + 100,
        }, index=dates)

        return data

    @pytest.fixture
    def mock_predictor(self):
        """Create mock predictor."""
        predictor = Mock()
        predictor.predict.return_value = {
            'prediction': np.array([[0.001]]),
            'uncertainty': 0.3
        }
        return predictor

    @pytest.fixture
    def mock_agent(self):
        """Create mock agent."""
        agent = Mock()
        agent.select_action.return_value = (1, 0.8, None)  # BUY action
        return agent

    def test_strategy_name(self):
        """Test strategy name property."""
        strategy = CombinedPredictorAgentStrategy()
        assert strategy.name == "combined_predictor_agent"

    def test_generate_signal_no_models(self, sample_market_data):
        """Test signal generation without models."""
        strategy = CombinedPredictorAgentStrategy()

        signal = strategy.generate_signal(
            market_data=sample_market_data,
            positions=[],
            symbol="EURUSD"
        )

        assert signal.action == SignalType.HOLD
        assert signal.symbol == "EURUSD"

    def test_generate_signal_with_predictor(self, sample_market_data, mock_predictor):
        """Test signal generation with predictor only."""
        strategy = CombinedPredictorAgentStrategy(
            predictor=mock_predictor,
            feature_names=['rsi_14', 'macd', 'sma_20']
        )

        signal = strategy.generate_signal(
            market_data=sample_market_data,
            positions=[],
            symbol="EURUSD"
        )

        # With only predictor and no agent, should HOLD (agent_action is HOLD by default)
        assert signal.symbol == "EURUSD"
        assert signal.predicted_return == pytest.approx(0.001, rel=0.1)

    def test_generate_signal_with_both_models(
        self, sample_market_data, mock_predictor, mock_agent
    ):
        """Test signal generation with both predictor and agent."""
        strategy = CombinedPredictorAgentStrategy(
            predictor=mock_predictor,
            agent=mock_agent,
            feature_names=['rsi_14', 'macd', 'sma_20']
        )

        signal = strategy.generate_signal(
            market_data=sample_market_data,
            positions=[],
            symbol="EURUSD"
        )

        # Both predict positive/BUY, should be BUY
        assert signal.action == SignalType.BUY
        assert signal.predicted_return == pytest.approx(0.001, rel=0.1)
        assert signal.agent_action == Action.BUY

    def test_generate_signal_contradiction(
        self, sample_market_data, mock_agent
    ):
        """Test signal generation with contradicting signals."""
        # Predictor predicts negative, agent says BUY
        mock_predictor = Mock()
        mock_predictor.predict.return_value = {
            'prediction': np.array([[-0.002]]),  # Strong negative prediction
            'uncertainty': 0.2
        }

        strategy = CombinedPredictorAgentStrategy(
            predictor=mock_predictor,
            agent=mock_agent,
            feature_names=['rsi_14', 'macd', 'sma_20']
        )

        signal = strategy.generate_signal(
            market_data=sample_market_data,
            positions=[],
            symbol="EURUSD"
        )

        # Strong contradiction should result in HOLD
        assert signal.action == SignalType.HOLD

    def test_generate_signal_close_action(self, sample_market_data, mock_predictor):
        """Test signal generation with CLOSE action from agent."""
        mock_agent = Mock()
        mock_agent.select_action.return_value = (3, 0.9, None)  # CLOSE action

        strategy = CombinedPredictorAgentStrategy(
            predictor=mock_predictor,
            agent=mock_agent,
            feature_names=['rsi_14', 'macd', 'sma_20']
        )

        signal = strategy.generate_signal(
            market_data=sample_market_data,
            positions=[],
            symbol="EURUSD"
        )

        assert signal.action == SignalType.CLOSE

    def test_generate_signal_low_confidence(
        self, sample_market_data, mock_agent
    ):
        """Test signal generation with low confidence."""
        # High uncertainty = low confidence
        mock_predictor = Mock()
        mock_predictor.predict.return_value = {
            'prediction': np.array([[0.001]]),
            'uncertainty': 0.6  # High uncertainty
        }

        config = StrategyConfig(min_confidence=0.6)
        strategy = CombinedPredictorAgentStrategy(
            predictor=mock_predictor,
            agent=mock_agent,
            config=config,
            feature_names=['rsi_14', 'macd', 'sma_20']
        )

        signal = strategy.generate_signal(
            market_data=sample_market_data,
            positions=[],
            symbol="EURUSD"
        )

        # Low confidence should result in HOLD
        assert signal.action == SignalType.HOLD

    def test_generate_signal_open_status_false(
        self, sample_market_data, mock_predictor, mock_agent
    ):
        """Test signal generation when new positions not allowed."""
        strategy = CombinedPredictorAgentStrategy(
            predictor=mock_predictor,
            agent=mock_agent,
            feature_names=['rsi_14', 'macd', 'sma_20']
        )

        signal = strategy.generate_signal(
            market_data=sample_market_data,
            positions=[],
            symbol="EURUSD",
            open_status=False
        )

        # Cannot open new positions, should HOLD
        assert signal.action == SignalType.HOLD

    def test_reset(self):
        """Test strategy reset."""
        strategy = CombinedPredictorAgentStrategy()
        strategy._last_prediction = {'test': True}
        strategy._last_action = Action.BUY

        strategy.reset()

        assert strategy._last_prediction is None
        assert strategy._last_action is None

    def test_insufficient_data(self):
        """Test with insufficient market data."""
        strategy = CombinedPredictorAgentStrategy()

        # Only 10 bars, but lookback is 60
        small_data = pd.DataFrame({
            'close': np.random.randn(10),
            'feature1': np.random.randn(10)
        })

        signal = strategy.generate_signal(
            market_data=small_data,
            positions=[]
        )

        # Should return HOLD due to insufficient data
        assert signal.action == SignalType.HOLD


class TestCallableStrategyAdapter:
    """Tests for CallableStrategyAdapter."""

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        return pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1100, 1200]
        })

    def test_adapter_name(self):
        """Test adapter name property."""
        def dummy_strategy(data, **kwargs):
            return {'action': 'hold'}

        adapter = CallableStrategyAdapter(dummy_strategy, "my_strategy")
        assert adapter.name == "my_strategy"

    def test_adapter_default_name(self):
        """Test adapter default name."""
        def dummy_strategy(data, **kwargs):
            return {'action': 'hold'}

        adapter = CallableStrategyAdapter(dummy_strategy)
        assert adapter.name == "callable_strategy"

    def test_adapter_generate_signal_buy(self, sample_market_data):
        """Test adapter generating BUY signal."""
        def buy_strategy(data, **kwargs):
            return {
                'action': 'buy',
                'stop_loss_pips': 30.0,
                'take_profit_pips': 60.0
            }

        adapter = CallableStrategyAdapter(buy_strategy)
        signal = adapter.generate_signal(
            market_data=sample_market_data,
            positions=[],
            symbol="EURUSD"
        )

        assert signal.action == SignalType.BUY
        assert signal.symbol == "EURUSD"
        assert signal.stop_loss_pips == 30.0
        assert signal.take_profit_pips == 60.0

    def test_adapter_generate_signal_sell(self, sample_market_data):
        """Test adapter generating SELL signal."""
        def sell_strategy(data, **kwargs):
            return {'action': 'sell'}

        adapter = CallableStrategyAdapter(sell_strategy)
        signal = adapter.generate_signal(
            market_data=sample_market_data,
            positions=[]
        )

        assert signal.action == SignalType.SELL

    def test_adapter_generate_signal_close(self, sample_market_data):
        """Test adapter generating CLOSE signal."""
        def close_strategy(data, **kwargs):
            return {'action': 'close'}

        adapter = CallableStrategyAdapter(close_strategy)
        signal = adapter.generate_signal(
            market_data=sample_market_data,
            positions=[]
        )

        assert signal.action == SignalType.CLOSE

    def test_adapter_passes_kwargs(self, sample_market_data):
        """Test that adapter passes kwargs to callable."""
        received_kwargs = {}

        def capturing_strategy(data, **kwargs):
            received_kwargs.update(kwargs)
            return {'action': 'hold'}

        mock_predictor = Mock()
        mock_agent = Mock()

        adapter = CallableStrategyAdapter(capturing_strategy)
        adapter.generate_signal(
            market_data=sample_market_data,
            positions=[],
            predictor=mock_predictor,
            agent=mock_agent
        )

        assert received_kwargs['predictor'] is mock_predictor
        assert received_kwargs['agent'] is mock_agent

    def test_adapter_unknown_action(self, sample_market_data):
        """Test adapter with unknown action defaults to HOLD."""
        def unknown_strategy(data, **kwargs):
            return {'action': 'unknown'}

        adapter = CallableStrategyAdapter(unknown_strategy)
        signal = adapter.generate_signal(
            market_data=sample_market_data,
            positions=[]
        )

        assert signal.action == SignalType.HOLD

    def test_adapter_case_insensitive(self, sample_market_data):
        """Test adapter handles action case insensitively."""
        def uppercase_strategy(data, **kwargs):
            return {'action': 'BUY'}

        adapter = CallableStrategyAdapter(uppercase_strategy)
        signal = adapter.generate_signal(
            market_data=sample_market_data,
            positions=[]
        )

        assert signal.action == SignalType.BUY


class TestCreateStrategy:
    """Tests for create_strategy factory function."""

    def test_create_combined_strategy(self):
        """Test creating combined strategy."""
        mock_predictor = Mock()
        mock_agent = Mock()

        strategy = create_strategy(
            'combined',
            predictor=mock_predictor,
            agent=mock_agent
        )

        assert isinstance(strategy, CombinedPredictorAgentStrategy)
        assert strategy.predictor is mock_predictor
        assert strategy.agent is mock_agent

    def test_create_combined_strategy_with_config(self):
        """Test creating combined strategy with config."""
        config = StrategyConfig(min_confidence=0.8)

        strategy = create_strategy('combined', config=config)

        assert strategy.config.min_confidence == 0.8

    def test_create_callable_strategy(self):
        """Test creating callable strategy."""
        def my_func(data, **kwargs):
            return {'action': 'hold'}

        strategy = create_strategy(
            'callable',
            callable_fn=my_func,
            strategy_name='my_func_strategy'
        )

        assert isinstance(strategy, CallableStrategyAdapter)
        assert strategy.name == 'my_func_strategy'

    def test_create_callable_strategy_requires_fn(self):
        """Test callable strategy requires callable_fn."""
        with pytest.raises(ValueError, match="callable_fn required"):
            create_strategy('callable')

    def test_create_unknown_strategy_raises(self):
        """Test unknown strategy type raises error."""
        with pytest.raises(ValueError, match="Unknown strategy type"):
            create_strategy('unknown')

    def test_create_strategy_case_insensitive(self):
        """Test strategy type is case insensitive."""
        strategy = create_strategy('COMBINED')
        assert isinstance(strategy, CombinedPredictorAgentStrategy)


class TestTradingStrategyABC:
    """Tests to verify TradingStrategy ABC behavior."""

    def test_cannot_instantiate_abc(self):
        """Test that TradingStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            TradingStrategy()

    def test_subclass_must_implement_name(self):
        """Test subclass must implement name property."""
        class IncompleteStrategy(TradingStrategy):
            def generate_signal(self, market_data, positions, **kwargs):
                return StrategySignal(action=SignalType.HOLD)

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteStrategy()

    def test_subclass_must_implement_generate_signal(self):
        """Test subclass must implement generate_signal."""
        class IncompleteStrategy(TradingStrategy):
            @property
            def name(self):
                return "incomplete"

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteStrategy()

    def test_valid_subclass(self):
        """Test valid TradingStrategy subclass."""
        class ValidStrategy(TradingStrategy):
            @property
            def name(self):
                return "valid_strategy"

            def generate_signal(self, market_data, positions, **kwargs):
                return StrategySignal(action=SignalType.HOLD)

        strategy = ValidStrategy()
        assert strategy.name == "valid_strategy"

        signal = strategy.generate_signal(pd.DataFrame(), [])
        assert signal.action == SignalType.HOLD

    def test_callbacks_have_default_implementations(self):
        """Test optional callbacks have default implementations."""
        class MinimalStrategy(TradingStrategy):
            @property
            def name(self):
                return "minimal"

            def generate_signal(self, market_data, positions, **kwargs):
                return StrategySignal(action=SignalType.HOLD)

        strategy = MinimalStrategy()

        # These should not raise
        strategy.on_trade_opened(Mock())
        strategy.on_trade_closed(Mock())
        strategy.reset()


class TestStrategyIntegration:
    """Integration tests for strategy components."""

    @pytest.fixture
    def sample_market_data(self):
        """Create realistic sample market data."""
        n_bars = 100
        np.random.seed(42)

        close = np.cumsum(np.random.randn(n_bars) * 0.001) + 1.10
        high = close + np.random.uniform(0.0001, 0.001, n_bars)
        low = close - np.random.uniform(0.0001, 0.001, n_bars)
        open_ = low + (high - low) * np.random.uniform(0.3, 0.7, n_bars)

        data = pd.DataFrame({
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.randint(1000, 5000, n_bars).astype(float),
            'rsi_14': np.random.uniform(20, 80, n_bars),
            'macd': np.random.randn(n_bars) * 0.001,
            'sma_20': close.copy(),
            'ema_12': close.copy(),
            'atr_14': np.random.uniform(0.0005, 0.002, n_bars),
        })

        return data

    def test_strategy_backtest_compatibility(self, sample_market_data):
        """Test strategy signal can be converted to backtest format."""
        def simple_strategy(data, **kwargs):
            if len(data) > 0 and data['rsi_14'].iloc[-1] < 30:
                return {'action': 'buy', 'stop_loss_pips': 50, 'take_profit_pips': 100}
            elif len(data) > 0 and data['rsi_14'].iloc[-1] > 70:
                return {'action': 'sell', 'stop_loss_pips': 50, 'take_profit_pips': 100}
            return {'action': 'hold'}

        adapter = CallableStrategyAdapter(simple_strategy, "rsi_strategy")

        # Generate signal
        signal = adapter.generate_signal(
            market_data=sample_market_data,
            positions=[]
        )

        # Convert to backtest format
        backtest_dict = signal.to_backtest_dict()

        assert 'action' in backtest_dict
        assert 'stop_loss_pips' in backtest_dict
        assert 'take_profit_pips' in backtest_dict
        assert backtest_dict['action'] in ['buy', 'sell', 'hold', 'close']

    def test_combined_strategy_full_workflow(self, sample_market_data):
        """Test combined strategy full workflow."""
        # Create mock models
        mock_predictor = Mock()
        mock_predictor.predict.return_value = {
            'prediction': np.array([[0.0015]]),  # Positive prediction
            'uncertainty': 0.25
        }

        mock_agent = Mock()
        mock_agent.select_action.return_value = (1, 0.85, None)  # BUY

        # Create strategy
        strategy = CombinedPredictorAgentStrategy(
            predictor=mock_predictor,
            agent=mock_agent,
            config=StrategyConfig(min_confidence=0.6),
            feature_names=['rsi_14', 'macd', 'sma_20', 'ema_12', 'atr_14']
        )

        # Generate signal
        signal = strategy.generate_signal(
            market_data=sample_market_data,
            positions=[],
            symbol="EURUSD"
        )

        # Should be BUY (agreement between predictor and agent)
        assert signal.action == SignalType.BUY
        assert signal.symbol == "EURUSD"
        assert signal.confidence > 0.6
        assert signal.predicted_return > 0
        assert signal.agent_action == Action.BUY

        # Convert to trading signal for order execution
        trading_signal = signal.to_trading_signal()
        assert trading_signal.signal_type == SignalType.BUY
        assert trading_signal.source == "strategy"

        # Convert to backtest format
        backtest_dict = signal.to_backtest_dict()
        assert backtest_dict['action'] == 'buy'


class TestBacktesterIntegration:
    """Tests for Backtester + TradingStrategy integration."""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV DataFrame for backtesting."""
        n_bars = 200
        np.random.seed(42)
        close = np.cumsum(np.random.randn(n_bars) * 0.001) + 1.10
        return pd.DataFrame({
            'open': close - 0.0005,
            'high': close + 0.001,
            'low': close - 0.001,
            'close': close,
            'volume': np.random.randint(1000, 5000, n_bars).astype(float),
        })

    def test_backtester_accepts_strategy_instance(self, sample_ohlcv_data):
        """Test Backtester accepts TradingStrategy instance."""
        from evaluation.backtester import Backtester

        strategy = CombinedPredictorAgentStrategy()
        backtester = Backtester(initial_balance=10000.0)

        result = backtester.run(
            data=sample_ohlcv_data,
            strategy=strategy,
            show_progress=False
        )

        assert result is not None
        assert isinstance(result.total_return, float)

    def test_backtester_callable_deprecation_warning(self, sample_ohlcv_data):
        """Test deprecation warning is shown for callable strategies."""
        from evaluation.backtester import Backtester
        import warnings

        def legacy_strategy(market_data, **kwargs):
            return {'action': 'hold'}

        backtester = Backtester(initial_balance=10000.0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            backtester.run(
                data=sample_ohlcv_data,
                strategy=legacy_strategy,
                show_progress=False
            )

            # Should have exactly one deprecation warning
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 1
            assert "deprecated" in str(deprecation_warnings[0].message).lower()

    def test_backtester_callable_warning_once_per_instance(self, sample_ohlcv_data):
        """Test deprecation warning is shown only once per backtester instance."""
        from evaluation.backtester import Backtester
        import warnings

        def legacy_strategy(market_data, **kwargs):
            return {'action': 'hold'}

        backtester = Backtester(initial_balance=10000.0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Run twice with same backtester instance
            backtester.run(data=sample_ohlcv_data, strategy=legacy_strategy, show_progress=False)
            backtester.run(data=sample_ohlcv_data, strategy=legacy_strategy, show_progress=False)

            # Should still have only one warning (per instance)
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 1

    def test_strategy_reset_called_at_start(self, sample_ohlcv_data):
        """Test strategy.reset() is called at backtest start."""
        from evaluation.backtester import Backtester

        mock_strategy = Mock(spec=TradingStrategy)
        mock_strategy.name = "mock_strategy"
        mock_strategy.generate_signal.return_value = StrategySignal(action=SignalType.HOLD)

        backtester = Backtester(initial_balance=10000.0)
        backtester.run(data=sample_ohlcv_data, strategy=mock_strategy, show_progress=False)

        mock_strategy.reset.assert_called_once()

    def test_strategy_lifecycle_callbacks_invoked(self, sample_ohlcv_data):
        """Test on_trade_opened and on_trade_closed callbacks are invoked."""
        from evaluation.backtester import Backtester

        class TrackingStrategy(TradingStrategy):
            def __init__(self):
                self.trades_opened = []
                self.trades_closed = []
                self._signal_count = 0

            @property
            def name(self):
                return "tracking"

            def generate_signal(self, market_data, positions, **kwargs):
                self._signal_count += 1
                # Open a trade on bar 10, close on bar 20
                if self._signal_count == 10:
                    return StrategySignal(
                        action=SignalType.BUY,
                        stop_loss_pips=50.0,
                        take_profit_pips=100.0
                    )
                elif self._signal_count == 20 and len(positions) > 0:
                    return StrategySignal(action=SignalType.CLOSE)
                return StrategySignal(action=SignalType.HOLD)

            def on_trade_opened(self, trade):
                self.trades_opened.append(trade)

            def on_trade_closed(self, trade):
                self.trades_closed.append(trade)

        strategy = TrackingStrategy()
        backtester = Backtester(initial_balance=10000.0)
        backtester.run(data=sample_ohlcv_data, strategy=strategy, show_progress=False)

        # Should have opened and closed at least one trade
        assert len(strategy.trades_opened) >= 1, "on_trade_opened should be called"
        assert len(strategy.trades_closed) >= 1, "on_trade_closed should be called"

    def test_no_warning_for_strategy_instance(self, sample_ohlcv_data):
        """Test no deprecation warning when using TradingStrategy instance."""
        from evaluation.backtester import Backtester
        import warnings

        strategy = CombinedPredictorAgentStrategy()
        backtester = Backtester(initial_balance=10000.0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            backtester.run(
                data=sample_ohlcv_data,
                strategy=strategy,
                show_progress=False
            )

            # Should have no deprecation warnings
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0
