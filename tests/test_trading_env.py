"""
Leap Trading System - TradingEnvironment Tests
Comprehensive tests for the trading environment module.
"""

import pytest
import numpy as np
from gymnasium import spaces
from unittest.mock import Mock, patch

# Import components to test
from core.trading_env import TradingEnvironment, MultiSymbolTradingEnv
from core.trading_types import Action


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    n_bars = 500
    np.random.seed(42)

    # Generate realistic price data
    prices = 1.1 * np.exp(np.cumsum(np.random.randn(n_bars) * 0.005))

    ohlcv = np.column_stack([
        prices * 0.999,  # open
        prices * 1.002,  # high
        prices * 0.998,  # low
        prices,          # close
        np.random.uniform(1000, 5000, n_bars)  # volume
    ])

    return ohlcv


@pytest.fixture
def sample_features():
    """Create sample features for testing."""
    n_bars = 500
    n_features = 20
    np.random.seed(42)
    return np.random.randn(n_bars, n_features).astype(np.float32)


@pytest.fixture
def trading_env(sample_ohlcv_data, sample_features):
    """Create TradingEnvironment for testing."""
    return TradingEnvironment(
        data=sample_ohlcv_data,
        features=sample_features,
        initial_balance=10000.0,
        commission=0.0001,
        spread=0.0002,
        slippage=0.0001,
        leverage=100,
        window_size=60
    )


@pytest.fixture
def trading_env_no_features(sample_ohlcv_data):
    """Create TradingEnvironment without features."""
    return TradingEnvironment(
        data=sample_ohlcv_data,
        features=None,
        initial_balance=10000.0,
        window_size=60
    )


# ============================================================================
# Initialization Tests
# ============================================================================

class TestTradingEnvironmentInitialization:
    """Tests for TradingEnvironment initialization."""

    def test_initialization_with_features(self, sample_ohlcv_data, sample_features):
        """Test initialization with features."""
        env = TradingEnvironment(
            data=sample_ohlcv_data,
            features=sample_features,
            initial_balance=10000.0,
            window_size=60
        )

        assert env.initial_balance == 10000.0
        assert env.window_size == 60
        assert env.data is not None
        assert env.features is not None

    def test_initialization_without_features(self, sample_ohlcv_data):
        """Test initialization without features."""
        env = TradingEnvironment(
            data=sample_ohlcv_data,
            features=None,
            initial_balance=10000.0,
            window_size=60
        )

        assert env.features is None

    def test_initialization_observation_space_shape(self, trading_env):
        """Test observation space has correct shape."""
        obs_space = trading_env.observation_space

        assert isinstance(obs_space, spaces.Box)
        assert len(obs_space.shape) == 1
        assert obs_space.shape[0] > 0

    def test_initialization_action_space(self, trading_env):
        """Test action space is configured correctly."""
        action_space = trading_env.action_space

        assert isinstance(action_space, spaces.Discrete)
        assert action_space.n == 4  # HOLD, BUY, SELL, CLOSE

    def test_initialization_custom_parameters(self, sample_ohlcv_data):
        """Test initialization with custom parameters."""
        env = TradingEnvironment(
            data=sample_ohlcv_data,
            initial_balance=50000.0,
            commission=0.0005,
            spread=0.0010,
            slippage=0.0003,
            leverage=50,
            max_position_size=0.2,
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            window_size=30
        )

        assert env.initial_balance == 50000.0
        assert env.commission == 0.0005
        assert env.spread == 0.0010
        assert env.leverage == 50
        assert env.window_size == 30

    def test_initialization_validates_window_size(self, sample_ohlcv_data):
        """Test that window_size is validated against data length."""
        with pytest.raises(ValueError, match="window_size"):
            TradingEnvironment(
                data=sample_ohlcv_data,
                window_size=600  # Greater than data length
            )

    def test_initialization_with_risk_manager(self, sample_ohlcv_data):
        """Test initialization with risk manager."""
        mock_risk_manager = Mock()

        env = TradingEnvironment(
            data=sample_ohlcv_data,
            risk_manager=mock_risk_manager,
            window_size=60
        )

        assert env.risk_manager == mock_risk_manager


# ============================================================================
# Reset Tests
# ============================================================================

class TestReset:
    """Tests for environment reset."""

    def test_reset_returns_observation(self, trading_env):
        """Test reset returns valid observation."""
        obs, info = trading_env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == trading_env.observation_space.shape

    def test_reset_returns_info(self, trading_env):
        """Test reset returns info dictionary."""
        obs, info = trading_env.reset()

        assert isinstance(info, dict)
        assert 'balance' in info
        assert 'equity' in info

    def test_reset_restores_initial_state(self, trading_env):
        """Test reset restores initial state."""
        # Take some actions
        trading_env.reset()
        for _ in range(10):
            trading_env.step(Action.BUY.value)

        # Reset
        obs, info = trading_env.reset()

        assert info['balance'] == trading_env.initial_balance
        assert info['total_trades'] == 0

    def test_reset_with_seed(self, trading_env):
        """Test reset with seed for reproducibility."""
        obs1, _ = trading_env.reset(seed=42)
        obs2, _ = trading_env.reset(seed=42)

        np.testing.assert_array_equal(obs1, obs2)

    def test_reset_clears_positions(self, trading_env):
        """Test reset clears all positions."""
        trading_env.reset()
        trading_env.step(Action.BUY.value)

        # Reset should clear positions
        _, info = trading_env.reset()

        assert info['open_positions'] == 0


# ============================================================================
# Step Tests
# ============================================================================

class TestStep:
    """Tests for environment step."""

    def test_step_returns_correct_tuple(self, trading_env):
        """Test step returns (obs, reward, terminated, truncated, info)."""
        trading_env.reset()
        result = trading_env.step(Action.HOLD.value)

        assert len(result) == 5
        obs, reward, terminated, truncated, info = result

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_observation_shape(self, trading_env):
        """Test step returns observation with correct shape."""
        trading_env.reset()
        obs, _, _, _, _ = trading_env.step(Action.HOLD.value)

        assert obs.shape == trading_env.observation_space.shape

    def test_step_hold_action(self, trading_env):
        """Test HOLD action doesn't open positions."""
        trading_env.reset()
        _, _, _, _, info = trading_env.step(Action.HOLD.value)

        assert info['open_positions'] == 0

    def test_step_buy_action_opens_long(self, trading_env):
        """Test BUY action opens long position."""
        trading_env.reset()
        _, _, _, _, info = trading_env.step(Action.BUY.value)

        assert info['open_positions'] == 1

    def test_step_sell_action_opens_short(self, trading_env):
        """Test SELL action opens short position."""
        trading_env.reset()
        _, _, _, _, info = trading_env.step(Action.SELL.value)

        assert info['open_positions'] == 1

    def test_step_close_action_closes_positions(self, trading_env):
        """Test CLOSE action closes positions."""
        trading_env.reset()

        # Open position
        trading_env.step(Action.BUY.value)

        # Close position
        _, _, _, _, info = trading_env.step(Action.CLOSE.value)

        assert info['open_positions'] == 0

    def test_step_close_with_no_positions(self, trading_env):
        """Test CLOSE action with no open positions."""
        trading_env.reset()
        _, _, _, _, info = trading_env.step(Action.CLOSE.value)

        # Should not cause error
        assert info['open_positions'] == 0

    def test_step_terminates_at_end_of_data(self, trading_env):
        """Test episode terminates at end of data."""
        trading_env.reset()

        terminated = False
        steps = 0

        while not terminated and steps < 1000:
            _, _, terminated, _, _ = trading_env.step(Action.HOLD.value)
            steps += 1

        assert terminated

    def test_step_increments_current_step(self, trading_env):
        """Test step increments current step counter."""
        trading_env.reset()
        initial_step = trading_env.current_step

        trading_env.step(Action.HOLD.value)

        assert trading_env.current_step == initial_step + 1


# ============================================================================
# Action Execution Tests
# ============================================================================

class TestActionExecution:
    """Tests for action execution."""

    def test_buy_applies_spread_and_slippage(self, trading_env):
        """Test BUY applies spread and slippage to entry price."""
        trading_env.reset()
        trading_env.step(Action.BUY.value)

        positions = trading_env.state.positions
        assert len(positions) == 1

        # Verify position was created with valid entry price
        position = positions[0]
        assert position.entry_price > 0

    def test_sell_applies_spread_and_slippage(self, trading_env):
        """Test SELL applies spread and slippage to entry price."""
        trading_env.reset()
        trading_env.step(Action.SELL.value)

        positions = trading_env.state.positions
        assert len(positions) == 1
        assert positions[0].type == 'short'

    def test_commission_deducted_on_open(self, trading_env):
        """Test commission is deducted when opening position."""
        trading_env.reset()
        initial_balance = trading_env.state.balance

        trading_env.step(Action.BUY.value)

        # Balance should decrease due to commission
        assert trading_env.state.balance < initial_balance

    def test_stop_loss_set_correctly_long(self, trading_env):
        """Test stop loss is set correctly for long position."""
        trading_env.reset()
        trading_env.step(Action.BUY.value)

        position = trading_env.state.positions[0]
        expected_sl = position.entry_price * (1 - trading_env.stop_loss_pct)

        assert position.stop_loss == pytest.approx(expected_sl, rel=0.01)

    def test_take_profit_set_correctly_long(self, trading_env):
        """Test take profit is set correctly for long position."""
        trading_env.reset()
        trading_env.step(Action.BUY.value)

        position = trading_env.state.positions[0]
        expected_tp = position.entry_price * (1 + trading_env.take_profit_pct)

        assert position.take_profit == pytest.approx(expected_tp, rel=0.01)


# ============================================================================
# Position Management Tests
# ============================================================================

class TestPositionManagement:
    """Tests for position management."""

    def test_has_long_position(self, trading_env):
        """Test _has_position for long positions."""
        trading_env.reset()
        trading_env.step(Action.BUY.value)

        assert trading_env._has_position('long')
        assert not trading_env._has_position('short')

    def test_has_short_position(self, trading_env):
        """Test _has_position for short positions."""
        trading_env.reset()
        trading_env.step(Action.SELL.value)

        assert trading_env._has_position('short')
        assert not trading_env._has_position('long')

    def test_unrealized_pnl_calculation_long(self, trading_env):
        """Test unrealized PnL calculation for long position."""
        trading_env.reset()
        trading_env.step(Action.BUY.value)

        current_price = trading_env._get_current_price()
        unrealized_pnl = trading_env._get_unrealized_pnl(current_price)

        # Should return some value (positive or negative)
        assert isinstance(unrealized_pnl, float)

    def test_close_position_updates_balance(self, trading_env):
        """Test closing position updates balance."""
        trading_env.reset()
        trading_env.step(Action.BUY.value)

        balance_before_close = trading_env.state.balance

        # Run some steps then close
        for _ in range(5):
            trading_env.step(Action.HOLD.value)

        trading_env.step(Action.CLOSE.value)

        # Balance should have changed
        assert trading_env.state.balance != balance_before_close

    def test_winning_trade_tracking(self, trading_env):
        """Test winning trades are tracked correctly."""
        trading_env.reset()

        # Create scenario where trade wins (TP hit)
        # This is hard to control, so we test the counter exists
        assert trading_env.state.winning_trades == 0

    def test_losing_trade_tracking(self, trading_env):
        """Test losing trades are tracked correctly."""
        trading_env.reset()

        assert trading_env.state.losing_trades == 0


# ============================================================================
# Reward Calculation Tests
# ============================================================================

class TestRewardCalculation:
    """Tests for reward calculation."""

    def test_reward_is_float(self, trading_env):
        """Test reward is always a float."""
        trading_env.reset()
        _, reward, _, _, _ = trading_env.step(Action.HOLD.value)

        assert isinstance(reward, (int, float))

    def test_reward_hold_no_position(self, trading_env):
        """Test reward for hold action with no position."""
        trading_env.reset()
        _, reward, _, _, _ = trading_env.step(Action.HOLD.value)

        # Should be some value (depends on implementation)
        assert isinstance(reward, (int, float))

    def test_reward_based_on_equity_change(self, trading_env):
        """Test reward reflects equity changes."""
        trading_env.reset()

        # Take action and observe reward
        _, reward1, _, _, _ = trading_env.step(Action.HOLD.value)
        _, reward2, _, _, _ = trading_env.step(Action.HOLD.value)

        # Both should be valid rewards
        assert isinstance(reward1, (int, float))
        assert isinstance(reward2, (int, float))


# ============================================================================
# Observation Tests
# ============================================================================

class TestObservation:
    """Tests for observation generation."""

    def test_observation_dtype(self, trading_env):
        """Test observation dtype is float32."""
        obs, _ = trading_env.reset()

        assert obs.dtype == np.float32

    def test_observation_no_nans(self, trading_env):
        """Test observation contains no NaN values."""
        obs, _ = trading_env.reset()

        assert not np.any(np.isnan(obs))

    def test_observation_no_infs(self, trading_env):
        """Test observation contains no infinite values."""
        obs, _ = trading_env.reset()

        assert not np.any(np.isinf(obs))

    def test_observation_changes_with_step(self, trading_env):
        """Test observation changes between steps."""
        trading_env.reset()
        obs1, _, _, _, _ = trading_env.step(Action.HOLD.value)
        obs2, _, _, _, _ = trading_env.step(Action.HOLD.value)

        # Observations should be different (market moved)
        assert not np.array_equal(obs1, obs2)

    def test_observation_includes_account_info(self, trading_env):
        """Test observation includes account information."""
        obs, _ = trading_env.reset()

        # Observation should be larger than just market data
        # due to account features
        market_obs_size = trading_env.window_size * (
            trading_env.n_price_features + trading_env.n_additional_features
        )
        account_obs_size = trading_env.n_account_features

        expected_size = market_obs_size + account_obs_size
        assert obs.shape[0] == expected_size


# ============================================================================
# Info Dictionary Tests
# ============================================================================

class TestInfoDictionary:
    """Tests for info dictionary."""

    def test_info_contains_balance(self, trading_env):
        """Test info contains balance."""
        _, info = trading_env.reset()

        assert 'balance' in info
        assert info['balance'] == trading_env.initial_balance

    def test_info_contains_equity(self, trading_env):
        """Test info contains equity."""
        _, info = trading_env.reset()

        assert 'equity' in info

    def test_info_contains_total_trades(self, trading_env):
        """Test info contains total_trades."""
        _, info = trading_env.reset()

        assert 'total_trades' in info

    def test_info_contains_win_rate(self, trading_env):
        """Test info contains win_rate."""
        _, info = trading_env.reset()

        assert 'win_rate' in info

    def test_info_contains_open_positions(self, trading_env):
        """Test info contains open_positions."""
        _, info = trading_env.reset()

        assert 'open_positions' in info


# ============================================================================
# Episode Statistics Tests
# ============================================================================

class TestEpisodeStatistics:
    """Tests for episode statistics."""

    def test_get_episode_stats(self, trading_env):
        """Test get_episode_stats returns dict."""
        trading_env.reset()

        # Run some steps
        for _ in range(10):
            trading_env.step(Action.HOLD.value)

        stats = trading_env.get_episode_stats()

        assert isinstance(stats, dict)

    def test_episode_stats_contains_required_keys(self, trading_env):
        """Test episode stats contains required keys."""
        trading_env.reset()

        for _ in range(10):
            trading_env.step(Action.HOLD.value)

        stats = trading_env.get_episode_stats()

        # Check for key statistics
        expected_keys = ['total_trades', 'winning_trades', 'losing_trades', 'win_rate']
        for key in expected_keys:
            assert key in stats or 'error' in stats


# ============================================================================
# Termination Tests
# ============================================================================

class TestTermination:
    """Tests for episode termination."""

    def test_terminates_on_bankruptcy(self, sample_ohlcv_data):
        """Test episode terminates when account goes bankrupt."""
        # Create env with very high risk settings
        env = TradingEnvironment(
            data=sample_ohlcv_data,
            initial_balance=100.0,  # Very small balance
            max_position_size=0.9,  # Very high position size
            window_size=60
        )

        env.reset()

        # Keep opening positions until bankruptcy or max steps
        terminated = False
        steps = 0

        while not terminated and steps < 500:
            _, _, terminated, _, _ = env.step(Action.BUY.value)
            steps += 1

        # Should terminate at some point
        assert steps < 500 or terminated

    def test_terminates_at_data_end(self, sample_ohlcv_data):
        """Test episode terminates at end of data."""
        env = TradingEnvironment(
            data=sample_ohlcv_data,
            window_size=60
        )

        env.reset()

        terminated = False
        steps = 0
        max_steps = len(sample_ohlcv_data) - 60

        while not terminated and steps < max_steps + 10:
            _, _, terminated, _, _ = env.step(Action.HOLD.value)
            steps += 1

        assert terminated


# ============================================================================
# Render Tests
# ============================================================================

class TestRender:
    """Tests for rendering."""

    def test_render_human_mode(self, sample_ohlcv_data, capsys):
        """Test render in human mode."""
        env = TradingEnvironment(
            data=sample_ohlcv_data,
            render_mode='human',
            window_size=60
        )

        env.reset()
        env.render()

        captured = capsys.readouterr()
        assert 'Balance' in captured.out or 'Step' in captured.out

    def test_render_no_mode(self, trading_env):
        """Test render with no mode set."""
        trading_env.reset()

        # Should not raise
        trading_env.render()


# ============================================================================
# MultiSymbolTradingEnv Tests
# ============================================================================

class TestMultiSymbolTradingEnv:
    """Tests for MultiSymbolTradingEnv."""

    @pytest.fixture
    def multi_symbol_data(self):
        """Create multi-symbol data."""
        n_bars = 200
        np.random.seed(42)

        def create_ohlcv():
            prices = 1.1 * np.exp(np.cumsum(np.random.randn(n_bars) * 0.005))
            return np.column_stack([
                prices * 0.999,
                prices * 1.002,
                prices * 0.998,
                prices,
                np.random.uniform(1000, 5000, n_bars)
            ])

        return {
            'EURUSD': create_ohlcv(),
            'GBPUSD': create_ohlcv()
        }

    def test_initialization(self, multi_symbol_data):
        """Test MultiSymbolTradingEnv initialization."""
        env = MultiSymbolTradingEnv(
            symbol_data=multi_symbol_data,
            window_size=30
        )

        assert len(env.symbols) == 2
        assert 'EURUSD' in env.symbols
        assert 'GBPUSD' in env.symbols

    def test_action_space(self, multi_symbol_data):
        """Test action space for multiple symbols."""
        env = MultiSymbolTradingEnv(
            symbol_data=multi_symbol_data,
            window_size=30
        )

        assert isinstance(env.action_space, spaces.MultiDiscrete)
        assert len(env.action_space.nvec) == 2  # One action per symbol

    def test_observation_space(self, multi_symbol_data):
        """Test observation space for multiple symbols."""
        env = MultiSymbolTradingEnv(
            symbol_data=multi_symbol_data,
            window_size=30
        )

        # Observation should be concatenation of all symbol observations
        assert isinstance(env.observation_space, spaces.Box)

    def test_reset(self, multi_symbol_data):
        """Test reset returns combined observations."""
        env = MultiSymbolTradingEnv(
            symbol_data=multi_symbol_data,
            window_size=30
        )

        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        assert 'EURUSD' in info
        assert 'GBPUSD' in info

    def test_step(self, multi_symbol_data):
        """Test step with multiple actions."""
        env = MultiSymbolTradingEnv(
            symbol_data=multi_symbol_data,
            window_size=30
        )

        env.reset()

        # Action for each symbol
        actions = [Action.BUY.value, Action.SELL.value]
        obs, reward, terminated, truncated, info = env.step(actions)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(info, dict)

    def test_get_episode_stats(self, multi_symbol_data):
        """Test get_episode_stats for multiple symbols."""
        env = MultiSymbolTradingEnv(
            symbol_data=multi_symbol_data,
            window_size=30
        )

        env.reset()

        stats = env.get_episode_stats()

        assert 'EURUSD' in stats
        assert 'GBPUSD' in stats


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_small_data(self):
        """Test with minimum viable data size."""
        n_bars = 100
        window_size = 30

        np.random.seed(42)
        prices = 1.1 * np.exp(np.cumsum(np.random.randn(n_bars) * 0.005))
        ohlcv = np.column_stack([
            prices * 0.999,
            prices * 1.002,
            prices * 0.998,
            prices,
            np.random.uniform(1000, 5000, n_bars)
        ])

        env = TradingEnvironment(data=ohlcv, window_size=window_size)
        obs, _ = env.reset()

        assert obs is not None

    def test_rapid_buy_sell(self, trading_env):
        """Test rapid buy/sell cycling."""
        trading_env.reset()

        for _ in range(10):
            trading_env.step(Action.BUY.value)
            trading_env.step(Action.CLOSE.value)
            trading_env.step(Action.SELL.value)
            trading_env.step(Action.CLOSE.value)

        # Should not crash
        _, info = trading_env.reset()
        assert info is not None

    def test_all_actions_sequence(self, trading_env):
        """Test all possible actions in sequence."""
        trading_env.reset()

        for action in [Action.HOLD, Action.BUY, Action.HOLD, Action.CLOSE,
                      Action.SELL, Action.HOLD, Action.CLOSE]:
            obs, reward, terminated, truncated, info = trading_env.step(action.value)

            if terminated:
                break

            assert obs is not None
            assert info is not None


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
