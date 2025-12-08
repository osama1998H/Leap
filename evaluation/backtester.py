"""
Leap Trading System - Backtesting Framework
Comprehensive backtesting with walk-forward optimization and Monte Carlo simulation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging
import json
import os

if TYPE_CHECKING:
    from core.risk_manager import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    direction: str  # 'long' or 'short'
    size: float
    stop_loss: Optional[float] = None  # Stop loss price level
    take_profit: Optional[float] = None  # Take profit price level
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    status: str = 'open'  # 'open', 'closed', 'stopped'


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    # Performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # in periods

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    avg_winner: float
    avg_loser: float
    largest_winner: float
    largest_loser: float
    avg_trade_duration: float

    # Risk metrics
    volatility: float
    downside_volatility: float
    var_95: float
    cvar_95: float
    skewness: float
    kurtosis: float

    # Time series data
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    drawdown_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    returns: np.ndarray = field(default_factory=lambda: np.array([]))
    trades: List[Trade] = field(default_factory=list)

    # Metadata
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    initial_balance: float = 10000.0


class Backtester:
    """
    Advanced backtesting engine for trading strategies.

    Features:
    - Realistic transaction costs (spread, commission, slippage)
    - Position management
    - Risk management (stop loss, take profit)
    - Detailed performance metrics
    - Trade-by-trade analysis
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        commission_rate: float = 0.0001,  # 0.01% per trade
        spread_pips: float = 1.5,
        slippage_pips: float = 0.5,
        pip_value: float = 0.0001,  # For forex pairs like EURUSD
        leverage: int = 100,
        risk_per_trade: float = 0.02,  # 2% risk per trade
        max_positions: int = 5,
        risk_manager: Optional['RiskManager'] = None
    ):
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.spread_pips = spread_pips
        self.slippage_pips = slippage_pips
        self.pip_value = pip_value
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.risk_manager = risk_manager

        # State
        self.balance = initial_balance
        self.equity = initial_balance
        self.positions: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[datetime] = []

    def reset(self):
        """Reset backtester state."""
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.positions = []
        self.closed_trades = []
        self.equity_curve = [self.initial_balance]
        self.timestamps = []

    def run(
        self,
        data: pd.DataFrame,
        strategy: Callable,
        predictor=None,
        agent=None
    ) -> BacktestResult:
        """
        Run backtest with given strategy.

        Args:
            data: DataFrame with OHLCV data and features
            strategy: Strategy function that returns actions
            predictor: Optional prediction model
            agent: Optional RL agent

        Returns:
            BacktestResult with comprehensive metrics
        """
        self.reset()

        n_steps = len(data)
        logger.info(f"Running backtest on {n_steps} bars...")

        for i in range(1, n_steps):
            current_bar = data.iloc[i]
            timestamp = current_bar.name if isinstance(current_bar.name, datetime) else datetime.now()

            # Get current price
            current_price = current_bar['close']
            high = current_bar['high']
            low = current_bar['low']

            # Update existing positions (check stop loss / take profit)
            self._update_positions(high, low, timestamp)

            # Get trading signal
            signal = strategy(
                data.iloc[:i + 1],
                predictor=predictor,
                agent=agent,
                positions=self.positions
            )

            # Execute signal
            self._execute_signal(signal, current_price, timestamp)

            # Update equity
            self._update_equity(current_price)
            self.equity_curve.append(self.equity)
            self.timestamps.append(timestamp)

        # Close all remaining positions
        if len(data) > 0:
            final_price = data.iloc[-1]['close']
            final_time = data.iloc[-1].name if isinstance(data.iloc[-1].name, datetime) else datetime.now()
            self._close_all_positions(final_price, final_time)

            # Update final equity after closing positions (balance is now the true equity)
            # Since all positions are closed, equity = balance
            self.equity = self.balance
            self.equity_curve.append(self.equity)
            self.timestamps.append(final_time)

        # Calculate results
        result = self._calculate_results()

        logger.info(f"Backtest complete. Total return: {result.total_return:.2%}")

        return result

    def _execute_signal(
        self,
        signal: Dict,
        price: float,
        timestamp: datetime
    ):
        """Execute trading signal."""
        action = signal.get('action', 'hold')

        if action == 'buy' and len(self.positions) < self.max_positions:
            self._open_position('long', price, timestamp, signal)

        elif action == 'sell' and len(self.positions) < self.max_positions:
            self._open_position('short', price, timestamp, signal)

        elif action == 'close':
            self._close_all_positions(price, timestamp)

        elif action == 'close_long':
            self._close_positions_by_type('long', price, timestamp)

        elif action == 'close_short':
            self._close_positions_by_type('short', price, timestamp)

    def _open_position(
        self,
        direction: str,
        price: float,
        timestamp: datetime,
        signal: Dict
    ):
        """Open a new position."""
        # Apply slippage
        slippage = self.slippage_pips * self.pip_value
        if direction == 'long':
            entry_price = price * (1 + self.spread_pips * self.pip_value / 2 + slippage)
        else:
            entry_price = price * (1 - self.spread_pips * self.pip_value / 2 - slippage)

        # Get stop loss and take profit from signal
        # Validate stop_loss_pips FIRST to ensure consistency between pricing and sizing
        stop_loss_pips = signal.get('stop_loss_pips', 50)
        MIN_STOP_PIPS = 1.0  # Minimum 1 pip stop loss
        if stop_loss_pips <= 0:
            raise ValueError(
                f"stop_loss_pips must be positive, got {stop_loss_pips}. "
                f"Check signal['stop_loss_pips'] or provide a valid default."
            )
        safe_stop_pips = max(stop_loss_pips, MIN_STOP_PIPS)

        # Use safe_stop_pips for take profit default to maintain R:R ratio
        take_profit_pips = signal.get('take_profit_pips', safe_stop_pips * 2)  # Default 2:1 R:R

        # Calculate stop loss and take profit price levels using safe_stop_pips
        # This ensures pricing matches the stop distance used for position sizing
        if direction == 'long':
            stop_loss_price = entry_price * (1 - safe_stop_pips * self.pip_value)
            take_profit_price = entry_price * (1 + take_profit_pips * self.pip_value)
        else:
            stop_loss_price = entry_price * (1 + safe_stop_pips * self.pip_value)
            take_profit_price = entry_price * (1 - take_profit_pips * self.pip_value)

        # Calculate position size based on risk (using same safe_stop_pips)
        risk_amount = self.balance * self.risk_per_trade
        size = risk_amount / (safe_stop_pips * self.pip_value * entry_price)

        # Apply leverage limit
        max_size = (self.balance * self.leverage) / entry_price
        size = min(size, max_size)

        # Commission - check balance sufficiency before deducting
        commission = size * entry_price * self.commission_rate
        if commission > self.balance:
            logger.warning(f"Insufficient balance for commission: {commission:.2f} > {self.balance:.2f}")
            return  # Skip opening this position
        self.balance -= commission

        # Create trade with stop loss and take profit
        trade = Trade(
            entry_time=timestamp,
            exit_time=None,
            entry_price=entry_price,
            exit_price=None,
            direction=direction,
            size=size,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price,
            commission=commission,
            slippage=slippage * size
        )

        self.positions.append(trade)

        # Notify risk manager of position opened
        notional = size * entry_price
        if self.risk_manager is not None:
            self.risk_manager.on_position_opened(notional)

    def _close_position(
        self,
        position: Trade,
        exit_price: float,
        timestamp: datetime,
        status: str = 'closed'
    ):
        """Close a position."""
        # Apply slippage
        slippage = self.slippage_pips * self.pip_value

        if position.direction == 'long':
            actual_exit = exit_price * (1 - self.spread_pips * self.pip_value / 2 - slippage)
            pnl = (actual_exit - position.entry_price) * position.size
        else:
            actual_exit = exit_price * (1 + self.spread_pips * self.pip_value / 2 + slippage)
            pnl = (position.entry_price - actual_exit) * position.size

        # Commission
        commission = position.size * actual_exit * self.commission_rate
        pnl -= commission

        # Update position
        position.exit_time = timestamp
        position.exit_price = actual_exit
        position.pnl = pnl
        position.pnl_pct = pnl / (position.entry_price * position.size)
        position.commission += commission
        position.status = status

        # Update balance
        self.balance += pnl

        # Move to closed trades
        self.positions.remove(position)
        self.closed_trades.append(position)

        # Notify risk manager of position closed (use entry notional for consistency)
        notional = position.size * position.entry_price
        if self.risk_manager is not None:
            self.risk_manager.on_position_closed(notional)

    def _close_all_positions(self, price: float, timestamp: datetime):
        """Close all open positions."""
        for position in list(self.positions):
            self._close_position(position, price, timestamp)

    def _close_positions_by_type(
        self,
        direction: str,
        price: float,
        timestamp: datetime
    ):
        """Close positions of a specific type."""
        for position in list(self.positions):
            if position.direction == direction:
                self._close_position(position, price, timestamp)

    def _update_positions(
        self,
        high: float,
        low: float,
        timestamp: datetime
    ):
        """Update positions, check for stop loss / take profit."""
        for position in list(self.positions):
            # Use stored stop loss and take profit levels from the trade
            stop_loss = position.stop_loss
            take_profit = position.take_profit

            # Skip if no stop loss or take profit set
            if stop_loss is None and take_profit is None:
                continue

            if position.direction == 'long':
                # Check if stopped out (price went below stop loss)
                if stop_loss is not None and low <= stop_loss:
                    self._close_position(position, stop_loss, timestamp, 'stopped')
                # Check if take profit hit (price went above take profit)
                elif take_profit is not None and high >= take_profit:
                    self._close_position(position, take_profit, timestamp, 'take_profit')

            else:  # short
                # Check if stopped out (price went above stop loss)
                if stop_loss is not None and high >= stop_loss:
                    self._close_position(position, stop_loss, timestamp, 'stopped')
                # Check if take profit hit (price went below take profit)
                elif take_profit is not None and low <= take_profit:
                    self._close_position(position, take_profit, timestamp, 'take_profit')

    def _update_equity(self, current_price: float):
        """Update equity with unrealized PnL."""
        unrealized_pnl = 0.0

        for position in self.positions:
            if position.direction == 'long':
                unrealized_pnl += (current_price - position.entry_price) * position.size
            else:
                unrealized_pnl += (position.entry_price - current_price) * position.size

        self.equity = self.balance + unrealized_pnl

    def _calculate_results(self) -> BacktestResult:
        """Calculate comprehensive backtest results."""
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]

        # Basic metrics
        total_return = (equity[-1] - equity[0]) / equity[0] if len(equity) > 1 else 0
        n_periods = len(equity)
        periods_per_year = 252 * 24  # Assuming hourly data
        annualized_return = (1 + total_return) ** (periods_per_year / max(1, n_periods)) - 1

        # Risk metrics
        volatility = np.std(returns) * np.sqrt(periods_per_year) if len(returns) > 1 else 0
        downside_returns = returns[returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(periods_per_year) if len(downside_returns) > 1 else 0.001

        # Sharpe and Sortino
        risk_free_rate = 0.02  # 2% annual
        excess_return = annualized_return - risk_free_rate
        sharpe = excess_return / volatility if volatility > 0 else 0
        sortino = excess_return / downside_vol if downside_vol > 0 else 0

        # Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        # Max drawdown duration
        dd_duration = 0
        max_dd_duration = 0
        for i in range(len(drawdown)):
            if drawdown[i] > 0:
                dd_duration += 1
                max_dd_duration = max(max_dd_duration, dd_duration)
            else:
                dd_duration = 0

        # Calmar ratio
        calmar = annualized_return / max_drawdown if max_drawdown > 0 else 0

        # Trade statistics
        trade_pnls = [t.pnl for t in self.closed_trades]
        winning_trades = [t for t in self.closed_trades if t.pnl > 0]
        losing_trades = [t for t in self.closed_trades if t.pnl <= 0]

        n_trades = len(self.closed_trades)
        n_winners = len(winning_trades)
        n_losers = len(losing_trades)
        win_rate = n_winners / n_trades if n_trades > 0 else 0

        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_trade = np.mean(trade_pnls) if trade_pnls else 0
        avg_winner = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loser = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

        largest_winner = max([t.pnl for t in winning_trades]) if winning_trades else 0
        largest_loser = min([t.pnl for t in losing_trades]) if losing_trades else 0

        # Average trade duration
        durations = []
        for t in self.closed_trades:
            if t.entry_time and t.exit_time:
                duration = (t.exit_time - t.entry_time).total_seconds() / 3600  # Hours
                durations.append(duration)
        avg_duration = np.mean(durations) if durations else 0

        # VaR and CVaR
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        cvar_95 = np.mean(returns[returns <= var_95]) if len(returns[returns <= var_95]) > 0 else 0

        # Higher moments
        skewness = float(pd.Series(returns).skew()) if len(returns) > 2 else 0
        kurtosis = float(pd.Series(returns).kurtosis()) if len(returns) > 3 else 0

        return BacktestResult(
            total_return=float(total_return),
            annualized_return=float(annualized_return),
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            calmar_ratio=float(calmar),
            max_drawdown=float(max_drawdown),
            max_drawdown_duration=int(max_dd_duration),
            total_trades=n_trades,
            winning_trades=n_winners,
            losing_trades=n_losers,
            win_rate=float(win_rate),
            profit_factor=float(profit_factor),
            avg_trade_return=float(avg_trade),
            avg_winner=float(avg_winner),
            avg_loser=float(avg_loser),
            largest_winner=float(largest_winner),
            largest_loser=float(largest_loser),
            avg_trade_duration=float(avg_duration),
            volatility=float(volatility),
            downside_volatility=float(downside_vol),
            var_95=float(var_95),
            cvar_95=float(cvar_95),
            skewness=float(skewness),
            kurtosis=float(kurtosis),
            equity_curve=equity,
            drawdown_curve=drawdown,
            returns=returns,
            trades=self.closed_trades,
            start_date=self.timestamps[0] if self.timestamps else None,
            end_date=self.timestamps[-1] if self.timestamps else None,
            initial_balance=self.initial_balance
        )


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization for robust strategy testing.

    Splits data into training and testing windows that roll forward through time,
    preventing look-ahead bias and testing out-of-sample performance.
    """

    def __init__(
        self,
        train_window: int = 180,  # Days
        test_window: int = 30,    # Days
        step_size: int = 30,      # Days to step forward
        n_splits: Optional[int] = None
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.n_splits = n_splits

    def generate_splits(
        self,
        data: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate train/test splits for walk-forward analysis."""
        n_bars = len(data)

        # Estimate bars per day (assuming hourly data)
        bars_per_day = 24

        train_bars = self.train_window * bars_per_day
        test_bars = self.test_window * bars_per_day
        step_bars = self.step_size * bars_per_day

        splits = []
        start_idx = 0

        while start_idx + train_bars + test_bars <= n_bars:
            train_end = start_idx + train_bars
            test_end = train_end + test_bars

            train_data = data.iloc[start_idx:train_end]
            test_data = data.iloc[train_end:test_end]

            splits.append((train_data, test_data))

            start_idx += step_bars

            if self.n_splits and len(splits) >= self.n_splits:
                break

        logger.info(f"Generated {len(splits)} walk-forward splits")

        return splits

    def run(
        self,
        data: pd.DataFrame,
        train_func: Callable,
        backtest_func: Callable,
        parallel: bool = True
    ) -> Dict:
        """
        Run walk-forward optimization.

        Args:
            data: Full dataset
            train_func: Function to train model on training data
            backtest_func: Function to backtest on test data

        Returns:
            Dictionary with aggregated results
        """
        splits = self.generate_splits(data)

        if len(splits) == 0:
            logger.warning("No valid splits generated")
            return {}

        results = []

        if parallel:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for i, (train_data, test_data) in enumerate(splits):
                    future = executor.submit(
                        self._run_single_fold,
                        i, train_data, test_data, train_func, backtest_func
                    )
                    futures.append(future)

                for future in futures:
                    result = future.result()
                    if result:
                        results.append(result)
        else:
            for i, (train_data, test_data) in enumerate(splits):
                result = self._run_single_fold(
                    i, train_data, test_data, train_func, backtest_func
                )
                if result:
                    results.append(result)

        # Aggregate results
        return self._aggregate_results(results)

    def _run_single_fold(
        self,
        fold_idx: int,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        train_func: Callable,
        backtest_func: Callable
    ) -> Optional[Dict]:
        """Run a single walk-forward fold."""
        try:
            logger.info(f"Running fold {fold_idx + 1}...")

            # Train on training data
            model = train_func(train_data)

            # Backtest on test data
            result = backtest_func(test_data, model)

            return {
                'fold': fold_idx,
                'train_start': train_data.index[0] if hasattr(train_data.index[0], 'strftime') else 0,
                'train_end': train_data.index[-1] if hasattr(train_data.index[-1], 'strftime') else len(train_data),
                'test_start': test_data.index[0] if hasattr(test_data.index[0], 'strftime') else 0,
                'test_end': test_data.index[-1] if hasattr(test_data.index[-1], 'strftime') else len(test_data),
                'result': result
            }

        except Exception:
            logger.exception(f"Error in fold {fold_idx}")
            return None

    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate results from all folds."""
        if not results:
            return {}

        # Extract metrics
        returns = [r['result'].total_return for r in results]
        sharpes = [r['result'].sharpe_ratio for r in results]
        drawdowns = [r['result'].max_drawdown for r in results]
        win_rates = [r['result'].win_rate for r in results]

        return {
            'n_folds': len(results),
            'aggregate': {
                'total_return': {
                    'mean': float(np.mean(returns)),
                    'std': float(np.std(returns)),
                    'min': float(np.min(returns)),
                    'max': float(np.max(returns))
                },
                'sharpe_ratio': {
                    'mean': float(np.mean(sharpes)),
                    'std': float(np.std(sharpes)),
                    'min': float(np.min(sharpes)),
                    'max': float(np.max(sharpes))
                },
                'max_drawdown': {
                    'mean': float(np.mean(drawdowns)),
                    'std': float(np.std(drawdowns)),
                    'min': float(np.min(drawdowns)),
                    'max': float(np.max(drawdowns))
                },
                'win_rate': {
                    'mean': float(np.mean(win_rates)),
                    'std': float(np.std(win_rates)),
                    'min': float(np.min(win_rates)),
                    'max': float(np.max(win_rates))
                }
            },
            'folds': results,
            'consistency': {
                'profitable_folds': sum(1 for r in returns if r > 0),
                'profitable_ratio': sum(1 for r in returns if r > 0) / len(returns)
            }
        }


class MonteCarloSimulator:
    """
    Monte Carlo simulation for risk analysis.
    """

    def __init__(self, n_simulations: int = 1000, confidence_level: float = 0.95):
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level

    def simulate_from_trades(self, trades: List[Trade]) -> Dict:
        """
        Run Monte Carlo simulation by resampling trades.
        """
        if len(trades) < 10:
            logger.warning("Not enough trades for Monte Carlo simulation")
            return {}

        trade_returns = [t.pnl_pct for t in trades]

        # Bootstrap resampling
        simulated_equity = []

        for _ in range(self.n_simulations):
            # Resample trades with replacement
            sampled_returns = np.random.choice(trade_returns, size=len(trades), replace=True)

            # Calculate equity curve
            equity = [1.0]
            for ret in sampled_returns:
                equity.append(equity[-1] * (1 + ret))

            simulated_equity.append(equity)

        simulated_equity = np.array(simulated_equity)

        # Calculate statistics
        final_values = simulated_equity[:, -1]
        max_drawdowns = []

        for sim in simulated_equity:
            peak = np.maximum.accumulate(sim)
            dd = (peak - sim) / peak
            max_drawdowns.append(np.max(dd))

        return {
            'final_equity': {
                'mean': float(np.mean(final_values)),
                'std': float(np.std(final_values)),
                'median': float(np.median(final_values)),
                'percentile_5': float(np.percentile(final_values, 5)),
                'percentile_95': float(np.percentile(final_values, 95))
            },
            'max_drawdown': {
                'mean': float(np.mean(max_drawdowns)),
                'std': float(np.std(max_drawdowns)),
                'percentile_95': float(np.percentile(max_drawdowns, 95))
            },
            'probability_of_profit': float(np.mean(final_values > 1.0)),
            'probability_of_ruin': float(np.mean(final_values < 0.5))  # 50% loss
        }
