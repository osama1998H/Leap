"""
Leap Trading System - Performance Metrics and Analysis
Comprehensive metrics calculation and performance reporting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

from .backtester import MonteCarloSimulator

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculate trading performance metrics.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252 * 24  # Hourly data
    ):
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def calculate_all(
        self,
        equity_curve: np.ndarray,
        trades: Optional[List] = None,
        benchmark: Optional[np.ndarray] = None
    ) -> Dict:
        """Calculate all metrics."""
        returns = self._calculate_returns(equity_curve)

        metrics = {
            # Return metrics
            'total_return': self.total_return(equity_curve),
            'annualized_return': self.annualized_return(equity_curve),
            'cagr': self.cagr(equity_curve),

            # Risk metrics
            'volatility': self.volatility(returns),
            'downside_volatility': self.downside_volatility(returns),
            'max_drawdown': self.max_drawdown(equity_curve),
            'avg_drawdown': self.average_drawdown(equity_curve),
            'max_drawdown_duration': self.max_drawdown_duration(equity_curve),

            # Risk-adjusted returns
            'sharpe_ratio': self.sharpe_ratio(returns),
            'sortino_ratio': self.sortino_ratio(returns),
            'calmar_ratio': self.calmar_ratio(equity_curve),
            'omega_ratio': self.omega_ratio(returns),
            'information_ratio': self.information_ratio(returns, benchmark) if benchmark is not None else None,

            # Distribution metrics
            'skewness': self.skewness(returns),
            'kurtosis': self.kurtosis(returns),
            'var_95': self.value_at_risk(returns, 0.95),
            'var_99': self.value_at_risk(returns, 0.99),
            'cvar_95': self.conditional_var(returns, 0.95),

            # Tail risk
            'tail_ratio': self.tail_ratio(returns),
            'gain_to_pain_ratio': self.gain_to_pain_ratio(returns),
        }

        # Trade metrics
        if trades:
            metrics.update(self.trade_metrics(trades))

        return metrics

    def _calculate_returns(self, equity_curve: np.ndarray) -> np.ndarray:
        """Calculate returns from equity curve."""
        return np.diff(equity_curve) / equity_curve[:-1]

    # Return Metrics
    def total_return(self, equity_curve: np.ndarray) -> float:
        """Calculate total return."""
        if len(equity_curve) < 2:
            return 0.0
        return (equity_curve[-1] - equity_curve[0]) / equity_curve[0]

    def annualized_return(self, equity_curve: np.ndarray) -> float:
        """Calculate annualized return."""
        total = self.total_return(equity_curve)
        n_periods = len(equity_curve)
        if n_periods < 2:
            return 0.0
        return (1 + total) ** (self.periods_per_year / n_periods) - 1

    def cagr(self, equity_curve: np.ndarray) -> float:
        """Calculate Compound Annual Growth Rate."""
        return self.annualized_return(equity_curve)

    # Risk Metrics
    def volatility(self, returns: np.ndarray) -> float:
        """Calculate annualized volatility."""
        if len(returns) < 2:
            return 0.0
        return np.std(returns) * np.sqrt(self.periods_per_year)

    def downside_volatility(self, returns: np.ndarray, target: float = 0.0) -> float:
        """Calculate downside volatility (semi-deviation)."""
        downside = returns[returns < target]
        if len(downside) < 2:
            return 0.0
        return np.std(downside) * np.sqrt(self.periods_per_year)

    def max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(equity_curve) < 2:
            return 0.0
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        return np.max(drawdown)

    def average_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate average drawdown."""
        if len(equity_curve) < 2:
            return 0.0
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        # Only count periods in drawdown
        in_drawdown = drawdown[drawdown > 0]
        return np.mean(in_drawdown) if len(in_drawdown) > 0 else 0.0

    def max_drawdown_duration(self, equity_curve: np.ndarray) -> int:
        """Calculate maximum drawdown duration in periods."""
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak

        max_duration = 0
        current_duration = 0

        for dd in drawdown:
            if dd > 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration

    # Risk-Adjusted Return Metrics
    def sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - self.risk_free_rate / self.periods_per_year
        vol = np.std(returns)

        if vol == 0:
            return 0.0

        return np.mean(excess_returns) / vol * np.sqrt(self.periods_per_year)

    def sortino_ratio(self, returns: np.ndarray, target: float = 0.0) -> float:
        """Calculate Sortino ratio."""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - self.risk_free_rate / self.periods_per_year
        downside_vol = self.downside_volatility(returns, target) / np.sqrt(self.periods_per_year)

        if downside_vol == 0:
            return 0.0

        return np.mean(excess_returns) * np.sqrt(self.periods_per_year) / (downside_vol * np.sqrt(self.periods_per_year))

    def calmar_ratio(self, equity_curve: np.ndarray) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        annual_return = self.annualized_return(equity_curve)
        max_dd = self.max_drawdown(equity_curve)

        if max_dd == 0:
            return 0.0

        return annual_return / max_dd

    def omega_ratio(self, returns: np.ndarray, threshold: float = 0.0) -> float:
        """Calculate Omega ratio."""
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]

        if len(losses) == 0 or np.sum(losses) == 0:
            return float('inf') if len(gains) > 0 else 0.0

        return np.sum(gains) / np.sum(losses)

    def information_ratio(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> float:
        """Calculate Information ratio."""
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 0.0

        active_returns = returns - benchmark_returns
        tracking_error = np.std(active_returns) * np.sqrt(self.periods_per_year)

        if tracking_error == 0:
            return 0.0

        return np.mean(active_returns) * self.periods_per_year / tracking_error

    # Distribution Metrics
    def skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns."""
        if len(returns) < 3:
            return 0.0
        return float(pd.Series(returns).skew())

    def kurtosis(self, returns: np.ndarray) -> float:
        """Calculate excess kurtosis of returns."""
        if len(returns) < 4:
            return 0.0
        return float(pd.Series(returns).kurtosis())

    def value_at_risk(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Value at Risk at given confidence level."""
        if len(returns) < 2:
            return 0.0
        return np.percentile(returns, (1 - confidence) * 100)

    def conditional_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        var = self.value_at_risk(returns, confidence)
        tail_losses = returns[returns <= var]

        if len(tail_losses) == 0:
            return var

        return np.mean(tail_losses)

    # Tail Risk Metrics
    def tail_ratio(self, returns: np.ndarray) -> float:
        """Calculate tail ratio (right tail / left tail)."""
        if len(returns) < 20:
            return 1.0

        right_tail = np.percentile(returns, 95)
        left_tail = abs(np.percentile(returns, 5))

        if left_tail == 0:
            return float('inf') if right_tail > 0 else 1.0

        return right_tail / left_tail

    def gain_to_pain_ratio(self, returns: np.ndarray) -> float:
        """Calculate gain to pain ratio."""
        gains = np.sum(returns[returns > 0])
        losses = abs(np.sum(returns[returns < 0]))

        if losses == 0:
            return float('inf') if gains > 0 else 0.0

        return gains / losses

    # Trade Metrics
    def trade_metrics(self, trades: List) -> Dict:
        """Calculate trade-level metrics."""
        if not trades:
            return {}

        pnls = [t.pnl for t in trades]
        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]

        return {
            'total_trades': len(trades),
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'win_rate': len(winners) / len(trades) if trades else 0,

            'avg_trade': np.mean(pnls) if pnls else 0,
            'avg_winner': np.mean([t.pnl for t in winners]) if winners else 0,
            'avg_loser': np.mean([t.pnl for t in losers]) if losers else 0,

            'largest_winner': max([t.pnl for t in winners]) if winners else 0,
            'largest_loser': min([t.pnl for t in losers]) if losers else 0,

            'profit_factor': self._profit_factor(trades),
            'payoff_ratio': self._payoff_ratio(trades),
            'expectancy': self._expectancy(trades),

            'consecutive_winners': self._max_consecutive(trades, 'win'),
            'consecutive_losers': self._max_consecutive(trades, 'loss'),
        }

    def _profit_factor(self, trades: List) -> float:
        """Calculate profit factor."""
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def _payoff_ratio(self, trades: List) -> float:
        """Calculate payoff ratio (avg win / avg loss)."""
        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl < 0]

        if not losers or not winners:
            return 0.0

        avg_win = np.mean([t.pnl for t in winners])
        avg_loss = abs(np.mean([t.pnl for t in losers]))

        if avg_loss == 0:
            return float('inf')

        return avg_win / avg_loss

    def _expectancy(self, trades: List) -> float:
        """Calculate expectancy per trade."""
        if not trades:
            return 0.0

        return np.mean([t.pnl for t in trades])

    def _max_consecutive(self, trades: List, type: str) -> int:
        """Calculate maximum consecutive wins or losses."""
        if not trades:
            return 0

        max_streak = 0
        current_streak = 0

        for t in trades:
            if (type == 'win' and t.pnl > 0) or (type == 'loss' and t.pnl <= 0):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak


class PerformanceAnalyzer:
    """
    High-level performance analysis and reporting.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.metrics_calculator = MetricsCalculator(
            risk_free_rate=self.config.get('risk_free_rate', 0.02),
            periods_per_year=self.config.get('periods_per_year', 252 * 24)
        )
        # Monte Carlo configuration
        self.enable_monte_carlo = self.config.get('enable_monte_carlo', False)
        self.n_simulations = self.config.get('n_simulations', 1000)
        self.confidence_level = self.config.get('confidence_level', 0.95)

    def analyze(
        self,
        backtest_result,
        benchmark_data: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Perform comprehensive performance analysis.
        """
        equity = backtest_result.equity_curve
        trades = backtest_result.trades

        # Calculate all metrics
        metrics = self.metrics_calculator.calculate_all(
            equity_curve=equity,
            trades=trades,
            benchmark=benchmark_data
        )

        # Add statistical tests
        metrics['statistical_tests'] = self._run_statistical_tests(backtest_result)

        # Add regime analysis
        metrics['regime_analysis'] = self._analyze_regimes(backtest_result)

        # Add Monte Carlo analysis if enabled
        if self.enable_monte_carlo:
            metrics['monte_carlo'] = self._run_monte_carlo(trades)

        # Generate summary
        metrics['summary'] = self._generate_summary(metrics)

        return metrics

    def _run_monte_carlo(self, trades: List) -> Dict:
        """Run Monte Carlo simulation on trades."""
        simulator = MonteCarloSimulator(
            n_simulations=self.n_simulations,
            confidence_level=self.confidence_level
        )
        result = simulator.simulate_from_trades(trades)

        if not result:
            return {'warning': 'Insufficient trades for Monte Carlo simulation (minimum 10 required)'}

        return result

    def _run_statistical_tests(self, result) -> Dict:
        """Run statistical significance tests."""
        returns = result.returns

        if len(returns) < 30:
            return {'warning': 'Insufficient data for statistical tests'}

        from scipy import stats

        tests = {}

        # Test if returns are significantly different from zero
        t_stat, p_value = stats.ttest_1samp(returns, 0)
        tests['returns_significance'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_5pct': p_value < 0.05,
            'significant_1pct': p_value < 0.01
        }

        # Normality test
        if len(returns) >= 20:
            stat, p = stats.shapiro(returns[:5000])  # Limit for Shapiro
            tests['normality'] = {
                'statistic': float(stat),
                'p_value': float(p),
                'is_normal': p > 0.05
            }

        # Runs test for randomness
        median = np.median(returns)
        runs = np.sum(np.diff(returns > median) != 0) + 1
        n_pos = np.sum(returns > median)
        n_neg = len(returns) - n_pos

        if n_pos > 0 and n_neg > 0:
            expected_runs = 1 + (2 * n_pos * n_neg) / len(returns)
            std_runs = np.sqrt((2 * n_pos * n_neg * (2 * n_pos * n_neg - len(returns))) /
                               (len(returns) ** 2 * (len(returns) - 1)))

            if std_runs > 0:
                z_stat = (runs - expected_runs) / std_runs
                tests['runs_test'] = {
                    'runs': int(runs),
                    'expected': float(expected_runs),
                    'z_statistic': float(z_stat),
                    'is_random': abs(z_stat) < 1.96
                }

        return tests

    def _analyze_regimes(self, result) -> Dict:
        """Analyze performance across different market regimes."""
        returns = result.returns
        equity = result.equity_curve

        if len(returns) < 50:
            return {}

        # Simple regime detection based on volatility
        rolling_vol = pd.Series(returns).rolling(20).std()

        high_vol_threshold = rolling_vol.quantile(0.75)
        low_vol_threshold = rolling_vol.quantile(0.25)

        high_vol_periods = rolling_vol > high_vol_threshold
        low_vol_periods = rolling_vol < low_vol_threshold
        normal_periods = ~high_vol_periods & ~low_vol_periods

        regimes = {
            'high_volatility': {
                'periods': int(high_vol_periods.sum()),
                'avg_return': float(returns[high_vol_periods.values].mean()) if high_vol_periods.sum() > 0 else 0,
                'volatility': float(returns[high_vol_periods.values].std()) if high_vol_periods.sum() > 0 else 0
            },
            'low_volatility': {
                'periods': int(low_vol_periods.sum()),
                'avg_return': float(returns[low_vol_periods.values].mean()) if low_vol_periods.sum() > 0 else 0,
                'volatility': float(returns[low_vol_periods.values].std()) if low_vol_periods.sum() > 0 else 0
            },
            'normal': {
                'periods': int(normal_periods.sum()),
                'avg_return': float(returns[normal_periods.values].mean()) if normal_periods.sum() > 0 else 0,
                'volatility': float(returns[normal_periods.values].std()) if normal_periods.sum() > 0 else 0
            }
        }

        return regimes

    def _generate_summary(self, metrics: Dict) -> Dict:
        """Generate human-readable summary."""
        summary = {
            'overall_rating': self._calculate_rating(metrics),
            'key_strengths': [],
            'key_weaknesses': [],
            'recommendations': []
        }

        # Analyze strengths
        if metrics.get('sharpe_ratio', 0) > 1.5:
            summary['key_strengths'].append('Excellent risk-adjusted returns (Sharpe > 1.5)')
        elif metrics.get('sharpe_ratio', 0) > 1.0:
            summary['key_strengths'].append('Good risk-adjusted returns (Sharpe > 1.0)')

        if metrics.get('win_rate', 0) > 0.55:
            summary['key_strengths'].append(f"High win rate ({metrics.get('win_rate', 0):.1%})")

        if metrics.get('max_drawdown', 1) < 0.1:
            summary['key_strengths'].append('Low maximum drawdown (<10%)')

        if metrics.get('profit_factor', 0) > 2.0:
            summary['key_strengths'].append('Excellent profit factor (>2.0)')

        # Analyze weaknesses
        if metrics.get('max_drawdown', 0) > 0.2:
            summary['key_weaknesses'].append(f"High drawdown ({metrics.get('max_drawdown', 0):.1%})")
            summary['recommendations'].append('Consider tighter risk management or position sizing')

        if metrics.get('sharpe_ratio', 0) < 0.5:
            summary['key_weaknesses'].append('Poor risk-adjusted returns')
            summary['recommendations'].append('Review strategy logic or market conditions')

        if metrics.get('skewness', 0) < -0.5:
            summary['key_weaknesses'].append('Negative skew in returns (tail risk)')
            summary['recommendations'].append('Implement better stop-loss mechanisms')

        return summary

    def _calculate_rating(self, metrics: Dict) -> str:
        """Calculate overall strategy rating."""
        score = 0

        # Sharpe contribution
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe > 2.0:
            score += 3
        elif sharpe > 1.0:
            score += 2
        elif sharpe > 0.5:
            score += 1

        # Drawdown contribution
        dd = metrics.get('max_drawdown', 1)
        if dd < 0.1:
            score += 2
        elif dd < 0.2:
            score += 1
        elif dd > 0.3:
            score -= 1

        # Win rate contribution
        wr = metrics.get('win_rate', 0)
        if wr > 0.6:
            score += 1
        elif wr < 0.4:
            score -= 1

        # Profit factor
        pf = metrics.get('profit_factor', 0)
        if pf > 2.0:
            score += 1
        elif pf < 1.0:
            score -= 2

        # Rating
        if score >= 5:
            return 'Excellent'
        elif score >= 3:
            return 'Good'
        elif score >= 1:
            return 'Average'
        elif score >= -1:
            return 'Below Average'
        else:
            return 'Poor'

    def generate_report(self, analysis: Dict, output_format: str = 'text') -> str:
        """Generate formatted performance report."""
        if output_format == 'text':
            return self._text_report(analysis)
        elif output_format == 'html':
            return self._html_report(analysis)
        else:
            return str(analysis)

    def _text_report(self, analysis: Dict) -> str:
        """Generate text-based report."""
        lines = [
            "=" * 60,
            "TRADING STRATEGY PERFORMANCE REPORT",
            "=" * 60,
            "",
            f"Overall Rating: {analysis.get('summary', {}).get('overall_rating', 'N/A')}",
            "",
            "--- Return Metrics ---",
            f"Total Return: {analysis.get('total_return', 0):.2%}",
            f"Annualized Return: {analysis.get('annualized_return', 0):.2%}",
            "",
            "--- Risk Metrics ---",
            f"Volatility: {analysis.get('volatility', 0):.2%}",
            f"Max Drawdown: {analysis.get('max_drawdown', 0):.2%}",
            f"VaR (95%): {analysis.get('var_95', 0):.4f}",
            "",
            "--- Risk-Adjusted Metrics ---",
            f"Sharpe Ratio: {analysis.get('sharpe_ratio', 0):.2f}",
            f"Sortino Ratio: {analysis.get('sortino_ratio', 0):.2f}",
            f"Calmar Ratio: {analysis.get('calmar_ratio', 0):.2f}",
            "",
            "--- Trade Statistics ---",
            f"Total Trades: {analysis.get('total_trades', 0)}",
            f"Win Rate: {analysis.get('win_rate', 0):.2%}",
            f"Profit Factor: {analysis.get('profit_factor', 0):.2f}",
            f"Avg Trade: ${analysis.get('avg_trade', 0):.2f}",
            "",
            "--- Key Insights ---"
        ]

        summary = analysis.get('summary', {})
        for strength in summary.get('key_strengths', []):
            lines.append(f"  ✓ {strength}")
        for weakness in summary.get('key_weaknesses', []):
            lines.append(f"  ✗ {weakness}")

        lines.extend([
            "",
            "--- Recommendations ---"
        ])
        for rec in summary.get('recommendations', []):
            lines.append(f"  • {rec}")

        # Monte Carlo results (if available)
        if 'monte_carlo' in analysis and 'warning' not in analysis['monte_carlo']:
            mc = analysis['monte_carlo']
            lines.extend([
                "",
                "--- Monte Carlo Analysis ---",
                f"Simulations: {self.n_simulations}",
                f"Final Equity (mean): {mc['final_equity']['mean']:.2%}",
                f"Final Equity (5th-95th): {mc['final_equity']['percentile_5']:.2%} - {mc['final_equity']['percentile_95']:.2%}",
                f"Max Drawdown (mean): {mc['max_drawdown']['mean']:.2%}",
                f"Max Drawdown (95th): {mc['max_drawdown']['percentile_95']:.2%}",
                f"Probability of Profit: {mc['probability_of_profit']:.1%}",
                f"Probability of Ruin: {mc['probability_of_ruin']:.1%}",
            ])
        elif 'monte_carlo' in analysis and 'warning' in analysis['monte_carlo']:
            lines.extend([
                "",
                "--- Monte Carlo Analysis ---",
                f"Warning: {analysis['monte_carlo']['warning']}"
            ])

        lines.append("=" * 60)

        return "\n".join(lines)

    def _html_report(self, analysis: Dict) -> str:
        """Generate HTML report."""
        # Simplified HTML report
        return f"""
        <html>
        <head><title>Performance Report</title></head>
        <body>
        <h1>Trading Strategy Performance Report</h1>
        <h2>Overall Rating: {analysis.get('summary', {}).get('overall_rating', 'N/A')}</h2>
        <h3>Key Metrics</h3>
        <ul>
            <li>Total Return: {analysis.get('total_return', 0):.2%}</li>
            <li>Sharpe Ratio: {analysis.get('sharpe_ratio', 0):.2f}</li>
            <li>Max Drawdown: {analysis.get('max_drawdown', 0):.2%}</li>
            <li>Win Rate: {analysis.get('win_rate', 0):.2%}</li>
        </ul>
        </body>
        </html>
        """
