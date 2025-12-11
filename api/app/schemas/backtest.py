"""Backtest-related schemas."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class BacktestConfig(BaseModel):
    """Backtest configuration."""

    initial_balance: float = Field(10000, alias="initialBalance", gt=0)
    leverage: int = Field(100, ge=1, le=500)
    spread_pips: float = Field(1.5, alias="spreadPips", ge=0)
    commission_per_lot: float = Field(7.0, alias="commissionPerLot", ge=0)
    slippage_pips: float = Field(1.0, alias="slippagePips", ge=0)
    risk_per_trade: float = Field(0.02, alias="riskPerTrade", gt=0, le=0.1)
    n_simulations: int = Field(1000, alias="nSimulations", ge=100, le=10000)


class BacktestRunRequest(BaseModel):
    """Request to run backtest."""

    symbol: str = "EURUSD"
    timeframe: str = "1h"
    bars: int = Field(50000, ge=1000, le=500000)
    model_dir: str = Field("./saved_models", alias="modelDir")
    realistic_mode: bool = Field(False, alias="realisticMode")
    monte_carlo: bool = Field(False, alias="monteCarlo")
    config: Optional[BacktestConfig] = None


class ReturnMetrics(BaseModel):
    """Return metrics."""

    total_return: float = Field(alias="totalReturn")
    annualized_return: float = Field(alias="annualizedReturn")
    cagr: float


class RiskMetrics(BaseModel):
    """Risk metrics."""

    volatility: float
    downside_volatility: float = Field(alias="downsideVolatility")
    max_drawdown: float = Field(alias="maxDrawdown")
    max_drawdown_duration: Optional[int] = Field(None, alias="maxDrawdownDuration")
    var_95: float = Field(alias="var95")
    cvar_95: float = Field(alias="cvar95")


class RiskAdjustedMetrics(BaseModel):
    """Risk-adjusted metrics."""

    sharpe_ratio: float = Field(alias="sharpeRatio")
    sortino_ratio: float = Field(alias="sortinoRatio")
    calmar_ratio: float = Field(alias="calmarRatio")
    omega_ratio: Optional[float] = Field(None, alias="omegaRatio")


class TradeMetrics(BaseModel):
    """Trade metrics."""

    total_trades: int = Field(alias="totalTrades")
    winning_trades: int = Field(alias="winningTrades")
    losing_trades: int = Field(alias="losingTrades")
    win_rate: float = Field(alias="winRate")
    profit_factor: float = Field(alias="profitFactor")
    avg_trade: float = Field(alias="avgTrade")
    avg_winner: float = Field(alias="avgWinner")
    avg_loser: float = Field(alias="avgLoser")
    largest_winner: float = Field(alias="largestWinner")
    largest_loser: float = Field(alias="largestLoser")


class DistributionMetrics(BaseModel):
    """Distribution metrics."""

    skewness: float
    kurtosis: float
    tail_ratio: Optional[float] = Field(None, alias="tailRatio")


class BacktestMetrics(BaseModel):
    """Complete backtest metrics."""

    returns: ReturnMetrics
    risk: RiskMetrics
    risk_adjusted: RiskAdjustedMetrics = Field(alias="riskAdjusted")
    trade: TradeMetrics
    distribution: DistributionMetrics


class Trade(BaseModel):
    """Individual trade."""

    id: str
    entry_time: str = Field(alias="entryTime")
    exit_time: str = Field(alias="exitTime")
    direction: str
    entry_price: float = Field(alias="entryPrice")
    exit_price: float = Field(alias="exitPrice")
    size: float
    pnl: float
    pnl_percent: float = Field(alias="pnlPercent")
    status: str


class TimeSeries(BaseModel):
    """Time series data."""

    equity_curve: list[float] = Field(alias="equityCurve")
    drawdown_curve: list[float] = Field(alias="drawdownCurve")
    timestamps: list[str]


class MonteCarloResults(BaseModel):
    """Monte Carlo simulation results."""

    final_equity: dict[str, float] = Field(alias="finalEquity")
    max_drawdown: dict[str, float] = Field(alias="maxDrawdown")
    probability_of_profit: float = Field(alias="probabilityOfProfit")
    probability_of_ruin: float = Field(alias="probabilityOfRuin")


class BacktestResultSummary(BaseModel):
    """Backtest result summary."""

    total_return: float = Field(alias="totalReturn")
    sharpe_ratio: float = Field(alias="sharpeRatio")
    max_drawdown: float = Field(alias="maxDrawdown")
    win_rate: float = Field(alias="winRate")
    total_trades: int = Field(alias="totalTrades")


class BacktestResultData(BaseModel):
    """Full backtest result data."""

    result_id: str = Field(alias="resultId")
    symbol: str
    timeframe: str
    config: Optional[dict[str, Any]] = None
    metrics: Optional[BacktestMetrics] = None
    time_series: Optional[TimeSeries] = Field(None, alias="timeSeries")
    trades: Optional[list[Trade]] = None
    monte_carlo: Optional[MonteCarloResults] = Field(None, alias="monteCarlo")
    completed_at: str = Field(alias="completedAt")


class BacktestResultResponse(BaseModel):
    """Backtest result response."""

    data: BacktestResultData


class BacktestResultSummaryData(BaseModel):
    """Backtest result summary for list."""

    result_id: str = Field(alias="resultId")
    symbol: str
    timeframe: str
    summary: BacktestResultSummary
    completed_at: str = Field(alias="completedAt")


class BacktestResultListData(BaseModel):
    """Backtest result list data."""

    results: list[BacktestResultSummaryData]
    total: int


class BacktestResultListResponse(BaseModel):
    """Backtest result list response."""

    data: BacktestResultListData


class BacktestCompareRequest(BaseModel):
    """Request to compare backtests."""

    result_ids: list[str] = Field(alias="resultIds")


class BacktestComparison(BaseModel):
    """Comparison summary."""

    best_by_return: str = Field(alias="bestByReturn")
    best_by_sharpe: str = Field(alias="bestBySharpe")
    best_by_drawdown: str = Field(alias="bestByDrawdown")


class BacktestCompareData(BaseModel):
    """Backtest comparison data."""

    results: list[BacktestResultData]
    comparison: BacktestComparison


class BacktestCompareResponse(BaseModel):
    """Backtest comparison response."""

    data: BacktestCompareData
