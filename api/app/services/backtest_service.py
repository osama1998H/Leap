"""Backtest service."""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..core.config import settings
from ..core.job_manager import Job, JobStatus, JobType, job_manager
from ..schemas.backtest import (
    BacktestCompareData,
    BacktestComparison,
    BacktestMetrics,
    BacktestResultData,
    BacktestResultSummary,
    BacktestResultSummaryData,
    BacktestRunRequest,
    DistributionMetrics,
    ReturnMetrics,
    RiskAdjustedMetrics,
    RiskMetrics,
    Trade,
    TradeMetrics,
    TimeSeries,
)

logger = logging.getLogger(__name__)


class BacktestService:
    """Service for managing backtests."""

    def __init__(self):
        self.job_manager = job_manager
        self.results: dict[str, BacktestResultData] = {}

    async def run_backtest(self, request: BacktestRunRequest) -> dict[str, Any]:
        """Run a new backtest."""
        config = {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "bars": request.bars,
            "modelDir": request.model_dir,
            "realisticMode": request.realistic_mode,
            "monteCarlo": request.monte_carlo,
        }

        if request.config:
            config.update(request.config.model_dump(by_alias=True))

        job = self.job_manager.create_job(JobType.BACKTEST, config)
        await self.job_manager.start_job(job.job_id)

        return {
            "jobId": job.job_id,
            "status": job.status.value,
            "createdAt": job.created_at.isoformat() + "Z",
        }

    async def list_results(
        self,
        symbol: Optional[str] = None,
        min_sharpe: Optional[float] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[BacktestResultSummaryData], int]:
        """List backtest results."""
        # Get completed backtest jobs
        jobs, _ = self.job_manager.list_jobs(
            job_type=JobType.BACKTEST,
            status=JobStatus.COMPLETED,
            limit=1000,  # Get all for filtering
            offset=0,
        )

        # Filter by symbol
        if symbol:
            jobs = [j for j in jobs if j.config.get("symbol") == symbol]

        results = []
        for job in jobs:
            result = self._job_to_summary(job)
            if result:
                if min_sharpe and result.summary.sharpe_ratio < min_sharpe:
                    continue
                results.append(result)

        # Add any stored results
        for result_id, result in self.results.items():
            summary = BacktestResultSummaryData(
                resultId=result.result_id,
                symbol=result.symbol,
                timeframe=result.timeframe,
                summary=BacktestResultSummary(
                    totalReturn=result.metrics.returns.total_return if result.metrics else 0,
                    sharpeRatio=result.metrics.risk_adjusted.sharpe_ratio if result.metrics else 0,
                    maxDrawdown=result.metrics.risk.max_drawdown if result.metrics else 0,
                    winRate=result.metrics.trade.win_rate if result.metrics else 0,
                    totalTrades=result.metrics.trade.total_trades if result.metrics else 0,
                ),
                completedAt=result.completed_at,
            )
            if min_sharpe and summary.summary.sharpe_ratio < min_sharpe:
                continue
            if symbol and result.symbol != symbol:
                continue
            results.append(summary)

        total = len(results)
        results = results[offset : offset + limit]

        return results, total

    async def get_result(self, result_id: str) -> Optional[BacktestResultData]:
        """Get a specific backtest result."""
        # Check stored results
        if result_id in self.results:
            return self.results[result_id]

        # Check jobs
        job = self.job_manager.get_job(result_id)
        if job and job.job_type == JobType.BACKTEST and job.status == JobStatus.COMPLETED:
            return self._job_to_result(job)

        return None

    async def compare_results(
        self, result_ids: list[str]
    ) -> BacktestCompareData:
        """Compare multiple backtest results."""
        results = []
        for result_id in result_ids:
            result = await self.get_result(result_id)
            if result:
                results.append(result)

        if not results:
            return BacktestCompareData(
                results=[],
                comparison=BacktestComparison(
                    bestByReturn="",
                    bestBySharpe="",
                    bestByDrawdown="",
                ),
            )

        # Find best by each metric
        best_by_return = max(
            results, key=lambda r: r.metrics.returns.total_return if r.metrics else 0
        )
        best_by_sharpe = max(
            results, key=lambda r: r.metrics.risk_adjusted.sharpe_ratio if r.metrics else 0
        )
        best_by_drawdown = max(
            results, key=lambda r: r.metrics.risk.max_drawdown if r.metrics else -1
        )

        return BacktestCompareData(
            results=results,
            comparison=BacktestComparison(
                bestByReturn=best_by_return.result_id,
                bestBySharpe=best_by_sharpe.result_id,
                bestByDrawdown=best_by_drawdown.result_id,
            ),
        )

    def _job_to_summary(self, job: Job) -> Optional[BacktestResultSummaryData]:
        """Convert job to result summary.

        Note: Currently returns placeholder metrics. In production, this should
        parse actual results from job.result or job.output_lines.
        """
        if job.status != JobStatus.COMPLETED:
            return None

        # Try to parse actual metrics from job result/output
        metrics = self._parse_metrics_from_job(job)

        return BacktestResultSummaryData(
            resultId=job.job_id,
            symbol=job.config.get("symbol", "EURUSD"),
            timeframe=job.config.get("timeframe", "1h"),
            summary=BacktestResultSummary(
                totalReturn=metrics.get("total_return", 0.0),
                sharpeRatio=metrics.get("sharpe_ratio", 0.0),
                maxDrawdown=metrics.get("max_drawdown", 0.0),
                winRate=metrics.get("win_rate", 0.0),
                totalTrades=metrics.get("total_trades", 0),
            ),
            completedAt=job.completed_at.isoformat() + "Z" if job.completed_at else datetime.utcnow().isoformat() + "Z",
        )

    def _parse_metrics_from_job(self, job: Job) -> dict[str, Any]:
        """Parse metrics from job output.

        Attempts to extract real metrics from job output lines.
        Returns default values if parsing fails.
        """
        metrics: dict[str, Any] = {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
        }

        # If job has stored result, use it
        if job.result:
            metrics.update({
                "total_return": job.result.get("total_return", 0.0),
                "sharpe_ratio": job.result.get("sharpe_ratio", 0.0),
                "max_drawdown": job.result.get("max_drawdown", 0.0),
                "win_rate": job.result.get("win_rate", 0.0),
                "total_trades": job.result.get("total_trades", 0),
            })
            return metrics

        # Try parsing from output lines
        for line in job.output_lines:
            line_lower = line.lower()
            try:
                if "total return" in line_lower or "total_return" in line_lower:
                    # Extract numeric value
                    for part in line.split():
                        part = part.strip("%").replace(",", "")
                        try:
                            val = float(part)
                            metrics["total_return"] = val / 100 if abs(val) > 1 else val
                            break
                        except ValueError:
                            continue
                elif "sharpe" in line_lower:
                    for part in line.split():
                        try:
                            metrics["sharpe_ratio"] = float(part)
                            break
                        except ValueError:
                            continue
                elif "drawdown" in line_lower:
                    for part in line.split():
                        part = part.strip("%").replace(",", "")
                        try:
                            val = float(part)
                            metrics["max_drawdown"] = -abs(val / 100 if abs(val) > 1 else val)
                            break
                        except ValueError:
                            continue
                elif "win rate" in line_lower or "win_rate" in line_lower:
                    for part in line.split():
                        part = part.strip("%").replace(",", "")
                        try:
                            val = float(part)
                            metrics["win_rate"] = val / 100 if val > 1 else val
                            break
                        except ValueError:
                            continue
                elif "total trades" in line_lower or "total_trades" in line_lower:
                    for part in line.split():
                        try:
                            metrics["total_trades"] = int(part)
                            break
                        except ValueError:
                            continue
            except Exception:
                continue

        return metrics

    def _job_to_result(self, job: Job) -> BacktestResultData:
        """Convert job to full result.

        Note: Uses parsed metrics from job output when available,
        otherwise returns zero values. Full implementation should
        parse detailed metrics from the backtest results file.
        """
        # Parse basic metrics from job
        parsed = self._parse_metrics_from_job(job)

        # Build metrics using parsed values where available
        total_trades = parsed.get("total_trades", 0)
        win_rate = parsed.get("win_rate", 0.0)
        winning_trades = int(total_trades * win_rate) if total_trades > 0 else 0
        losing_trades = total_trades - winning_trades

        metrics = BacktestMetrics(
            returns=ReturnMetrics(
                totalReturn=parsed.get("total_return", 0.0),
                annualizedReturn=parsed.get("total_return", 0.0) * 2,  # Rough estimate
                cagr=parsed.get("total_return", 0.0) * 2,
            ),
            risk=RiskMetrics(
                volatility=0.0,
                downsideVolatility=0.0,
                maxDrawdown=parsed.get("max_drawdown", 0.0),
                maxDrawdownDuration=0,
                var95=0.0,
                cvar95=0.0,
            ),
            riskAdjusted=RiskAdjustedMetrics(
                sharpeRatio=parsed.get("sharpe_ratio", 0.0),
                sortinoRatio=0.0,
                calmarRatio=0.0,
                omegaRatio=0.0,
            ),
            trade=TradeMetrics(
                totalTrades=total_trades,
                winningTrades=winning_trades,
                losingTrades=losing_trades,
                winRate=win_rate,
                profitFactor=0.0,
                avgTrade=0.0,
                avgWinner=0.0,
                avgLoser=0.0,
                largestWinner=0.0,
                largestLoser=0.0,
            ),
            distribution=DistributionMetrics(
                skewness=0.0,
                kurtosis=0.0,
                tailRatio=0.0,
            ),
        )

        return BacktestResultData(
            resultId=job.job_id,
            symbol=job.config.get("symbol", "EURUSD"),
            timeframe=job.config.get("timeframe", "1h"),
            config=job.config,
            metrics=metrics,
            timeSeries=None,  # TODO: Parse from results file
            trades=None,  # TODO: Parse from results file
            monteCarlo=None,  # TODO: Parse if enabled
            completedAt=job.completed_at.isoformat() + "Z" if job.completed_at else datetime.utcnow().isoformat() + "Z",
        )


backtest_service = BacktestService()
