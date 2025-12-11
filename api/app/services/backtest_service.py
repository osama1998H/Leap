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
        """Convert job to result summary."""
        if job.status != JobStatus.COMPLETED:
            return None

        return BacktestResultSummaryData(
            resultId=job.job_id,
            symbol=job.config.get("symbol", "EURUSD"),
            timeframe=job.config.get("timeframe", "1h"),
            summary=BacktestResultSummary(
                totalReturn=0.12,  # Placeholder - would come from actual results
                sharpeRatio=1.5,
                maxDrawdown=-0.08,
                winRate=0.55,
                totalTrades=100,
            ),
            completedAt=job.completed_at.isoformat() + "Z" if job.completed_at else datetime.utcnow().isoformat() + "Z",
        )

    def _job_to_result(self, job: Job) -> BacktestResultData:
        """Convert job to full result."""
        # Generate sample metrics - in real implementation, parse from job output
        metrics = BacktestMetrics(
            returns=ReturnMetrics(
                totalReturn=0.124,
                annualizedReturn=0.285,
                cagr=0.285,
            ),
            risk=RiskMetrics(
                volatility=0.182,
                downsideVolatility=0.145,
                maxDrawdown=-0.082,
                maxDrawdownDuration=72,
                var95=-0.012,
                cvar95=-0.018,
            ),
            riskAdjusted=RiskAdjustedMetrics(
                sharpeRatio=1.85,
                sortinoRatio=2.12,
                calmarRatio=3.48,
                omegaRatio=1.65,
            ),
            trade=TradeMetrics(
                totalTrades=156,
                winningTrades=91,
                losingTrades=65,
                winRate=0.583,
                profitFactor=1.87,
                avgTrade=79.5,
                avgWinner=125.4,
                avgLoser=-87.2,
                largestWinner=485.0,
                largestLoser=-312.0,
            ),
            distribution=DistributionMetrics(
                skewness=0.34,
                kurtosis=2.15,
                tailRatio=1.25,
            ),
        )

        return BacktestResultData(
            resultId=job.job_id,
            symbol=job.config.get("symbol", "EURUSD"),
            timeframe=job.config.get("timeframe", "1h"),
            config=job.config,
            metrics=metrics,
            timeSeries=None,  # Would be populated from actual results
            trades=None,  # Would be populated from actual results
            monteCarlo=None,  # Would be populated if enabled
            completedAt=job.completed_at.isoformat() + "Z" if job.completed_at else datetime.utcnow().isoformat() + "Z",
        )


backtest_service = BacktestService()
