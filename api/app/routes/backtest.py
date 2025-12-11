"""Backtest API routes."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ..schemas.backtest import (
    BacktestCompareRequest,
    BacktestCompareResponse,
    BacktestJobListData,
    BacktestJobListResponse,
    BacktestResultListData,
    BacktestResultListResponse,
    BacktestResultResponse,
    BacktestRunRequest,
)
from ..services.backtest_service import backtest_service

router = APIRouter(prefix="/backtest", tags=["backtest"])


@router.post("/run", status_code=201)
async def run_backtest(request: BacktestRunRequest):
    """Run a new backtest."""
    result = await backtest_service.run_backtest(request)
    return {"data": result}


@router.get("/jobs", response_model=BacktestJobListResponse)
async def list_backtest_jobs(
    status: Optional[str] = Query(None, description="Filter by status (pending, running, completed, failed)"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List backtest jobs (all statuses, not just completed)."""
    jobs, total = await backtest_service.list_jobs(status, limit, offset)
    return BacktestJobListResponse(
        data=BacktestJobListData(
            jobs=jobs,
            total=total,
        )
    )


@router.get("/results", response_model=BacktestResultListResponse)
async def list_backtest_results(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    min_sharpe: Optional[float] = Query(None, alias="minSharpe", description="Minimum Sharpe ratio"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List backtest results."""
    results, total = await backtest_service.list_results(symbol, min_sharpe, limit, offset)
    return BacktestResultListResponse(
        data=BacktestResultListData(
            results=results,
            total=total,
        )
    )


@router.get("/results/{result_id}", response_model=BacktestResultResponse)
async def get_backtest_result(result_id: str):
    """Get a specific backtest result."""
    result = await backtest_service.get_result(result_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Result {result_id} not found")
    return BacktestResultResponse(data=result)


@router.post("/compare", response_model=BacktestCompareResponse)
async def compare_backtests(request: BacktestCompareRequest):
    """Compare multiple backtest results."""
    comparison = await backtest_service.compare_results(request.result_ids)
    return BacktestCompareResponse(data=comparison)
