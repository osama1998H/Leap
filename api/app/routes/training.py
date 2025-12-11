"""Training API routes."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ..schemas.training import (
    TrainingJobListResponse,
    TrainingJobListData,
    TrainingJobResponse,
    TrainingLogsData,
    TrainingLogsResponse,
    TrainingStartRequest,
)
from ..services.training_service import training_service

router = APIRouter(prefix="/training", tags=["training"])


@router.post("/start", response_model=TrainingJobResponse, status_code=201)
async def start_training(request: TrainingStartRequest):
    """Start a new training job."""
    job_data = await training_service.start_training(request)
    return TrainingJobResponse(data=job_data)


@router.get("/jobs", response_model=TrainingJobListResponse)
async def list_training_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List training jobs."""
    try:
        jobs, total = await training_service.list_jobs(status, symbol, limit, offset)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return TrainingJobListResponse(
        data=TrainingJobListData(
            jobs=jobs,
            total=total,
            limit=limit,
            offset=offset,
        )
    )


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(job_id: str):
    """Get a specific training job."""
    job_data = await training_service.get_job(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return TrainingJobResponse(data=job_data)


@router.post("/jobs/{job_id}/stop", response_model=TrainingJobResponse)
async def stop_training_job(job_id: str):
    """Stop a running training job."""
    try:
        job_data = await training_service.stop_job(job_id)
        return TrainingJobResponse(data=job_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/jobs/{job_id}/pause", response_model=TrainingJobResponse)
async def pause_training_job(job_id: str):
    """Pause a running training job (sends SIGSTOP to process)."""
    try:
        job_data = await training_service.pause_job(job_id)
        return TrainingJobResponse(data=job_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/jobs/{job_id}/resume", response_model=TrainingJobResponse)
async def resume_training_job(job_id: str):
    """Resume a paused training job (sends SIGCONT to process)."""
    try:
        job_data = await training_service.resume_job(job_id)
        return TrainingJobResponse(data=job_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/jobs/{job_id}/logs", response_model=TrainingLogsResponse)
async def get_training_logs(
    job_id: str,
    level: Optional[str] = Query(None, description="Filter by log level"),
    since: Optional[datetime] = Query(None, description="Logs since timestamp"),
    limit: int = Query(100, ge=1, le=1000),
):
    """Get logs for a training job."""
    # Verify job exists first for consistent error handling
    job_data = await training_service.get_job(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    logs, has_more = training_service.get_job_logs(job_id, level, since, limit)
    return TrainingLogsResponse(
        data=TrainingLogsData(
            logs=logs,
            hasMore=has_more,
        )
    )
