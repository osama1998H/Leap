"""System API routes."""

from fastapi import APIRouter, Query

from ..schemas.system import (
    HealthResponse,
    MLflowLaunchResponse,
    MLflowStatusResponse,
    SystemMetricsResponse,
)
from ..services.system_service import system_service

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    health = system_service.get_health()
    return HealthResponse(data=health)


@router.get("/metrics/system", response_model=SystemMetricsResponse)
async def get_system_metrics():
    """Get system resource metrics."""
    metrics = system_service.get_metrics()
    return SystemMetricsResponse(data=metrics)


@router.get("/mlflow/status", response_model=MLflowStatusResponse)
async def get_mlflow_status(port: int = Query(default=5000, ge=1024, le=65535)):
    """Get MLflow server status."""
    status = system_service.get_mlflow_status(port=port)
    return MLflowStatusResponse(data=status)


@router.post("/mlflow/launch", response_model=MLflowLaunchResponse)
async def launch_mlflow(port: int = Query(default=5000, ge=1024, le=65535)):
    """Launch MLflow UI server."""
    result = system_service.launch_mlflow(port=port)
    return MLflowLaunchResponse(data=result)
